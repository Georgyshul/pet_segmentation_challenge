import json
import time
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from pet_seg.callbacks import WorstImages
from pet_seg.data import EvalDataset, PetDataset, TestDataset, find_data_root, find_paths, split_names
from pet_seg.infer import kaggle_submit, predict_test, resolve_checkpoint, write_submission
from pet_seg.model import SegModule

DEFAULTS = {
    "name": None,
    "arch": "unet",
    "encoder": "resnet34",
    "encoder_weights": "imagenet",
    "init_checkpoint": None,

    "image_size": 384,
    "epochs": 30,
    "batch_size": 16,
    "lr": 5e-4,
    "encoder_lr_mult": 0.2,
    "weight_decay": 1e-4,
    "warmup_epochs": 1,
    "grad_clip": 1.0,
    "loss": "bce_dice",
    "boundary": "fg",
    "val_boundary": "fg",
    "precision": "16-mixed",
    "patience": 6,

    "crop_scale": [0.4, 1.0],
    "zoom_out_p": 0.3,
    "zoom_out_scale": [0.35, 1.0],
    "mosaic_p": 0.2,
    "color_p": 0.7,

    "val_size": 700,
    "limit_batches": 1.0,
    "num_workers": 4,
    "seed": 42,
    "data_root": None,
    "output_dir": "runs",

    "skip_test": False,
    "tta_sizes": None,
    "flip_tta": True,
    "threshold": 0.5,
    "refine": False,
    "refine_margin": 0.15,
    "refine_max_frac": 0.6,
    "postprocess": False,
    "min_component_frac": 0.1,
    "submit": False,
    "message": "",

    "wandb": True,
    "project": "pet_segmentation",
    "entity": "georgy-shulyndin-cu",
    "worst_every": 3,
}


def train(cfg):
    start = time.time()
    L.seed_everything(cfg["seed"], workers=True)
    run_dir = Path(cfg["output_dir"]) / cfg["name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    paths = find_paths(find_data_root(cfg["data_root"]))
    print({k: str(v) for k, v in paths.items()})
    train_names, val_names = split_names(paths, cfg["val_size"])
    print(f"train: {len(train_names)}, val: {len(val_names)}")

    nw = cfg["num_workers"]
    train_loader = DataLoader(
        PetDataset(paths, train_names, cfg),
        batch_size=cfg["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=nw,
        pin_memory=True,
        persistent_workers=nw > 0,
    )
    val_loaders = None
    if val_names:
        val_loaders = [
            DataLoader(EvalDataset(paths, val_names, small=small, boundary=cfg["val_boundary"]), batch_size=cfg["batch_size"], num_workers=nw, pin_memory=True, persistent_workers=nw > 0)
            for small in (False, True)
        ]

    module = SegModule(cfg)
    if cfg["init_checkpoint"]:
        path = resolve_checkpoint(cfg["init_checkpoint"])
        module.model.load_state_dict(torch.load(path, map_location="cpu"))
        print(f"init from {path}")

    logger = False
    callbacks = [WorstImages(cfg["worst_every"]), TQDMProgressBar(refresh_rate=50)]
    if cfg["wandb"]:
        logger = WandbLogger(project=cfg["project"], entity=cfg["entity"], name=cfg["name"])
        callbacks.append(LearningRateMonitor("step"))
    if val_loaders:
        checkpoint = ModelCheckpoint(dirpath=run_dir, filename="best", monitor="val_score", mode="max", save_weights_only=True)
        callbacks += [checkpoint, EarlyStopping("val_score", mode="max", patience=cfg["patience"])]
    else:
        checkpoint = ModelCheckpoint(dirpath=run_dir, filename="last", save_weights_only=True)
        callbacks.append(checkpoint)

    cuda = torch.cuda.is_available()
    trainer = L.Trainer(
        max_epochs=cfg["epochs"],
        accelerator="gpu" if cuda else "cpu",
        devices=1,
        precision=cfg["precision"] if cuda else "32-true",
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=cfg["grad_clip"],
        log_every_n_steps=10,
        limit_train_batches=cfg["limit_batches"],
        limit_val_batches=cfg["limit_batches"],
    )
    trainer.fit(module, train_loader, val_loaders)

    ckpt = torch.load(checkpoint.best_model_path, map_location="cpu", weights_only=False)
    state = {k[len("model."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    torch.save(state, run_dir / "best_model.pth")
    best_score = checkpoint.best_model_score
    saved = dict(cfg)
    saved["best_epoch"] = ckpt["epoch"] + 1
    saved["best_val_score"] = float(best_score) if best_score is not None else None
    saved["train_minutes"] = round((time.time() - start) / 60, 1)
    (run_dir / "config.json").write_text(json.dumps(saved, indent=2, ensure_ascii=False))
    print(f"best epoch {saved['best_epoch']}, val_score {saved['best_val_score']}, {saved['train_minutes']} min")

    if logger:
        import wandb
        art = wandb.Artifact(cfg["name"], type="model", metadata=saved)
        art.add_file(str(run_dir / "best_model.pth"))
        art.add_file(str(run_dir / "config.json"))
        logger.experiment.log_artifact(art)
        logger.experiment.summary["best_epoch"] = saved["best_epoch"]

    if not cfg["skip_test"] and paths["test_images"] is not None:
        device = torch.device("cuda" if cuda else "cpu")
        model = module.model
        model.load_state_dict(state)
        model.to(device).eval()
        members = [{"model": model, "image_size": cfg["image_size"], "sizes": cfg["tta_sizes"] or [cfg["image_size"]]}]
        loader = DataLoader(TestDataset(paths["test_images"]), batch_size=8, num_workers=nw)
        masks = predict_test(members, [1.0], loader, cfg, device)
        sub_path = run_dir / "submission.csv"
        write_submission(masks, sub_path)
        if logger:
            art = wandb.Artifact(f"{cfg['name']}-submission", type="submission")
            art.add_file(str(sub_path))
            logger.experiment.log_artifact(art)
        if cfg["submit"]:
            kaggle_submit(sub_path, cfg["message"] or cfg["name"])

    if logger:
        wandb.finish()
    return run_dir
