"""
python -m pet_seg train --config configs/unet_convnext_tiny_384.yaml
python -m pet_seg train --config configs/unet_r34_256.yaml --no-wandb --set epochs=2 limit_batches=0.05
python -m pet_seg predict runs/unet_r34_256 --sizes 256 384 --refine
python -m pet_seg ensemble --config configs/ensemble.yaml --submit -m "ensemble v1"
python -m pet_seg submit runs/ens/submission.csv -m "message"
"""
import argparse
import os

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import yaml


def load_config(defaults, path=None, overrides=None):
    cfg = dict(defaults)
    if path:
        with open(path) as f:
            cfg.update(yaml.safe_load(f) or {})
    for key, value in (overrides or {}).items():
        if key not in defaults:
            raise KeyError(f"unknown config key: {key}")
        cfg[key] = value
    return cfg


def parse_set(items):
    out = {}
    for item in items or []:
        key, value = item.split("=", 1)
        out[key] = yaml.safe_load(value)
    return out


def common_overrides(args):
    out = parse_set(args.set)
    if args.no_wandb:
        out["wandb"] = False
    if args.submit:
        out["submit"] = True
    if args.message:
        out["message"] = args.message
    if args.data_root:
        out["data_root"] = args.data_root
    return out


def add_common(p):
    p.add_argument("--set", nargs="*", help="key=value, например --set lr=1e-4 mosaic_p=0")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--submit", action="store_true")
    p.add_argument("-m", "--message")
    p.add_argument("--data-root")


def main():
    parser = argparse.ArgumentParser(prog="pet_seg", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train")
    p.add_argument("--config")
    p.add_argument("--name")
    p.add_argument("--epochs", type=int)
    add_common(p)

    p = sub.add_parser("ensemble")
    p.add_argument("--config", required=True)
    add_common(p)

    p = sub.add_parser("predict", help="инференс одной модели")
    p.add_argument("checkpoint", help="директория запуска, путь к best_model.pth или wandb://entity/project/name:v0")
    p.add_argument("--name")
    p.add_argument("--sizes", type=int, nargs="+")
    p.add_argument("--refine", action="store_true")
    p.add_argument("--no-search", action="store_true")
    add_common(p)

    p = sub.add_parser("submit")
    p.add_argument("file")
    p.add_argument("-m", "--message", required=True)

    args = parser.parse_args()

    if args.command == "train":
        from pet_seg.train import DEFAULTS, train
        overrides = common_overrides(args)
        if args.name:
            overrides["name"] = args.name
        if args.epochs:
            overrides["epochs"] = args.epochs
        cfg = load_config(DEFAULTS, args.config, overrides)
        if not cfg["name"]:
            cfg["name"] = os.path.splitext(os.path.basename(args.config or "run"))[0]
        train(cfg)

    elif args.command == "ensemble":
        from pet_seg.infer import INFER_DEFAULTS, run_inference
        cfg = load_config(INFER_DEFAULTS, args.config, common_overrides(args))
        run_inference(cfg)

    elif args.command == "predict":
        from pet_seg.infer import INFER_DEFAULTS, run_inference
        overrides = common_overrides(args)
        member = {"checkpoint": args.checkpoint}
        if args.sizes:
            member["sizes"] = args.sizes
        overrides["members"] = [member]
        overrides["name"] = args.name or os.path.basename(os.path.normpath(args.checkpoint)).split(":")[0] + "_predict"
        if args.refine:
            overrides["refine"] = True
        if args.no_search:
            overrides["search"] = False
        cfg = load_config(INFER_DEFAULTS, None, overrides)
        run_inference(cfg)

    elif args.command == "submit":
        from pet_seg.infer import kaggle_submit
        kaggle_submit(args.file, args.message)


if __name__ == "__main__":
    main()
