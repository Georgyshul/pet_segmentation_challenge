import csv
import itertools
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from pet_seg.data import TEST_SIZE, EvalDataset, TestDataset, find_data_root, find_paths, split_names
from pet_seg.model import build_model, image_iou, normalize

INFER_DEFAULTS = {
    "name": "ensemble",
    "members": [],
    "flip_tta": True,
    "threshold": 0.5,
    "refine": False,
    "refine_margin": 0.15,
    "refine_max_frac": 0.6,
    "postprocess": False,
    "min_component_frac": 0.1,
    "search": True,

    "search_threshold": True,
    "eval": True,

    "val_boundary": "fg",
    "batch_size": 8,
    "num_workers": 4,
    "data_root": None,
    "output_dir": "runs",
    "wandb": True,
    "project": "pet_segmentation",
    "entity": "georgy-shulyndin-cu",
    "submit": False,
    "message": "",
}


def rle_encode(mask: np.ndarray) -> str:
    """
    RLE (column-major / Fortran order) для бинарной 2D маски.
    Возвращает строку вида: "start1 len1 start2 len2 ..." с 1-based индексацией, как в Kaggle/COCO.
    """
    if mask.ndim != 2:
        raise ValueError("rle_encode expects a 2D mask")

    flat = np.ravel(mask.astype(np.uint8) > 0, order="F").astype(np.uint8) # Бинаризуем и разворачиваем по колонкам (column-major)
    if np.sum(flat) == 0: # для пустых масок, т.к. kaggle не даст загрузить пустой rle
        return "EMPTY"

    flat = np.concatenate(([0], flat, [0])) # На края добавляем нули, чтобы найти переходы 0->1 и 1->0
    changes = np.where(flat[1:] != flat[:-1])[0] + 1 # Получим индексы начала и конца серий, индексация с 1
    changes[1::2] -= changes[::2] # превратим пары [start, end] в пары [start, len]
    return " ".join(map(str, changes.tolist())) # засунем в строку, разделитель - пробел


def resolve_checkpoint(spec):
    spec = str(spec)
    if spec.startswith("wandb://"):
        import wandb
        path = Path(wandb.Api().artifact(spec[len("wandb://"):]).download())
    else:
        path = Path(spec)
    if path.is_dir():
        path = path / "best_model.pth"
    return path


def load_member(spec, device):
    path = resolve_checkpoint(spec)
    cfg = json.loads((path.parent / "config.json").read_text())
    model = build_model(cfg, weights=False)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.to(device).eval()
    return model, cfg


@torch.no_grad()
def predict_probs(model, images, sizes, flip=True, out_size=TEST_SIZE):
    x = normalize(images)
    total, n = 0, 0
    for size in sizes:
        xs = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        for flipped in ([False, True] if flip else [False]):
            inp = xs.flip(-1) if flipped else xs
            with torch.autocast("cuda", enabled=x.is_cuda):
                logits = model(inp).float()
            if flipped:
                logits = logits.flip(-1)
            logits = F.interpolate(logits, size=(out_size, out_size), mode="bilinear", align_corners=False)
            total = total + logits.sigmoid()[:, 0]
            n += 1
    return total / n


def ensemble_probs(members, weights, images, flip=True, out_size=TEST_SIZE, crop=False):
    probs = 0
    for member, w in zip(members, weights):
        if w == 0:
            continue
        sizes = [member["image_size"]] if crop else member["sizes"]
        probs = probs + w * predict_probs(member["model"], images, sizes, flip, out_size)
    return probs / sum(weights)


def refine(members, weights, image, prob, cfg):
    fg = prob > cfg["threshold"]
    if fg.sum() < 16:
        return prob
    ys, xs = torch.where(fg)
    y0, y1, x0, x1 = ys.min().item(), ys.max().item(), xs.min().item(), xs.max().item()
    full = prob.shape[0]
    side = int(max(y1 - y0, x1 - x0) * (1 + 2 * cfg["refine_margin"]))
    if side > cfg["refine_max_frac"] * full:
        return prob
    side = max(side, 64)
    top = min(max((y0 + y1) // 2 - side // 2, 0), full - side)
    left = min(max((x0 + x1) // 2 - side // 2, 0), full - side)
    crop = image[:, top:top + side, left:left + side].unsqueeze(0)
    crop_prob = ensemble_probs(members, weights, crop, cfg["flip_tta"], out_size=side, crop=True)[0]
    prob = prob.clone()
    prob[top:top + side, left:left + side] = (prob[top:top + side, left:left + side] + crop_prob) / 2
    return prob


def remove_small(mask, frac=0.1):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 2:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.where(areas >= frac * areas.max())[0] + 1
    return np.isin(labels, keep)


def iou_np(pred, gt):
    union = (pred | gt).sum()
    if union == 0:
        return 1.0
    return (pred & gt).sum() / union


@torch.no_grad()
def evaluate(members, weights, loader, cfg, device):
    thr = cfg["threshold"]
    ious = {"base": [], "base_post": [], "refine": [], "refine_post": []}
    for images, masks in tqdm(loader, desc="eval", leave=False):
        images = images.to(device)
        probs = ensemble_probs(members, weights, images, cfg["flip_tta"])
        refined = torch.stack([refine(members, weights, images[i], probs[i], cfg) for i in range(len(images))])
        valid = masks.numpy() != 255
        gt = masks.numpy() == 1
        for name, p in [("base", probs), ("refine", refined)]:
            pred = (p > thr).cpu().numpy()
            for i in range(len(pred)):
                ious[name].append(iou_np(pred[i] & valid[i], gt[i]))
                ious[name + "_post"].append(iou_np(remove_small(pred[i], cfg["min_component_frac"]) & valid[i], gt[i]))
    return {k: float(np.mean(v)) for k, v in ious.items()}


@torch.no_grad()
def predict_test(members, weights, loader, cfg, device):
    masks = {}
    for images, ids in tqdm(loader, desc="test"):
        images = images.to(device)
        probs = ensemble_probs(members, weights, images, cfg["flip_tta"])
        for i in range(len(images)):
            p = refine(members, weights, images[i], probs[i], cfg) if cfg["refine"] else probs[i]
            mask = (p > cfg["threshold"]).cpu().numpy()
            if cfg["postprocess"]:
                mask = remove_small(mask, cfg["min_component_frac"])
            masks[int(ids[i])] = mask
    return masks


def write_submission(masks, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ImageID", "RLE"])
        for idx in sorted(masks):
            writer.writerow([str(idx), rle_encode(masks[idx])])
    print(f"submission: {path} ({len(masks)} rows)")


def kaggle_submit(path, message):
    from pet_seg import data
    cmd = ["kaggle", "competitions", "submit", "-c", data.COMPETITION, "-f", str(path), "-m", message]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


@torch.no_grad()
def collect_low_res(member, loaders, flip, device, size=256):
    probs, gts = [], []
    for loader in loaders:
        for images, masks in tqdm(loader, desc="val probs", leave=False):
            p = predict_probs(member["model"], images.to(device), member["sizes"], flip)
            probs.append(F.interpolate(p[:, None], size=(size, size), mode="bilinear", align_corners=False)[:, 0].half())
            gts.append(F.interpolate(masks[:, None].float(), size=(size, size), mode="nearest")[:, 0].byte())
    return torch.cat(probs), torch.cat(gts).to(device)


def search_weights(probs, masks, n_val, thresholds, step=0.1):
    gt, valid = masks == 1, masks != 255
    grid = [w for w in itertools.product(np.arange(0, 1 + 1e-9, step), repeat=len(probs)) if abs(sum(w) - 1) < 1e-6]
    best = (-1.0, None, None)
    for w in tqdm(grid, desc="search", leave=False):
        p = sum(float(wi) * pi.float() for wi, pi in zip(w, probs) if wi > 0)
        for t in thresholds:
            iou = image_iou((p > t) & valid, gt)
            score = ((iou[:n_val].mean() + iou[n_val:].mean()) / 2).item()
            if score > best[0]:
                best = (score, [round(float(x), 2) for x in w], round(float(t), 2))
    return best


def run_inference(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(cfg["output_dir"]) / cfg["name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    members = []
    for m in cfg["members"]:
        model, mcfg = load_member(m["checkpoint"], device)
        members.append({
            "model": model,
            "name": mcfg["name"],
            "image_size": mcfg["image_size"],
            "sizes": m.get("sizes") or [mcfg["image_size"]],
        })
        print(f"{mcfg['name']}: {mcfg['arch']} {mcfg['encoder']} sizes={members[-1]['sizes']}")
    weights = [float(m.get("weight", 1.0)) for m in cfg["members"]]

    paths = find_paths(find_data_root(cfg["data_root"]))
    report = {}
    if cfg["eval"]:
        _, val_names = split_names(paths, 700)
        loaders = [
            DataLoader(EvalDataset(paths, val_names, small=small, boundary=cfg["val_boundary"]), batch_size=cfg["batch_size"], num_workers=cfg["num_workers"])
            for small in (False, True)
        ]
        if cfg["search"]:
            low = []
            for m in members:
                p, masks = collect_low_res(m, loaders, cfg["flip_tta"], device)
                low.append(p)
            n = len(val_names)
            for m, p in zip(members, low):
                iou = image_iou((p > 0.5) & (masks != 255), masks == 1)
                print(f"{m['name']}: val {iou[:n].mean():.4f} val_small {iou[n:].mean():.4f}")
            thresholds = np.arange(0.3, 0.71, 0.05) if cfg["search_threshold"] else [cfg["threshold"]]
            score, weights, thr = search_weights(low, masks, n, thresholds)
            cfg["threshold"] = thr
            print(f"search: weights={weights} threshold={thr} score(256px)={score:.4f}")
            del low
        for name, loader in zip(["val", "val_small"], loaders):
            report[name] = evaluate(members, weights, loader, cfg, device)
            print(name, {k: round(v, 4) for k, v in report[name].items()})

    loader = DataLoader(TestDataset(paths["test_images"]), batch_size=cfg["batch_size"], num_workers=cfg["num_workers"])
    masks = predict_test(members, weights, loader, cfg, device)
    sub_path = run_dir / "submission.csv"
    write_submission(masks, sub_path)

    saved = {k: v for k, v in cfg.items()}
    saved["weights"] = weights
    saved["report"] = report
    (run_dir / "inference.json").write_text(json.dumps(saved, indent=2, ensure_ascii=False))

    if cfg["wandb"]:
        import wandb
        run = wandb.init(project=cfg["project"], entity=cfg["entity"], name=cfg["name"], job_type="inference", config=saved)
        for split, scores in report.items():
            for k, v in scores.items():
                run.summary[f"{split}/{k}"] = v
        art = wandb.Artifact(f"{cfg['name']}-submission", type="submission")
        art.add_file(str(sub_path))
        art.add_file(str(run_dir / "inference.json"))
        run.log_artifact(art)
        run.finish()

    if cfg["submit"]:
        kaggle_submit(sub_path, cfg["message"] or cfg["name"])
    return sub_path
