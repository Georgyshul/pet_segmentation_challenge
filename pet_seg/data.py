import json
import os
import random
import subprocess
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

cv2.setNumThreads(0)

COMPETITION = "cu-cv-pet-segmentation-challenge-2026"
SLUGS = ["cu-cv-pet-segmentation-challenge-2026", "cu-cv-pet-segmentation-challenge"]
TEST_SIZE = 512


def _kaggle_credentials():
    username = os.environ.get("KAGGLE_USERNAME", "")
    key = os.environ.get("KAGGLE_KEY", "")
    if key.startswith("KGAT_") and not os.environ.get("KAGGLE_API_TOKEN"):
        os.environ["KAGGLE_API_TOKEN"] = key
    if not username or not key:
        print("KAGGLE_USERNAME или KAGGLE_KEY пустые, kaggle скачает данные только если уже залогинен")
        return
    folder = Path.home() / ".kaggle"
    folder.mkdir(exist_ok=True)
    path = folder / "kaggle.json"
    path.write_text(json.dumps({"username": username, "key": key}))
    path.chmod(0o600)


def _download_competition():
    _kaggle_credentials()
    dest = Path("data")
    dest.mkdir(exist_ok=True)
    cmd = ["kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(dest), "--unzip"]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    return dest


def find_data_root(data_root=None):
    candidates = [data_root, os.environ.get("PET_DATA"), "data"]
    for slug in SLUGS:
        candidates += [f"/kaggle/input/{slug}", f"/kaggle/input/competitions/{slug}"]
    for path in candidates:
        if path and Path(path).exists() and any(Path(path).rglob("trainval.txt")):
            return Path(path)
    return _download_competition()


def find_paths(root):
    root = Path(root)
    lists_dir = next(root.rglob("trainval.txt")).parent
    paths = {
        "trainval": lists_dir / "trainval.txt",
        "test_list": lists_dir / "test.txt",
        "trimaps": next(p for p in root.rglob("trimaps") if p.is_dir()),
        "images": next(root.rglob("*.jpg")).parent,
        "test_images": None,
    }
    for p in root.rglob("0.png"):
        if "trimaps" not in p.parts:
            paths["test_images"] = p.parent
            break
    return paths


def read_names(path):
    with open(path) as f:
        return [line.split()[0] for line in f if line.strip()]


def split_names(paths, val_size=700, seed=42):
    trainval = read_names(paths["trainval"])
    test = read_names(paths["test_list"])
    random.Random(seed).shuffle(test)
    val = sorted(test[:val_size])
    train = trainval + test[val_size:]
    return train, val


def read_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_mask(path, boundary="fg"):
    # 1 - животное, 2 - фон, 3 - граница
    trimap = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    mask = (trimap == 1).astype(np.uint8)
    if boundary == "fg":
        mask[trimap == 3] = 1
    elif boundary == "ignore":
        mask[trimap == 3] = 255
    return mask


def train_transforms(cfg):
    size = cfg["image_size"]
    geo = A.Compose([
        A.RandomResizedCrop(size=(size, size), scale=tuple(cfg["crop_scale"]), ratio=(0.75, 1.33)),
        A.HorizontalFlip(),
        A.Affine(
            scale=tuple(cfg["zoom_out_scale"]),
            translate_percent=(-0.15, 0.15),
            rotate=(-15, 15),
            border_mode=cv2.BORDER_REFLECT_101,
            p=cfg["zoom_out_p"],
        ),
    ])
    tile = A.Compose([
        A.RandomResizedCrop(size=(size // 2, size // 2), scale=(0.5, 1.0), ratio=(0.75, 1.33)),
        A.HorizontalFlip(),
    ])
    color = A.Compose([
        A.OneOf([
            A.RandomBrightnessContrast(0.25, 0.25),
            A.HueSaturationValue(10, 25, 20),
            A.ColorJitter(0.2, 0.2, 0.2, 0.05),
        ], p=cfg["color_p"]),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7)),
            A.MotionBlur(blur_limit=(3, 7)),
            A.GaussNoise(std_range=(0.02, 0.08)),
        ], p=0.3),
        A.ImageCompression(quality_range=(40, 95), p=0.3),
        A.ToGray(p=0.05),
    ])
    return geo, tile, color


class PetDataset(Dataset):
    def __init__(self, paths, names, cfg):
        self.paths = paths
        self.names = names
        self.boundary = cfg["boundary"]
        self.size = cfg["image_size"]
        self.mosaic_p = cfg["mosaic_p"]
        self.geo, self.tile, self.color = train_transforms(cfg)

    def __len__(self):
        return len(self.names)

    def load(self, idx):
        name = self.names[idx]
        image = read_image(self.paths["images"] / f"{name}.jpg")
        mask = read_mask(self.paths["trimaps"] / f"{name}.png", self.boundary)
        return image, mask

    def mosaic(self, idx):
        s = self.size // 2
        image = np.zeros((s * 2, s * 2, 3), np.uint8)
        mask = np.zeros((s * 2, s * 2), np.uint8)
        ids = [idx] + random.sample(range(len(self.names)), 3)
        random.shuffle(ids)
        for k, i in enumerate(ids):
            img, m = self.load(i)
            out = self.tile(image=img, mask=m)
            y, x = (k // 2) * s, (k % 2) * s
            image[y:y + s, x:x + s] = out["image"]
            mask[y:y + s, x:x + s] = out["mask"]
        return image, mask

    def __getitem__(self, idx):
        if random.random() < self.mosaic_p:
            image, mask = self.mosaic(idx)
        else:
            image, mask = self.load(idx)
            out = self.geo(image=image, mask=mask)
            image, mask = out["image"], out["mask"]
        image = self.color(image=image)["image"]
        image = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
        mask = torch.from_numpy(np.ascontiguousarray(mask)).long()
        return image, mask


def make_small(image, mask, rng, out_size=TEST_SIZE):
    small = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    hole = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    hole = cv2.dilate(hole, np.ones((15, 15), np.uint8))
    canvas = cv2.inpaint(small, hole, 7, cv2.INPAINT_TELEA)
    canvas = cv2.resize(canvas, (out_size, out_size), interpolation=cv2.INTER_CUBIC)
    canvas = cv2.GaussianBlur(canvas, (0, 0), 1.5)
    canvas_mask = np.zeros((out_size, out_size), np.uint8)

    h, w = mask.shape
    k = rng.uniform(0.3, 0.6) * out_size / max(h, w)
    nh, nw = max(int(h * k), 8), max(int(w * k), 8)
    y = rng.randint(0, out_size - nh)
    x = rng.randint(0, out_size - nw)
    pet = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA).astype(np.float32)
    ramp = max(min(nh, nw) * 0.15, 1)
    yy = np.minimum(np.arange(nh), np.arange(nh)[::-1])[:, None]
    xx = np.minimum(np.arange(nw), np.arange(nw)[::-1])[None, :]
    alpha = np.clip(np.minimum(yy, xx) / ramp, 0, 1)[..., None]
    region = canvas[y:y + nh, x:x + nw].astype(np.float32)
    canvas[y:y + nh, x:x + nw] = (alpha * pet + (1 - alpha) * region).astype(np.uint8)
    canvas_mask[y:y + nh, x:x + nw] = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    return canvas, canvas_mask


class EvalDataset(Dataset):
    def __init__(self, paths, names, small=False, boundary="fg"):
        self.paths = paths
        self.names = names
        self.small = small
        self.boundary = boundary
        self.cache_dir = Path(f"/tmp/pet_val_small_{boundary}")
        if small:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        cached = self.cache_dir / f"{name}.png"
        if self.small and cached.exists():
            data = cv2.imread(str(cached), cv2.IMREAD_UNCHANGED)
            image = cv2.cvtColor(data[..., :3], cv2.COLOR_BGR2RGB)
            return torch.from_numpy(image).permute(2, 0, 1), torch.from_numpy(data[..., 3].astype(np.int64))

        image = read_image(self.paths["images"] / f"{name}.jpg")
        mask = read_mask(self.paths["trimaps"] / f"{name}.png", self.boundary)
        if self.small:
            image, mask = make_small(image, mask, random.Random(name))
            cv2.imwrite(str(cached), np.dstack([cv2.cvtColor(image, cv2.COLOR_RGB2BGR), mask]))
        else:
            image = cv2.resize(image, (TEST_SIZE, TEST_SIZE), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (TEST_SIZE, TEST_SIZE), interpolation=cv2.INTER_NEAREST)
        image = torch.from_numpy(image).permute(2, 0, 1)
        return image, torch.from_numpy(mask).long()


class TestDataset(Dataset):
    def __init__(self, test_dir):
        self.test_dir = Path(test_dir)
        self.n = len(list(self.test_dir.glob("*.png")))

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        image = read_image(self.test_dir / f"{idx}.png")
        return torch.from_numpy(image).permute(2, 0, 1), idx
