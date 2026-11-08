"""
Тестовые картинки с масками из сабмитов -> wandb Table. GPU не нужен.

python -m pet_seg.view \
    wandb://georgy-shulyndin-cu/pet_segmentation/segformer_b2_384-submission:latest \
    wandb://georgy-shulyndin-cu/pet_segmentation/unet_effv2s_384-submission:latest \
    --names segformer effv2s

В таблице каждая маска - отдельный слой, включается и выключается в UI.
Сортировка по agreement (IoU между сабмитами) по возрастанию -> сверху самые спорные картинки,
по area -> пустые и крошечные предсказания.
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import wandb

from pet_seg.data import find_data_root, find_paths, read_image


def rle_decode(rle, shape):
    # обратное к rle_encode: пары "start len", 1-based, column-major
    flat = np.zeros(shape[0] * shape[1], np.uint8)
    if rle != "EMPTY":
        s = np.array(rle.split(), dtype=np.int64)
        for start, length in zip(s[::2] - 1, s[1::2]):
            flat[start:start + length] = 1
    return flat.reshape(shape, order="F")


def resolve(spec):
    if spec.startswith("wandb://"):
        return Path(wandb.Api().artifact(spec[len("wandb://"):]).download()) / "submission.csv"
    return Path(spec)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("subs", nargs="+", help="submission.csv или wandb://entity/project/<name>-submission:latest")
    p.add_argument("--names", nargs="+")
    p.add_argument("--size", type=int, default=256, help="размер картинки в таблице")
    p.add_argument("--project", default="pet_segmentation")
    p.add_argument("--entity", default="georgy-shulyndin-cu")
    args = p.parse_args()

    names = args.names or [f"sub{i}" for i in range(len(args.subs))]
    subs = []
    for spec in args.subs:
        with open(resolve(spec)) as f:
            subs.append({int(r["ImageID"]): r["RLE"] for r in csv.DictReader(f)})
    for name, sub in zip(names, subs):
        print(f"{name}: {len(sub)} картинок, EMPTY {sum(v == 'EMPTY' for v in sub.values())}")

    paths = find_paths(find_data_root())
    size = (args.size, args.size)
    table = wandb.Table(columns=["id", "image", *[f"area_{n}" for n in names], "agreement"])
    for idx in sorted(subs[0]):
        image = read_image(paths["test_images"] / f"{idx}.png")
        masks = [rle_decode(sub[idx], image.shape[:2]) for sub in subs]
        layers = {
            n: {"mask_data": cv2.resize(m, size, interpolation=cv2.INTER_NEAREST), "class_labels": {1: "pet"}}
            for n, m in zip(names, masks)
        }
        union = np.logical_or.reduce(masks).sum()
        agreement = np.logical_and.reduce(masks).sum() / union if union else 1.0
        areas = [round(float(m.mean()), 4) for m in masks]
        table.add_data(idx, wandb.Image(cv2.resize(image, size, interpolation=cv2.INTER_AREA), masks=layers), *areas, round(float(agreement), 4))

    run = wandb.init(project=args.project, entity=args.entity, name="view_" + "_vs_".join(names), job_type="view")
    run.log({"test_view": table})
    run.finish()


if __name__ == "__main__":
    main()
