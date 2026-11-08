import cv2
import lightning as L
import numpy as np
from lightning.pytorch.loggers import WandbLogger

from pet_seg.model import VAL_NAMES

TILE = 192


def overlay(image, mask, color):
    out = image.copy()
    out[mask] = (0.5 * out[mask] + 0.5 * np.array(color)).astype(np.uint8)
    return out


class WorstImages(L.Callback):
    """Раз в every эпох кидаем в wandb k картинок с худшим IoU: картинка | gt | предикт."""

    def __init__(self, every=3, k=8):
        self.every = every
        self.k = k
        self.items = []

    def _active(self, trainer):
        return (
            self.every > 0
            and not trainer.sanity_checking
            and isinstance(trainer.logger, WandbLogger)
            and (trainer.current_epoch + 1) % self.every == 0
        )

    def on_validation_epoch_start(self, trainer, pl_module):
        self.items = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not self._active(trainer):
            return
        images, masks = batch
        ious = outputs["iou"].tolist()
        for i, iou in enumerate(ious):
            if len(self.items) >= self.k and iou >= self.items[-1][0]:
                continue
            image = images[i].permute(1, 2, 0).cpu().numpy()
            image = cv2.resize(image, (TILE, TILE))
            gt = cv2.resize(masks[i].cpu().numpy().astype(np.uint8), (TILE, TILE), interpolation=cv2.INTER_NEAREST) == 1
            pred = cv2.resize(outputs["pred"][i].cpu().numpy().astype(np.uint8), (TILE, TILE), interpolation=cv2.INTER_NEAREST) > 0
            self.items.append((iou, VAL_NAMES[dataloader_idx], image, gt, pred))
            self.items.sort(key=lambda t: t[0])
            self.items = self.items[:self.k]

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self._active(trainer) or not self.items:
            return
        rows = []
        for iou, _, image, gt, pred in self.items:
            rows.append(np.concatenate([image, overlay(image, gt, (0, 255, 0)), overlay(image, pred, (255, 0, 0))], axis=1))
        grid = np.concatenate(rows, axis=0)
        caption = ", ".join(f"{name} {iou:.3f}" for iou, name, *_ in self.items)
        trainer.logger.log_image(key="worst_val", images=[grid], caption=[caption], step=trainer.global_step)
