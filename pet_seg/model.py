import math

import lightning as L
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MeanMetric

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
VAL_NAMES = ["val", "val_small"]


def normalize(x):
    x = x.float() / 255.0
    mean = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def build_model(cfg, weights=True):
    return smp.create_model(
        cfg["arch"],
        encoder_name=cfg["encoder"],
        encoder_weights=cfg["encoder_weights"] if weights else None,
        classes=1,
    )


class BCEDice(nn.Module):
    def __init__(self, second="dice", bce_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        if second == "lovasz":
            self.second = smp.losses.LovaszLoss("binary", per_image=True, ignore_index=255)
        else:
            self.second = smp.losses.DiceLoss("binary", from_logits=True, ignore_index=255)

    def forward(self, logits, target):
        valid = (target != 255).float()
        bce = F.binary_cross_entropy_with_logits(logits, target.clamp(0, 1).float(), reduction="none")
        bce = (bce * valid).sum() / valid.sum().clamp(min=1)
        return self.bce_weight * bce + (1 - self.bce_weight) * self.second(logits, target)


def image_iou(pred, target):
    inter = (pred & target).flatten(1).sum(1).float()
    union = (pred | target).flatten(1).sum(1).float()
    return torch.where(union > 0, inter / union.clamp(min=1), torch.ones_like(union))


class SegModule(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.model = build_model(cfg)
        self.loss_fn = BCEDice("lovasz" if cfg["loss"] == "bce_lovasz" else "dice")
        self.val_iou = nn.ModuleList([MeanMetric() for _ in VAL_NAMES])
        self.val_miou = nn.ModuleList([MeanMetric() for _ in VAL_NAMES])

    def forward(self, images):
        return self.model(normalize(images))

    def training_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        loss = self.loss_fn(logits, masks.unsqueeze(1))
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        images, masks = batch
        size = self.cfg["image_size"]
        x = F.interpolate(images.float(), size=(size, size), mode="bilinear", align_corners=False)
        logits = self(x)
        logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        valid = masks != 255
        pred = (logits[:, 0] > 0) & valid
        gt = masks == 1
        iou = image_iou(pred, gt)
        miou = (iou + image_iou(~pred & valid, ~gt & valid)) / 2
        self.val_iou[dataloader_idx].update(iou)
        self.val_miou[dataloader_idx].update(miou)
        return {"iou": iou, "pred": pred}

    def on_validation_epoch_end(self):
        scores = []
        for i, name in enumerate(VAL_NAMES):
            if self.val_iou[i].update_count == 0:
                continue
            iou = self.val_iou[i].compute()
            self.log(f"{name}_iou", iou, prog_bar=True)
            self.log(f"{name}_miou", self.val_miou[i].compute())
            scores.append(iou)
            self.val_iou[i].reset()
            self.val_miou[i].reset()
        if scores:
            self.log("val_score", sum(scores) / len(scores), prog_bar=True)

    def configure_optimizers(self):
        cfg = self.cfg
        encoder = list(self.model.encoder.parameters())
        encoder_ids = {id(p) for p in encoder}
        rest = [p for p in self.model.parameters() if id(p) not in encoder_ids]
        optimizer = torch.optim.AdamW([
            {"params": encoder, "lr": cfg["lr"] * cfg["encoder_lr_mult"]},
            {"params": rest, "lr": cfg["lr"]},
        ], weight_decay=cfg["weight_decay"])

        total = self.trainer.estimated_stepping_batches
        warmup = int(total * cfg["warmup_epochs"] / cfg["epochs"])

        def lr_lambda(step):
            if step < warmup:
                return (step + 1) / warmup
            progress = (step - warmup) / max(1, total - warmup)
            return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
