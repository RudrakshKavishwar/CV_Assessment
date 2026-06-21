"""
Task 1: Green Box Detection & Counting
Faster R-CNN with ResNet-50 FPN backbone — NO YOLO
Calibrated HSV range for olive-green ammo/military boxes
"""

import os
import json
import torch
import numpy as np
import cv2
import argparse
import logging
import time
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Calibrated HSV range (from real box samples) ────────────────────
GREEN_LOWER = np.array([55, 20, 60])
GREEN_UPPER = np.array([100, 90, 160])
MIN_BOX_AREA = 2500   # filters out noise specks, real boxes are much larger


# ── Dataset ────────────────────────────────────────────────────────

class GreenBoxDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transforms=None):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transforms = transforms
        self.files = sorted(
            f for f in self.images_dir.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        logger.info(f"Dataset: {len(self.files)} images from {images_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path   = self.files[idx]
        label_path = self.labels_dir / (img_path.stem + ".json")

        image = Image.open(img_path).convert("RGB")

        if label_path.exists():
            with open(label_path) as f:
                ann = json.load(f)
            boxes  = torch.as_tensor(ann["boxes"],  dtype=torch.float32)
            labels = torch.as_tensor(ann.get("labels", [1]*len(ann["boxes"])), dtype=torch.int64)
        else:
            boxes  = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) else torch.zeros(0)

        target = {
            "boxes":    boxes,
            "labels":   labels,
            "image_id": torch.tensor([idx]),
            "area":     area,
            "iscrowd":  torch.zeros(len(boxes), dtype=torch.int64),
        }

        if self.transforms:
            image = self.transforms(image)

        return image, target


def get_transforms(train=True):
    tfms = [T.ToTensor()]
    if train:
        tfms += [
            T.RandomHorizontalFlip(0.5),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        ]
    return T.Compose(tfms)


def collate_fn(batch):
    return tuple(zip(*batch))


# ── Model ──────────────────────────────────────────────────────────

def build_model(num_classes=2, pretrained=True):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model   = fasterrcnn_resnet50_fpn(weights=weights)
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)
    return model


# ── Auto-annotation (calibrated for olive-green ammo boxes) ────────

def auto_annotate(images_dir, labels_dir):
    """
    Generates label JSONs using calibrated HSV segmentation.
    Range tuned for dark olive-green military/ammo boxes.
    """
    os.makedirs(labels_dir, exist_ok=True)
    image_files = [
        f for f in Path(images_dir).iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png")
    ]
    logger.info(f"Auto-annotating {len(image_files)} images...")
    logger.info(f"HSV range: {GREEN_LOWER.tolist()} to {GREEN_UPPER.tolist()}")

    total_boxes = 0
    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"Could not read {img_path.name}, skipping")
            continue

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)

        # Close gaps from text/stencils, then remove small noise
        k_close = np.ones((15, 15), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
        k_open = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for cnt in contours:
            if cv2.contourArea(cnt) < MIN_BOX_AREA:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append([x, y, x+w, y+h])

        out = Path(labels_dir) / (img_path.stem + ".json")
        with open(out, "w") as f:
            json.dump({"boxes": boxes, "labels": [1]*len(boxes)}, f)

        total_boxes += len(boxes)
        logger.info(f"  {img_path.name}: {len(boxes)} boxes")

    avg = total_boxes / max(1, len(image_files))
    logger.info(f"Auto-annotation completed. Avg boxes/image: {avg:.1f}")


# ── Training loop ──────────────────────────────────────────────────

def train_one_epoch(model, optimizer, loader, device, epoch):
    model.train()
    total = 0
    epoch_start = time.time()

    for i, (images, targets) in enumerate(loader):
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss      = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()

        if (i + 1) % 5 == 0:
            logger.info(f"Epoch {epoch} | Step {i+1}/{len(loader)} | Loss: {loss.item():.4f}")

    elapsed = time.time() - epoch_start
    avg = total / len(loader)
    logger.info(f"Epoch {epoch} complete | Avg Loss: {avg:.4f} | Time: {elapsed:.1f}s")
    return avg


# ── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir",    type=str,   required=True)
    parser.add_argument("--labels_dir",    type=str,   required=True)
    parser.add_argument("--auto_annotate", action="store_true")
    parser.add_argument("--epochs",        type=int,   default=20)
    parser.add_argument("--batch_size",    type=int,   default=4)
    parser.add_argument("--lr",            type=float, default=0.005)
    parser.add_argument("--output_model",  type=str,   default="models/green_box_detector.pth")
    parser.add_argument("--resume_from",   type=str,   default=None,
                        help="Path to existing checkpoint to resume training from")
    args = parser.parse_args()

    if args.auto_annotate:
        auto_annotate(args.images_dir, args.labels_dir)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"=" * 50)
    logger.info(f"DEVICE: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("RUNNING ON CPU — this will be slow.")
    logger.info(f"=" * 50)

    dataset = GreenBoxDataset(args.images_dir, args.labels_dir, get_transforms(train=True))
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         num_workers=0, collate_fn=collate_fn)

    model = build_model(num_classes=2, pretrained=True).to(device)

    best = float("inf")

    # ── Resume from checkpoint if provided ──────────────────────
    if args.resume_from and os.path.exists(args.resume_from):
        logger.info(f"Resuming from checkpoint: {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        best = ckpt.get("loss", float("inf"))
        logger.info(f"Resumed with previous best loss: {best:.4f}")
    elif args.resume_from:
        logger.warning(f"--resume_from path not found: {args.resume_from}. Starting fresh instead.")

    params    = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    if args.resume_from and os.path.exists(args.resume_from):
        ckpt = torch.load(args.resume_from, map_location=device)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            logger.info("Optimizer state restored.")

    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, optimizer, loader, device, epoch)
        scheduler.step()
        if loss < best:
            best = loss
            torch.save({
                "epoch":               epoch,
                "model_state_dict":    model.state_dict(),
                "optimizer_state_dict":optimizer.state_dict(),
                "loss":                loss,
            }, args.output_model)
            logger.info(f"Best model saved (epoch {epoch}, loss={loss:.4f})")
        else:
            logger.info(f"Epoch {epoch} loss ({loss:.4f}) did not beat best ({best:.4f}) — not saved.")

    logger.info(f"Done. Final best loss: {best:.4f}. Model saved to {args.output_model}")


if __name__ == "__main__":
    main()