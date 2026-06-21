"""
split_dataset.py — Randomly splits a single image folder into train/test sets
"""

import os
import shutil
import random
import argparse
from pathlib import Path


def split_dataset(source_dir, train_dir, test_dir, train_ratio=0.8, seed=42):
    source_dir = Path(source_dir)
    train_dir  = Path(train_dir)
    test_dir   = Path(test_dir)

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    exts = (".jpg", ".jpeg", ".png", ".bmp")
    images = [f for f in source_dir.iterdir() if f.suffix.lower() in exts]

    if not images:
        print(f"[ERROR] No images found in {source_dir}")
        return

    random.seed(seed)
    random.shuffle(images)

    split_idx = int(len(images) * train_ratio)
    train_images = images[:split_idx]
    test_images  = images[split_idx:]

    for img in train_images:
        shutil.copy(img, train_dir / img.name)

    for img in test_images:
        shutil.copy(img, test_dir / img.name)

    print(f"Total images : {len(images)}")
    print(f"Train images : {len(train_images)} -> {train_dir}")
    print(f"Test images  : {len(test_images)} -> {test_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=str, default="data/dataset")
    parser.add_argument("--train_dir",  type=str, default="data/train_images")
    parser.add_argument("--test_dir",   type=str, default="data/test_images")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    args = parser.parse_args()

    split_dataset(args.source_dir, args.train_dir, args.test_dir, args.train_ratio)