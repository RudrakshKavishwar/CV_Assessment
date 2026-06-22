"""
Task 2: OCR Evaluation — Character-Level & Word-Level Accuracy
"""

import os
import json
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from difflib import SequenceMatcher
import re

from task2_ocr.inference import YellowTextOCR


# ── Normalization ──────────────────────────────────────────────────

def normalize(text: str) -> str:
    """Normalize text for fair comparison:
    - lowercase
    - remove pipe separators we inject
    - collapse whitespace
    - remove punctuation differences (spaces around hyphens etc.)
    """
    text = text.lower()
    text = text.replace("|", " ")
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


# ── Metrics ────────────────────────────────────────────────────────

def char_accuracy(pred: str, gt: str) -> float:
    pred = normalize(pred)
    gt   = normalize(gt)
    if not gt:
        return 1.0 if not pred else 0.0
    return SequenceMatcher(None, pred, gt).ratio()


def word_accuracy(pred: str, gt: str) -> float:
    pred_w = normalize(pred).split()
    gt_w   = normalize(gt).split()
    if not gt_w:
        return 1.0 if not pred_w else 0.0
    correct = sum(1 for w in pred_w if w in gt_w)
    return correct / len(gt_w)


# ── Main ───────────────────────────────────────────────────────────

def evaluate(images_dir, gt_dir, output_dir="outputs/ocr_evaluation"):
    os.makedirs(output_dir, exist_ok=True)
    ocr = YellowTextOCR()

    files = sorted(
        f for f in Path(images_dir).iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    char_scores = []
    word_scores = []
    log         = []

    for img_path in files:
        gt_path = Path(gt_dir) / (img_path.stem + ".txt")
        if not gt_path.exists():
            print(f"  [SKIP] No GT for {img_path.name}")
            continue

        gt_text   = gt_path.read_text(encoding="utf-8").strip()
        image     = cv2.imread(str(img_path))
        result    = ocr.predict(image)
        pred_text = result["full_text"]

        ca = char_accuracy(pred_text, gt_text)
        wa = word_accuracy(pred_text, gt_text)
        char_scores.append(ca)
        word_scores.append(wa)

        log.append({
            "image":          img_path.name,
            "predicted":      pred_text,
            "ground_truth":   gt_text,
            "char_accuracy":  round(ca, 4),
            "word_accuracy":  round(wa, 4),
        })

        print(f"  {img_path.name}: char={ca:.2f} word={wa:.2f}")
        print(f"    Pred: {pred_text[:80]}")
        print(f"    GT:   {gt_text[:80]}")

        vis_path = os.path.join(output_dir, f"{img_path.stem}_ocr_eval.jpg")
        cv2.imwrite(vis_path, result["visualized_image"])

    avg_char = float(np.mean(char_scores)) if char_scores else 0
    avg_word = float(np.mean(word_scores)) if word_scores else 0

    print("\n====== OCR EVALUATION ======")
    print(f"  Images evaluated   : {len(char_scores)}")
    print(f"  Char Accuracy (avg): {avg_char:.4f}  ({avg_char*100:.1f}%)")
    print(f"  Word Accuracy (avg): {avg_word:.4f}  ({avg_word*100:.1f}%)")
    print("============================\n")

    summary = {
        "avg_char_accuracy": round(avg_char, 4),
        "avg_word_accuracy": round(avg_word, 4),
        "per_image":         log,
    }
    with open(os.path.join(output_dir, "ocr_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    plt.figure(figsize=(5, 4))
    bars = plt.bar(["Char Accuracy", "Word Accuracy"], [avg_char, avg_word],
                   color=["#4CAF50", "#2196F3"])
    for bar, v in zip(bars, [avg_char, avg_word]):
        plt.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                 f"{v*100:.1f}%", ha="center", fontsize=12)
    plt.ylim(0, 1.15)
    plt.title("OCR Accuracy Metrics")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ocr_metrics_chart.png"))
    plt.close()

    print(f"Saved to: {output_dir}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--gt_dir",     type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/ocr_evaluation")
    args = parser.parse_args()
    evaluate(args.images_dir, args.gt_dir, args.output_dir)