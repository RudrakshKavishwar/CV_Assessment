"""
Task 1: Evaluation — Precision, Recall, F1, mAP, Confusion Matrix
"""

import os
import json
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from task1_green_box_detection.inference import GreenBoxDetector


# ── IoU ────────────────────────────────────────────────────────────

def iou(a, b):
    xi1 = max(a[0], b[0]); yi1 = max(a[1], b[1])
    xi2 = min(a[2], b[2]); yi2 = min(a[3], b[3])
    inter = max(0, xi2-xi1) * max(0, yi2-yi1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0


def match(pred_boxes, gt_boxes, iou_thresh=0.5):
    matched = set()
    tp = fp = 0
    for pred in pred_boxes:
        best_iou, best_i = 0, -1
        for i, gt in enumerate(gt_boxes):
            if i in matched:
                continue
            v = iou(pred, gt)
            if v > best_iou:
                best_iou, best_i = v, i
        if best_iou >= iou_thresh and best_i not in matched:
            tp += 1
            matched.add(best_i)
        else:
            fp += 1
    fn = len(gt_boxes) - len(matched)
    return tp, fp, fn


# ── mAP calculation ──────────────────────────────────────────────

def compute_average_precision(all_preds, all_gts, iou_thresh=0.5):
    """
    Computes Average Precision (AP) at a single IoU threshold across the
    entire dataset, using the standard 11-point interpolation method.

    all_preds: list of (boxes, scores) per image
    all_gts:   list of gt_boxes per image
    """
    flat_preds = []
    for img_idx, (boxes, scores) in enumerate(all_preds):
        for box, score in zip(boxes, scores):
            flat_preds.append((img_idx, box, score))
    flat_preds.sort(key=lambda x: x[2], reverse=True)

    total_gt = sum(len(gts) for gts in all_gts)
    if total_gt == 0 or len(flat_preds) == 0:
        return 0.0

    matched_gt_per_image = {i: set() for i in range(len(all_gts))}
    tp_list = []
    fp_list = []

    for img_idx, pred_box, score in flat_preds:
        gt_boxes = all_gts[img_idx]
        matched = matched_gt_per_image[img_idx]

        best_iou, best_i = 0, -1
        for i, gt in enumerate(gt_boxes):
            if i in matched:
                continue
            v = iou(pred_box, gt)
            if v > best_iou:
                best_iou, best_i = v, i

        if best_iou >= iou_thresh and best_i not in matched:
            tp_list.append(1)
            fp_list.append(0)
            matched.add(best_i)
        else:
            tp_list.append(0)
            fp_list.append(1)

    tp_cum = np.cumsum(tp_list)
    fp_cum = np.cumsum(fp_list)

    recalls    = tp_cum / total_gt
    precisions = tp_cum / (tp_cum + fp_cum)

    # 11-point interpolation (Pascal VOC style)
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        prec_at_recall = [p for p, r in zip(precisions, recalls) if r >= t]
        ap += (max(prec_at_recall) if prec_at_recall else 0.0)
    ap /= 11
    return ap


def compute_map(all_preds, all_gts, iou_thresholds=None):
    """
    Computes mAP averaged over multiple IoU thresholds (COCO-style mAP@[0.5:0.95])
    plus the standard single-threshold mAP@0.5 (Pascal VOC style).
    """
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05)  # 0.5, 0.55, ..., 0.95

    ap_per_threshold = {}
    for t in iou_thresholds:
        ap = compute_average_precision(all_preds, all_gts, iou_thresh=round(t, 2))
        ap_per_threshold[round(t, 2)] = ap

    map_50_95 = float(np.mean(list(ap_per_threshold.values())))
    map_50    = ap_per_threshold.get(0.5, 0.0)

    return map_50, map_50_95, ap_per_threshold


# ── Main ───────────────────────────────────────────────────────────

def evaluate(model_path, images_dir, labels_dir,
             output_dir="outputs/evaluation", iou_thresh=0.5):

    os.makedirs(output_dir, exist_ok=True)
    detector = GreenBoxDetector(model_path=model_path)

    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    files = sorted(f for f in images_dir.iterdir()
                   if f.suffix.lower() in (".jpg", ".jpeg", ".png"))

    total_tp = total_fp = total_fn = 0
    count_errors = []
    gt_counts = []

    all_preds = []   # list of (boxes, scores) for mAP calculation
    all_gts   = []   # list of gt_boxes for mAP calculation

    for img_path in files:
        label_path = labels_dir / (img_path.stem + ".json")
        if not label_path.exists():
            continue

        with open(label_path) as f:
            ann = json.load(f)
        gt_boxes = ann["boxes"]

        image  = cv2.imread(str(img_path))
        result = detector.predict(image)

        tp, fp, fn = match(result["boxes"], gt_boxes, iou_thresh)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        count_errors.append(abs(result["count"] - len(gt_boxes)))
        gt_counts.append(len(gt_boxes))

        all_preds.append((result["boxes"], result["scores"]))
        all_gts.append(gt_boxes)

    # ── Standard metrics ─────────────────────────────────────────
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
    f1        = 2*precision*recall / (precision+recall) if (precision+recall) else 0
    avg_gt    = np.mean(gt_counts) if gt_counts else 1
    count_acc = 1 - (np.mean(count_errors) / max(1, avg_gt)) if count_errors else 0

    # ── mAP ──────────────────────────────────────────────────────
    print("\nComputing mAP across IoU thresholds 0.5-0.95...")
    map_50, map_50_95, ap_per_threshold = compute_map(all_preds, all_gts)

    print("\n====== DETECTION EVALUATION ======")
    print(f"  Precision         : {precision:.4f}")
    print(f"  Recall            : {recall:.4f}")
    print(f"  F1 Score          : {f1:.4f}")
    print(f"  Counting Accuracy : {count_acc:.4f}")
    print(f"  mAP@0.5           : {map_50:.4f}")
    print(f"  mAP@0.5:0.95      : {map_50_95:.4f}")
    print(f"  TP={total_tp}  FP={total_fp}  FN={total_fn}")
    print("===================================\n")

    metrics = {
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1_score":  round(f1, 4),
        "counting_accuracy": round(count_acc, 4),
        "mAP_50":    round(map_50, 4),
        "mAP_50_95": round(map_50_95, 4),
        "AP_per_iou_threshold": {str(k): round(v, 4) for k, v in ap_per_threshold.items()},
        "TP": total_tp, "FP": total_fp, "FN": total_fn,
    }
    with open(os.path.join(output_dir, "detection_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Confusion matrix ───────────────────────────────────────────
    cm = np.array([[total_tp, total_fn], [total_fp, 0]])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred +", "Pred -"],
                yticklabels=["GT +",   "GT -"])
    plt.title("Confusion Matrix — Green Box Detection")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    # ── Metrics bar chart (now includes mAP) ───────────────────────
    plt.figure(figsize=(7, 4))
    keys = ["Precision", "Recall", "F1", "mAP@0.5", "mAP@.5:.95"]
    vals = [precision, recall, f1, map_50, map_50_95]
    bars = plt.bar(keys, vals, color=["#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#E91E63"])
    for bar, v in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    plt.ylim(0, 1.15)
    plt.title("Detection Metrics (incl. mAP)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_chart.png"))
    plt.close()

    # ── mAP vs IoU threshold chart ──────────────────────────────────
    plt.figure(figsize=(7, 4))
    thresholds = list(ap_per_threshold.keys())
    ap_values  = list(ap_per_threshold.values())
    plt.plot(thresholds, ap_values, marker="o", color="#9C27B0")
    plt.xlabel("IoU Threshold")
    plt.ylabel("Average Precision (AP)")
    plt.title("AP vs IoU Threshold")
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "map_curve.png"))
    plt.close()

    print(f"Saved to: {output_dir}")
    print(f"  - detection_metrics.json")
    print(f"  - confusion_matrix.png")
    print(f"  - metrics_chart.png")
    print(f"  - map_curve.png")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",    type=str, default="models/green_box_detector.pth")
    parser.add_argument("--images_dir",    type=str, required=True)
    parser.add_argument("--labels_dir",    type=str, required=True)
    parser.add_argument("--output_dir",    type=str, default="outputs/evaluation")
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    args = parser.parse_args()
    evaluate(args.model_path, args.images_dir, args.labels_dir,
             args.output_dir, args.iou_threshold)