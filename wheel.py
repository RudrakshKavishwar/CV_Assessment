"""
wheel.py — Unified inference entry point for Green Box Detection
Usage:
    python wheel.py --image path/to/image.jpg --task detection
    python wheel.py --folder path/to/folder/ --task detection
"""

import argparse
import os
import cv2
import json
from pathlib import Path

from task1_green_box_detection.inference import GreenBoxDetector


def run_on_image(image_path, detector, output_dir="outputs"):
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Could not read: {image_path}")
        return

    img_name = Path(image_path).stem
    print(f"[Detection] Running on {img_name}...")
    result = detector.predict(image)

    out_path = os.path.join(output_dir, "detection", f"{img_name}_detected.jpg")
    cv2.imwrite(out_path, result["annotated_image"])
    print(f"  Boxes found: {result['count']} -> saved {out_path}")

    json_path = os.path.join(output_dir, f"{img_name}_result.json")
    with open(json_path, "w") as f:
        json.dump({
            "image": image_path,
            "boxes": result["boxes"],
            "scores": result["scores"],
            "count": result["count"]
        }, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(description="Green Box Detection Inference Wheel")
    parser.add_argument("--image",      type=str, help="Path to single image")
    parser.add_argument("--folder",     type=str, help="Path to folder of images")
    parser.add_argument("--task",       type=str, choices=["detection"], default="detection")
    parser.add_argument("--det_model",  type=str, default="models/green_box_detector.pth")
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.output_dir, "detection"), exist_ok=True)

    detector = GreenBoxDetector(model_path=args.det_model)

    if args.image:
        run_on_image(args.image, detector, args.output_dir)

    elif args.folder:
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        images = sorted(f for f in Path(args.folder).iterdir() if f.suffix.lower() in exts)
        print(f"Found {len(images)} images in {args.folder}")
        for img_path in images:
            run_on_image(str(img_path), detector, args.output_dir)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()