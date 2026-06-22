"""
wheel.py — Unified inference entry point
Usage:
    python wheel.py --image test.jpg --task detection
    python wheel.py --image test.jpg --task ocr
    python wheel.py --image test.jpg --task both
    python wheel.py --folder data/test_images --task detection
    python wheel.py --folder data/ocr_images --task ocr
"""

import argparse
import os
import cv2
import json
from pathlib import Path

from task1_green_box_detection.inference import GreenBoxDetector
from task2_ocr.inference import YellowTextOCR


def run_on_image(image_path, task, detector=None, ocr=None, output_dir="outputs"):
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Could not read: {image_path}")
        return

    img_name = Path(image_path).stem
    results = {"image": image_path}

    if task in ("detection", "both") and detector:
        print(f"[Detection] Running on {img_name}...")
        det = detector.predict(image)
        results["detection"] = {"boxes": det["boxes"], "scores": det["scores"], "count": det["count"]}
        out_path = os.path.join(output_dir, "detection", f"{img_name}_detected.jpg")
        cv2.imwrite(out_path, det["annotated_image"])
        print(f"  Boxes found: {det['count']} -> saved {out_path}")

    if task in ("ocr", "both") and ocr:
        print(f"[OCR] Running on {img_name}...")
        ocr_res = ocr.predict(image)
        results["ocr"] = {"texts": ocr_res["extracted_texts"], "full_text": ocr_res["full_text"]}
        out_img = os.path.join(output_dir, "ocr", f"{img_name}_ocr.jpg")
        out_txt = os.path.join(output_dir, "ocr", f"{img_name}_text.txt")
        cv2.imwrite(out_img, ocr_res["visualized_image"])
        with open(out_txt, "w") as f:
            f.write("\n".join(ocr_res["extracted_texts"]))
        print(f"  Texts: {ocr_res['extracted_texts']} -> saved {out_img}")

    json_path = os.path.join(output_dir, f"{img_name}_result.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Assessment Inference Wheel")
    parser.add_argument("--image",      type=str, help="Path to single image")
    parser.add_argument("--folder",     type=str, help="Path to folder of images")
    parser.add_argument("--task",       type=str, choices=["detection", "ocr", "both"], default="both")
    parser.add_argument("--det_model",  type=str, default="models/green_box_detector.pth")
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.output_dir, "detection"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "ocr"),       exist_ok=True)

    detector = GreenBoxDetector(model_path=args.det_model) if args.task in ("detection", "both") else None
    ocr      = YellowTextOCR()                             if args.task in ("ocr",       "both") else None

    if args.image:
        run_on_image(args.image, args.task, detector, ocr, args.output_dir)

    elif args.folder:
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        images = sorted(f for f in Path(args.folder).iterdir() if f.suffix.lower() in exts)
        print(f"Found {len(images)} images in {args.folder}")
        for img_path in images:
            run_on_image(str(img_path), args.task, detector, ocr, args.output_dir)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()