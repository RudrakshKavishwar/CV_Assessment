"""
Task 2: OCR for Yellow Text on Green Boxes
Pipeline: PaddleOCR full-image detection -> filter by yellow pixel overlap
"""

import os
os.environ['FLAGS_use_mkldnn'] = '0'

import cv2
import numpy as np
from paddleocr import PaddleOCR


class YellowTextOCR:

    def __init__(self, lang="en"):
        print("[YellowTextOCR] Loading PaddleOCR...")
        self.ocr = PaddleOCR(use_textline_orientation=True, lang=lang, enable_mkldnn=False)

        self.yellow_lower = np.array([12, 60, 40])
        self.yellow_upper = np.array([45, 255, 180])

        print("[YellowTextOCR] Ready.")

    def predict(self, image: np.ndarray) -> dict:
        visualized = image.copy()
        all_texts  = []

        # PaddleOCR's predict() accepts a file path OR numpy array
        results = self.ocr.predict(image)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)

        box_texts = []
        for res in results:
            polys  = res.get("rec_polys", res.get("dt_polys", []))
            texts  = res.get("rec_texts", [])
            scores = res.get("rec_scores", [])

            for poly, text, score in zip(polys, texts, scores):
                if not text.strip():
                    continue

                pts = np.array(poly, dtype=np.int32)
                x_min, y_min = max(0, pts[:, 0].min()), max(0, pts[:, 1].min())
                x_max = min(image.shape[1], pts[:, 0].max())
                y_max = min(image.shape[0], pts[:, 1].max())

                region_mask = yellow_mask[y_min:y_max, x_min:x_max]
                yellow_ratio = (region_mask > 0).sum() / max(1, region_mask.size)

                # Keep only text overlapping meaningfully with yellow pixels
                if yellow_ratio < 0.05:
                    continue

                box_texts.append({"text": text.strip(), "confidence": round(float(score), 4)})

                cv2.polylines(visualized, [pts], True, (0, 255, 255), 2)
                cv2.putText(visualized, f"{text.strip()} ({score:.2f})",
                            (pts[0][0], pts[0][1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        all_texts.extend([t["text"] for t in box_texts])

        return {
            "extracted_texts":  all_texts,
            "text_regions":     [{"coords": None, "texts": box_texts}],
            "visualized_image": visualized,
            "full_text":        " | ".join(all_texts),
        }