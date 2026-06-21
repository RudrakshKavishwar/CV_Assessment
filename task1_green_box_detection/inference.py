"""
Task 1: Green Box Detection — Inference class
"""

import torch
import cv2
import numpy as np
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T


class GreenBoxDetector:

    def __init__(self, model_path="models/green_box_detector.pth",
                 confidence_threshold=0.70, device=None):
        self.threshold = confidence_threshold
        self.device    = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.transform = T.Compose([T.ToTensor()])
        self.model     = self._load(model_path)
        print(f"[GreenBoxDetector] Ready on {self.device}")

    def _load(self, path):
        model   = fasterrcnn_resnet50_fpn(weights=None)
        in_feat = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes=2)
        ckpt    = torch.load(path, map_location=self.device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(self.device).eval()
        return model

    def predict(self, image: np.ndarray) -> dict:
        """
        Args:    image — BGR numpy array (cv2.imread)
        Returns: dict with boxes, scores, count, annotated_image
        """
        rgb    = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self.transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(tensor)[0]

        boxes  = out["boxes"].cpu().numpy()
        scores = out["scores"].cpu().numpy()

        keep   = scores >= self.threshold
        boxes  = boxes[keep]
        scores = scores[keep]

        return {
            "boxes":          boxes.tolist(),
            "scores":         scores.tolist(),
            "count":          len(boxes),
            "annotated_image": self._draw(image.copy(), boxes, scores),
        }

    def _draw(self, image, boxes, scores) -> np.ndarray:
        for (x1, y1, x2, y2), score in zip(boxes, scores):
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{score:.2f}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        label = f"Green Boxes: {len(boxes)}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)
        cv2.rectangle(image, (8, 8), (18 + tw, 22 + th), (0, 0, 0), -1)
        cv2.putText(image, label, (13, 14 + th),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
        return image