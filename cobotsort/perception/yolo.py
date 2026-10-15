"""YOLO detector (Ultralytics) trained on the synthetic dataset."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

from cobotsort.config import CLASS_NAMES
from cobotsort.perception.detector import Detection, Detector

log = logging.getLogger(__name__)


class YoloDetector(Detector):
    """Runs on CPU by default (set ``COBOTSORT_YOLO_DEVICE=0`` to use a GPU)."""

    name = "yolo"

    def __init__(
        self,
        weights: Path | str,
        conf: float = 0.25,
        imgsz: int = 640,
        device: str | None = None,
    ) -> None:
        from ultralytics import YOLO

        weights = Path(weights)
        if not weights.exists():
            raise FileNotFoundError(f"{weights} not found: run `cobotsort train` first")
        self.model = YOLO(str(weights))
        self.conf = conf
        self.imgsz = imgsz
        self.device = device or os.getenv("COBOTSORT_YOLO_DEVICE", "cpu")
        names = self.model.names
        self.names = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
        if self.names != list(CLASS_NAMES):
            log.warning("model classes %s differ from %s", self.names, CLASS_NAMES)

    def detect(self, rgb: np.ndarray, conf: float | None = None) -> list[Detection]:
        bgr = np.ascontiguousarray(rgb[..., ::-1])  # Ultralytics expects OpenCV (BGR) arrays
        res = self.model.predict(
            bgr,
            imgsz=self.imgsz,
            conf=self.conf if conf is None else conf,
            device=self.device,
            verbose=False,
        )[0]
        boxes = res.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        scores = boxes.conf.cpu().numpy()
        return [
            Detection.from_box(self.names[c], s, b)
            for b, c, s in zip(xyxy, cls, scores, strict=True)
        ]
