"""Classical baseline: HSV thresholds, morphology, contours, area filter, minAreaRect.

OpenCV hue runs 0-179 (degrees / 2). Nominal hues: red 0, green 65, blue 114, purple 142.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from cobotsort.perception.detector import Detection, Detector

# (hue ranges, min saturation, min value) per class.
DEFAULT_RANGES: dict[str, tuple[list[tuple[int, int]], int, int]] = {
    "red_cube": ([(0, 9), (171, 179)], 110, 50),
    "green_cube": ([(45, 88)], 100, 40),
    "blue_cube": ([(103, 127)], 120, 35),
    "defect": ([(130, 162)], 80, 35),
}


@dataclass
class HSVDetector(Detector):
    """Color segmentation per class; one detection per blob of plausible size."""

    ranges: dict[str, tuple[list[tuple[int, int]], int, int]] = field(
        default_factory=lambda: dict(DEFAULT_RANGES)
    )
    min_area: int = 120
    max_area: int = 4000
    kernel: int = 3
    name: str = "hsv"

    def masks(self, rgb: np.ndarray) -> dict[str, np.ndarray]:
        hsv = cv2.cvtColor(cv2.GaussianBlur(rgb, (3, 3), 0), cv2.COLOR_RGB2HSV)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (self.kernel, self.kernel))
        out = {}
        for cls, (hues, s_min, v_min) in self.ranges.items():
            mask = np.zeros(hsv.shape[:2], np.uint8)
            for lo, hi in hues:
                mask |= cv2.inRange(hsv, (lo, s_min, v_min), (hi, 255, 255))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
            out[cls] = mask
        return out

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        dets = []
        for cls, mask in self.masks(rgb).items():
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if not self.min_area <= area <= self.max_area:
                    continue
                (cx, cy), (w, h), _ = cv2.minAreaRect(c)
                rect_area = max(w * h, 1.0)
                # Heuristic score: cube blobs are compact and nearly rectangular.
                rectangularity = float(np.clip(area / rect_area, 0.0, 1.0))
                aspect = min(w, h) / max(w, h, 1e-6)
                size = min(1.0, area / 500.0)
                conf = float(np.clip(0.35 + 0.65 * rectangularity * aspect * size, 0.0, 1.0))
                x, y, bw, bh = cv2.boundingRect(c)
                dets.append(
                    Detection(
                        cls,
                        conf,
                        (float(x), float(y), float(x + bw), float(y + bh)),
                        (x + bw / 2.0, y + bh / 2.0),
                    )
                )
        return dets
