"""Detection dataclass and the Detector interface."""

from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np

from cobotsort.perception.camera import CameraModel


@dataclass
class Detection:
    """One detected object in the top camera image."""

    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    center_px: tuple[float, float]
    world_xy: tuple[float, float] | None = None

    def to_world(self, camera: CameraModel, z_plane: float) -> Detection:
        """Fill ``world_xy`` by intersecting the center ray with the cube-center plane."""
        p = camera.pixel_to_world(self.center_px[0], self.center_px[1], z_plane)
        self.world_xy = (float(p[0]), float(p[1]))
        return self

    @staticmethod
    def from_box(class_name: str, confidence: float, box: np.ndarray | tuple) -> Detection:
        x1, y1, x2, y2 = (float(v) for v in box)
        return Detection(
            class_name, float(confidence), (x1, y1, x2, y2), ((x1 + x2) / 2, (y1 + y2) / 2)
        )


class Detector(abc.ABC):
    """Anything that turns an RGB image into detections."""

    name: str = "detector"
    needs_image: bool = True  # False for detectors that read the simulator directly

    @abc.abstractmethod
    def detect(self, rgb: np.ndarray) -> list[Detection]:
        """Detect objects in an (H, W, 3) uint8 RGB image."""
