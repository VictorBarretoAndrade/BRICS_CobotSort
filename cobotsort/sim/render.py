"""Offscreen RGB / segmentation rendering and simple image overlays."""

from __future__ import annotations

import logging
from collections.abc import Iterable

import cv2
import mujoco
import numpy as np

log = logging.getLogger(__name__)

CLASS_BGR: dict[str, tuple[int, int, int]] = {
    "red_cube": (40, 40, 230),
    "green_cube": (60, 190, 60),
    "blue_cube": (230, 110, 40),
    "defect": (200, 60, 170),
}


class CellRenderer:
    """Caches one ``mujoco.Renderer`` per image size.

    Renderers own an OpenGL context, so create and use them in one thread.
    """

    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = model
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}

    def _get(self, width: int, height: int) -> mujoco.Renderer:
        key = (width, height)
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(self.model, height=height, width=width)
        return self._renderers[key]

    def rgb(self, data: mujoco.MjData, camera: str | int, width: int, height: int) -> np.ndarray:
        """Render an RGB image (H, W, 3) uint8."""
        r = self._get(width, height)
        r.update_scene(data, camera=camera)
        return r.render().copy()

    def segmentation(
        self, data: mujoco.MjData, camera: str | int, width: int, height: int
    ) -> np.ndarray:
        """Render geom ids (H, W) int32; -1 where no geom is visible."""
        r = self._get(width, height)
        r.enable_segmentation_rendering()
        try:
            r.update_scene(data, camera=camera)
            seg = r.render().copy()
        finally:
            r.disable_segmentation_rendering()
        ids = seg[..., 0].astype(np.int32)
        ids[seg[..., 1] != int(mujoco.mjtObj.mjOBJ_GEOM)] = -1
        return ids

    def close(self) -> None:
        for r in self._renderers.values():
            r.close()
        self._renderers.clear()


def draw_boxes(img: np.ndarray, detections: Iterable, show_conf: bool = True) -> np.ndarray:
    """Draw detection boxes on an RGB image (returns a copy)."""
    out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = (int(round(v)) for v in det.bbox_xyxy)
        color = CLASS_BGR.get(det.class_name, (200, 200, 200))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{i}:{det.class_name.split('_')[0]}"
        if show_conf:
            label += f" {det.confidence:.2f}"
        cv2.putText(
            out, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA
        )
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def put_label(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    scale: float = 0.7,
    color: tuple[int, int, int] = (255, 255, 255),
    bg: tuple[int, int, int] | None = (0, 0, 0),
    thickness: int = 2,
) -> None:
    """Write ``text`` in place on an RGB image with an optional background box."""
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = org
    if bg is not None:
        sub = img[max(0, y - h - 6) : y + base + 4, max(0, x - 6) : x + w + 6]
        sub[:] = (sub * 0.35 + np.array(bg) * 0.65).astype(np.uint8)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
