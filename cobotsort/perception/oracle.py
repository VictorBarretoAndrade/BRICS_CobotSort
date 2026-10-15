"""Ground-truth detector: the upper bound used in benchmarks."""

from __future__ import annotations

import numpy as np

from cobotsort.perception.camera import CameraModel
from cobotsort.perception.detector import Detection, Detector
from cobotsort.sim.world import SimWorld, quat_to_mat


class OracleDetector(Detector):
    """Reads cube poses from the simulator; boxes are the projected cube corners."""

    name = "oracle"

    def __init__(self, world: SimWorld, camera: CameraModel) -> None:
        self.world = world
        self.camera = camera
        h = world.cfg.scene.cube_edge / 2
        self._corners = np.array(
            [[sx * h, sy * h, sz * h] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        )

    def detect(self, rgb: np.ndarray | None = None) -> list[Detection]:
        out = []
        cam = self.camera
        for cube in self.world.cubes():
            if cube.index == self.world.attached or cube.pos[2] > 0.1:
                continue
            pts = cube.pos + self._corners @ quat_to_mat(cube.quat).T
            uv = cam.world_to_pixel(pts)
            x1, y1 = uv.min(axis=0)
            x2, y2 = uv.max(axis=0)
            if x2 < 0 or y2 < 0 or x1 > cam.width or y1 > cam.height:
                continue
            box = (max(x1, 0.0), max(y1, 0.0), min(x2, cam.width), min(y2, cam.height))
            center = cam.world_to_pixel(cube.pos[None])[0]
            det = Detection(cube.class_name, 1.0, box, (float(center[0]), float(center[1])))
            det.world_xy = (float(cube.pos[0]), float(cube.pos[1]))
            out.append(det)
        return out
