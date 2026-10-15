"""Pinhole camera model of a MuJoCo camera: intrinsics from fovy, pixel <-> world.

MuJoCo cameras look along their -z axis with +y up, while image rows grow downwards, so the
v axis is flipped. Pixel coordinates are continuous: pixel (i, j) covers [i, i+1) x [j, j+1).
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class CameraModel:
    width: int
    height: int
    fovy_deg: float
    pos: np.ndarray  # camera center in the world frame
    rot: np.ndarray  # columns: camera x, y, z axes in the world frame

    @property
    def fy(self) -> float:
        return (self.height / 2) / np.tan(np.radians(self.fovy_deg) / 2)

    @property
    def fx(self) -> float:
        return self.fy

    @property
    def cx(self) -> float:
        return self.width / 2

    @property
    def cy(self) -> float:
        return self.height / 2

    @classmethod
    def from_sim(
        cls, model: mujoco.MjModel, data: mujoco.MjData, camera: str | int, width: int, height: int
    ) -> CameraModel:
        cam = camera if isinstance(camera, int) else model.camera(camera).id
        return cls(
            width,
            height,
            float(model.cam_fovy[cam]),
            data.cam_xpos[cam].copy(),
            data.cam_xmat[cam].reshape(3, 3).copy(),
        )

    def world_to_pixel(self, points: np.ndarray) -> np.ndarray:
        """Project world points (N, 3) to pixel coordinates (N, 2)."""
        p = np.atleast_2d(points) - self.pos
        pc = p @ self.rot  # coordinates in the camera frame
        depth = -pc[:, 2]
        u = self.cx + self.fx * pc[:, 0] / depth
        v = self.cy - self.fy * pc[:, 1] / depth
        return np.stack([u, v], axis=1)

    def pixel_to_world(self, u: float, v: float, z_plane: float) -> np.ndarray:
        """Intersect the ray through pixel (u, v) with the plane z = ``z_plane``."""
        d_cam = np.array([(u - self.cx) / self.fx, -(v - self.cy) / self.fy, -1.0])
        d = self.rot @ d_cam
        if abs(d[2]) < 1e-9:
            raise ValueError("ray parallel to the plane")
        t = (z_plane - self.pos[2]) / d[2]
        return self.pos + t * d
