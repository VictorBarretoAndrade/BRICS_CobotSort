"""UR5e Denavit-Hartenberg forward kinematics.

Standard DH: ``T_i = Rz(theta_i) Tz(d_i) Tx(a_i) Rx(alpha_i)``, four parameters per link.

Two tables are provided:

* ``UR5E_OFFICIAL``: the table published by Universal Robots for the UR5e.
* ``UR5E_MENAGERIE``: the same structure with the link offsets used by the MuJoCo Menagerie
  model (rounded values: d1 = 0.163, a3 = -0.392, d4 = 0.134, d5 = d6 = 0.1).

Because the simulator uses the Menagerie geometry, the Menagerie table is the one that
matches MuJoCo to numerical precision; the official table differs by about a millimeter
(see ``docs/decisions.md``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class DHTable:
    """Standard DH parameters for a 6-joint serial arm."""

    name: str
    d: tuple[float, ...]
    a: tuple[float, ...]
    alpha: tuple[float, ...]


HALF_PI = np.pi / 2
UR5E_OFFICIAL = DHTable(
    name="UR5e (Universal Robots)",
    d=(0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996),
    a=(0.0, -0.425, -0.3922, 0.0, 0.0, 0.0),
    alpha=(HALF_PI, 0.0, 0.0, HALF_PI, -HALF_PI, 0.0),
)
UR5E_MENAGERIE = DHTable(
    name="UR5e (MuJoCo Menagerie geometry)",
    d=(0.163, 0.0, 0.0, 0.134, 0.1, 0.1),
    a=(0.0, -0.425, -0.392, 0.0, 0.0, 0.0),
    alpha=(HALF_PI, 0.0, 0.0, HALF_PI, -HALF_PI, 0.0),
)


def dh_transform(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    """Homogeneous transform of one standard-DH link."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def forward(q: np.ndarray, table: DHTable = UR5E_MENAGERIE) -> np.ndarray:
    """Flange pose (4x4) in the DH base frame."""
    t = np.eye(4)
    for i in range(6):
        t = t @ dh_transform(float(q[i]), table.d[i], table.a[i], table.alpha[i])
    return t


def forward_batch(qs: np.ndarray, table: DHTable = UR5E_MENAGERIE) -> np.ndarray:
    """Vectorized flange poses (N, 4, 4) for joint vectors ``qs`` (N, 6)."""
    qs = np.atleast_2d(qs)
    n = len(qs)
    t = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
    for i in range(6):
        ct, st = np.cos(qs[:, i]), np.sin(qs[:, i])
        ca, sa = np.cos(table.alpha[i]), np.sin(table.alpha[i])
        link = np.zeros((n, 4, 4))
        link[:, 0, 0], link[:, 0, 1], link[:, 0, 2], link[:, 0, 3] = (
            ct,
            -st * ca,
            st * sa,
            table.a[i] * ct,
        )
        link[:, 1, 0], link[:, 1, 1], link[:, 1, 2], link[:, 1, 3] = (
            st,
            ct * ca,
            -ct * sa,
            table.a[i] * st,
        )
        link[:, 2, 1], link[:, 2, 2], link[:, 2, 3] = sa, ca, table.d[i]
        link[:, 3, 3] = 1.0
        t = t @ link
    return t


def tcp_positions(qs: np.ndarray, tool_length: float = 0.03) -> np.ndarray:
    """TCP positions (N, 3) of the cell (flange + suction cup) for joint vectors ``qs``."""
    t = forward_batch(qs)
    return t[:, :3, 3] + tool_length * t[:, :3, 2]


def forward_all(q: np.ndarray, table: DHTable = UR5E_MENAGERIE) -> list[np.ndarray]:
    """Frames 0..6 (base to flange) as a list of 4x4 transforms."""
    frames = [np.eye(4)]
    for i in range(6):
        frames.append(
            frames[-1] @ dh_transform(float(q[i]), table.d[i], table.a[i], table.alpha[i])
        )
    return frames


def pose_to_matrix(pos: np.ndarray, rot: np.ndarray) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = rot
    t[:3, 3] = pos
    return t


def _vec_to_tf(v: np.ndarray) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = Rotation.from_rotvec(v[3:]).as_matrix()
    t[:3, 3] = v[:3]
    return t


@dataclass
class DHModel:
    """DH chain with constant base and flange transforms: ``T = B @ FK(q) @ F``."""

    table: DHTable = UR5E_MENAGERIE
    base: np.ndarray = None  # type: ignore[assignment]
    flange: np.ndarray = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.base = np.eye(4) if self.base is None else self.base
        self.flange = np.eye(4) if self.flange is None else self.flange

    def fk(self, q: np.ndarray) -> np.ndarray:
        return self.base @ forward(q, self.table) @ self.flange

    def fit(self, qs: np.ndarray, poses: list[np.ndarray]) -> float:
        """Fit the constant base and flange transforms to measured flange poses.

        Returns the RMS position residual in meters.
        """

        def residual(v: np.ndarray) -> np.ndarray:
            b, f = _vec_to_tf(v[:6]), _vec_to_tf(v[6:])
            res = []
            for q, target in zip(qs, poses, strict=True):
                t = b @ forward(q, self.table) @ f
                res.append(t[:3, 3] - target[:3, 3])
                res.append(0.1 * (t[:3, :3] - target[:3, :3]).ravel())
            return np.concatenate(res)

        best = None
        # A few starting guesses for the flange rotation (the axis conventions may differ).
        for rv in ([0, 0, 0], [np.pi / 2, 0, 0], [-np.pi / 2, 0, 0], [0, 0, np.pi]):
            x0 = np.zeros(12)
            x0[9:] = rv
            sol = least_squares(residual, x0, xtol=1e-12, ftol=1e-12, gtol=1e-12)
            if best is None or sol.cost < best.cost:
                best = sol
        assert best is not None
        self.base, self.flange = _vec_to_tf(best.x[:6]), _vec_to_tf(best.x[6:])
        pos_err = [
            np.linalg.norm(self.fk(q)[:3, 3] - p[:3, 3]) for q, p in zip(qs, poses, strict=True)
        ]
        return float(np.sqrt(np.mean(np.square(pos_err))))


def pose_error(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Position error (m) and rotation error (deg) between two 4x4 poses."""
    dp = float(np.linalg.norm(a[:3, 3] - b[:3, 3]))
    dr = Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude()
    return dp, float(np.degrees(dr))
