"""Yoshikawa manipulability and singularity checks.

``w = sqrt(det(J J^T))`` drops to zero at a singularity: the arm loses the ability to move
the tool in some direction, and joint speeds blow up for small Cartesian motions. On a real
cobot this is what triggers "singularity" faults (the author met it on the ARM95 when the arm
was nearly stretched).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cobotsort.kinematics.ik import IKSolver


def manipulability(jac: np.ndarray) -> float:
    """Yoshikawa index of a Jacobian."""
    return float(np.sqrt(max(np.linalg.det(jac @ jac.T), 0.0)))


@dataclass
class ManipulabilityCheck:
    """Classify configurations as ok / warn / block from calibrated thresholds."""

    solver: IKSolver
    warn: float
    block: float

    def index(self, q: np.ndarray) -> float:
        return manipulability(self.solver.jacobian(q))

    def classify(self, q: np.ndarray) -> tuple[str, float]:
        w = self.index(q)
        if w < self.block:
            return "block", w
        if w < self.warn:
            return "warn", w
        return "ok", w

    def min_along(self, qs: np.ndarray, stride: int = 1) -> tuple[float, int]:
        """Lowest index along a joint path and its sample index."""
        best, arg = np.inf, -1
        for i in range(0, len(qs), stride):
            w = self.index(qs[i])
            if w < best:
                best, arg = w, i
        if len(qs) and arg != len(qs) - 1:
            w = self.index(qs[-1])
            if w < best:
                best, arg = w, len(qs) - 1
        return float(best), arg
