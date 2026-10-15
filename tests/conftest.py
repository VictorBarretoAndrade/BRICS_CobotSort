"""Shared fixtures. The scene needs the Menagerie UR5e (``python scripts/fetch_assets.py``)."""

from __future__ import annotations

import pytest

from cobotsort.config import MENAGERIE_UR5E

if not MENAGERIE_UR5E.exists():
    pytest.skip("run `python scripts/fetch_assets.py` first", allow_module_level=True)


@pytest.fixture
def world():
    from cobotsort.sim.world import SimWorld

    w = SimWorld()
    w.reset(seed=0, classes=["red_cube"], positions=[(0.45, 0.05)])
    return w


@pytest.fixture
def api(world):
    from cobotsort.robot.api import make_api

    a = make_api(world)
    a.engage()
    return a
