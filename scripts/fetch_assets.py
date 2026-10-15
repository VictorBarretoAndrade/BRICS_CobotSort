"""Sparse-clone the UR5e model from MuJoCo Menagerie into ``third_party/``.

Usage::

    python scripts/fetch_assets.py
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/google-deepmind/mujoco_menagerie"
ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "third_party" / "mujoco_menagerie"
ROBOT_DIR = "universal_robots_ur5e"

log = logging.getLogger("fetch_assets")


def _run(args: list[str], cwd: Path | None = None) -> None:
    log.info("$ %s", " ".join(args))
    subprocess.run(args, cwd=cwd, check=True)


def fetch() -> Path:
    """Clone (or update) the sparse checkout and return the UR5e folder."""
    target = DEST / ROBOT_DIR / "ur5e.xml"
    if target.exists():
        log.info("UR5e already present at %s", target.parent)
        return target.parent
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if not DEST.exists():
        _run(
            ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", REPO_URL, str(DEST)]
        )
    _run(["git", "sparse-checkout", "set", ROBOT_DIR], cwd=DEST)
    if not target.exists():
        raise FileNotFoundError(f"sparse checkout finished but {target} is missing")
    log.info("UR5e ready at %s", target.parent)
    return target.parent


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        fetch()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        log.error("fetch failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
