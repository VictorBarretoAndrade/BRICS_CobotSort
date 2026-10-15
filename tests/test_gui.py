"""GUI: run the self-test (every mode plus the E-stop flow) on Qt's offscreen platform."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

pytest.importorskip("PyQt6")


def test_gui_selftest_offscreen(tmp_path):
    code = (
        "import sys; from pathlib import Path; from cobotsort.gui.app import run_gui; "
        f"sys.exit(run_gui('oracle', 3, selftest=True, out_dir=Path({str(tmp_path)!r}), "
        f"report=Path({str(tmp_path / 'report.md')!r})))"
    )
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=600,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "FAIL" not in report
    assert "E-stop during a motion | pass" in report
    for shot in (
        "gui_manual.png",
        "gui_semi.png",
        "gui_auto.png",
        "gui_agent.png",
        "gui_estop.png",
    ):
        assert (tmp_path / shot).stat().st_size > 10_000
