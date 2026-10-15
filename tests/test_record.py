"""Demo recorder: frame composition, video and GIF writing."""

from __future__ import annotations

from PIL import Image

from cobotsort.cell.sorter import build_cell
from cobotsort.metrics.record import RecordConfig, Recorder, save_gif


def test_recorder_writes_video_and_gif(tmp_path):
    cell = build_cell("oracle", seed=11)
    cfg = RecordConfig(gif_start_s=0.0, gif_seconds=1.0)
    rec = Recorder(cell.world, cell, tmp_path / "t.mp4", cfg)
    rec.order = "Separe as peças por cor."
    rec.log_line("agent (mock): scan_workspace")
    frame = rec.compose(6.0)
    assert frame.shape == (720, 1280, 3)
    for i in range(12):
        f = frame.copy()
        f[:40, : 40 + 60 * i] = 255  # make every frame different
        rec.write(f)
    rec.close()
    save_gif(rec.gif_frames, tmp_path / "t.gif", cfg.gif_fps)
    assert (tmp_path / "t.mp4").stat().st_size > 0
    gif = Image.open(tmp_path / "t.gif")
    assert len(rec.gif_frames) == 4
    assert gif.size[0] == cfg.gif_width and gif.n_frames == 4
