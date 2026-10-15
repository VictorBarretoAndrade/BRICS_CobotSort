"""Cell logic: rules, slot allocation, semi-automatic mode and benchmark scoring."""

from __future__ import annotations

import pytest

from cobotsort.cell.rules import SortingRules
from cobotsort.cell.slots import BinFullError, SlotAllocator, UnknownBinError
from cobotsort.perception.detector import Detection


def _det(cls: str, conf: float = 0.9) -> Detection:
    return Detection(cls, conf, (0, 0, 10, 10), (5, 5), (0.45, 0.0))


def test_default_rules_and_flags():
    rules = SortingRules()
    assert rules.bin_for(_det("red_cube")) == "A"
    assert rules.bin_for(_det("green_cube")) == "B"
    assert rules.bin_for(_det("blue_cube")) == "C"
    assert rules.bin_for(_det("defect")) == "REJECT"
    assert rules.bin_for(_det("red_cube", 0.4)) is None
    assert "low confidence" in rules.why_not(_det("red_cube", 0.4))
    assert rules.bin_for(_det("yellow_cube")) is None
    assert "unknown class" in rules.why_not(_det("yellow_cube"))


def test_slots_fill_in_order_and_refuse_when_full():
    slots = SlotAllocator()
    labels = [slots.allocate("A").label for _ in range(4)]
    assert labels == ["A1", "A2", "A3", "A4"]
    assert slots.free("A") == 0
    with pytest.raises(BinFullError):
        slots.allocate("A")
    with pytest.raises(UnknownBinError):
        slots.free("D")
    s = slots.allocate("B")
    slots.release(s)
    assert slots.free("B") == 4


def test_semi_auto_scripted_session(capsys):
    from cobotsort.cli import main

    assert main(["semi", "--detector", "oracle", "--seed", "11", "--commands", "0 A,7 B,q"]) == 0
    out = capsys.readouterr().out
    assert "OK:" in out
    assert "refused: object 7" in out


def test_benchmark_scoring_one_episode():
    from cobotsort.cell.sorter import build_cell
    from cobotsort.metrics.benchmark import score_episode, summarize

    cell = build_cell("oracle", seed=11)
    report = cell.run_auto()
    ep = score_episode(cell, report, 11)
    assert ep.cubes == 6 and ep.correct == 6 and ep.wrong_bin == 0 and ep.completed
    assert max(ep.placement_errors_mm) < 5.0
    row = summarize("oracle", [ep], (10, 10))
    assert row["cubes_sorted_pct"] == 100.0
    assert row["adversarial_blocked"] == "10/10"
