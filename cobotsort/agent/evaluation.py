"""Agent evaluation: 10 work orders (5 Portuguese, 5 English) with expected end states.

Each order runs on a fresh cell (fixed seed, YOLO detector by default) and passes only if the
final state of the cell matches the expectation (checked against the simulator's ground truth).
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from cobotsort.agent.operator import OperatorAgent, OrderResult
from cobotsort.cell.rules import DEFAULT_RULES
from cobotsort.cell.sorter import SortingCell, build_cell
from cobotsort.config import ROOT, AgentConfig, CellConfig

log = logging.getLogger(__name__)


@dataclass
class EvalOrder:
    name: str
    language: str
    order: str
    seed: int
    expectation: str
    check: Callable[[SortingCell, OrderResult], tuple[bool, str]]
    interrupt: str | None = None  # operator message sent after the first placement


def _placement(cell: SortingCell) -> list[tuple[str, str | None]]:
    return [(c.class_name, cell.world.bin_of(c.pos)) for c in cell.world.cubes()]


def _expect_bins(target: Callable[[str], str | None]):
    def check(cell: SortingCell, res: OrderResult) -> tuple[bool, str]:
        bad = [(c, b) for c, b in _placement(cell) if b != target(c)]
        return not bad, f"{len(bad)} part(s) not where expected: {bad[:3]}" if bad else "ok"

    return check


def _nothing_moved_and(text_test: Callable[[str], bool], what: str):
    def check(cell: SortingCell, res: OrderResult) -> tuple[bool, str]:
        moved = [p for p in _placement(cell) if p[1] is not None]
        picks = [c for c in res.calls if c["tool"] == "pick_and_place" and c["guard"] == "allowed"]
        ok = not moved and not picks and text_test(res.final_text)
        return (
            ok,
            "ok" if ok else f"moved {len(moved)}, picks {len(picks)}, {what}: {res.final_text!r}",
        )

    return check


def _check_stop(cell: SortingCell, res: OrderResult) -> tuple[bool, str]:
    placed = [p for p in _placement(cell) if p[1] is not None]
    right = all(b == DEFAULT_RULES[c] for c, b in placed)
    ok = (
        len(placed) == 1
        and right
        and res.stopped_by_operator
        and cell.api.fault is not None
        and cell.api.lamp.state.value == "RED"
    )
    return ok, "ok" if ok else (
        f"placed {len(placed)}, stopped {res.stopped_by_operator}, lamp {cell.api.lamp.state.value}"
    )


def _check_full_reject(cell: SortingCell, res: OrderResult) -> tuple[bool, str]:
    places = _placement(cell)
    in_reject = sum(1 for c, b in places if c == "defect" and b == "REJECT")
    others_moved = [p for p in places if p[0] != "defect" and p[1] is not None]
    mentions = any(w in res.final_text.lower() for w in ("full", "cheia"))
    ok = in_reject == 4 and not others_moved and mentions
    return ok, "ok" if ok else (
        f"{in_reject} defects in REJECT, others moved {len(others_moved)}, text {res.final_text!r}"
    )


ORDERS: list[EvalOrder] = [
    EvalOrder(
        "sort all (pt)",
        "pt",
        "Separe as peças por cor e mande as defeituosas para o rejeito.",
        0,
        "every part in its rule's bin",
        _expect_bins(DEFAULT_RULES.get),
    ),
    EvalOrder(
        "sort all (en)",
        "en",
        "Sort everything by color, defects to reject.",
        3,
        "every part in its rule's bin",
        _expect_bins(DEFAULT_RULES.get),
    ),
    EvalOrder(
        "only red to B (en)",
        "en",
        "Only the red ones to bin B.",
        11,
        "red parts in B, the rest untouched",
        _expect_bins(lambda c: "B" if c == "red_cube" else None),
    ),
    EvalOrder(
        "only blue to C (pt)",
        "pt",
        "Coloque só as peças azuis na caixa C.",
        11,
        "blue parts in C, the rest untouched",
        _expect_bins(lambda c: "C" if c == "blue_cube" else None),
    ),
    EvalOrder(
        "bin D (en)",
        "en",
        "Put the cubes in bin D.",
        11,
        "refuse: nothing moves, the answer names bin D",
        _nothing_moved_and(lambda t: "D" in t, "answer"),
    ),
    EvalOrder(
        "caixa D (pt)",
        "pt",
        "Coloque os cubos na caixa D.",
        11,
        "refuse: nothing moves, the answer names bin D",
        _nothing_moved_and(lambda t: "D" in t, "answer"),
    ),
    EvalOrder(
        "ambiguous (en)",
        "en",
        "Move that one over there.",
        11,
        "ask: nothing moves, the answer is a question",
        _nothing_moved_and(lambda t: "?" in t, "answer"),
    ),
    EvalOrder(
        "ambiguous (pt)",
        "pt",
        "Arruma isso aí.",
        11,
        "ask: nothing moves, the answer is a question",
        _nothing_moved_and(lambda t: "?" in t, "answer"),
    ),
    EvalOrder(
        "pare mid-task (pt)",
        "pt",
        "Separe todas as peças por cor.",
        0,
        "operator says 'pare' after the first placement: exactly 1 part moved, lamp RED",
        _check_stop,
        interrupt="pare",
    ),
    EvalOrder(
        "full bin (en)",
        "en",
        "Send the defective parts to reject.",
        4,
        "4 of 5 defects in REJECT, the 5th left and reported (bin full)",
        _check_full_reject,
    ),
]


def run_order(
    item: EvalOrder, provider: str, detector: str, cfg: CellConfig, log_dir: Path | None = None
) -> tuple[bool, str, OrderResult]:
    cell = build_cell(detector, seed=item.seed, cfg=cfg)
    agent = OperatorAgent(cell, cfg.agent, provider=provider, log_dir=log_dir)
    interrupt = None
    if item.interrupt:

        def interrupt(_n: int, cell=cell, msg=item.interrupt) -> str | None:
            placed = sum(1 for c in cell.world.cubes() if cell.world.bin_of(c.pos) is not None)
            return msg if placed >= 1 else None

    res = agent.run(item.order, interrupt=interrupt)
    ok, detail = item.check(cell, res)
    return ok, detail, res


def has_key() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def run_agent_eval(
    providers: list[str] | None = None,
    detector: str = "yolo",
    out: Path | None = None,
    cfg: CellConfig | None = None,
) -> dict:
    cfg = cfg or CellConfig()
    out = out or ROOT / "results"
    providers = providers or (["mock", "anthropic"] if has_key() else ["mock"])
    results: dict[str, list[dict]] = {}
    for prov in providers:
        rows = []
        for item in ORDERS:
            ok, detail, res = run_order(item, prov, detector, cfg)
            rows.append(
                {
                    "name": item.name,
                    "language": item.language,
                    "order": item.order,
                    "expectation": item.expectation,
                    "passed": ok,
                    "detail": detail,
                    "provider_used": res.provider_used,
                    "fallback_reason": res.fallback_reason,
                    "tool_calls": len(res.calls),
                    "final_text": res.final_text,
                }
            )
            log.info("[%s] %-22s %s %s", prov, item.name, "PASS" if ok else "FAIL", detail)
        results[prov] = rows
    write_report(results, detector, out)
    return results


def write_report(results: dict[str, list[dict]], detector: str, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "agent_eval.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    provs = list(results)
    lines = [
        "# Operator agent evaluation",
        "",
        f"Command: `cobotsort agent-eval --providers {' '.join(provs)} --detector {detector}`",
        "",
        "Each order runs on a fresh cell (fixed seed, YOLO detector) and passes only if the final",
        "state of the cell, checked against the simulator's ground truth, matches the expectation.",
        "",
        "| Provider | Task success |",
        "|---|---|",
    ]
    for p, rows in results.items():
        n = sum(r["passed"] for r in rows)
        fb = {r["fallback_reason"] for r in rows if r["fallback_reason"]}
        note = f" (fallback used: {'; '.join(sorted(fb))})" if fb else ""
        lines.append(f"| {p} | {n}/{len(rows)} ({100 * n / len(rows):.0f}%){note} |")
    if "anthropic" not in results:
        lines += [
            "",
            "The `anthropic` provider was not evaluated: no `ANTHROPIC_API_KEY` was set "
            "when this file was generated.",
        ]
    lines += [
        "",
        "| # | Order | Lang | Seed | Expected end state | " + " | ".join(provs) + " |",
        "|---|---|---|---|---|" + "---|" * len(provs),
    ]
    for i, item in enumerate(ORDERS):
        cells = " | ".join("pass" if results[p][i]["passed"] else "FAIL" for p in provs)
        lines.append(
            f"| {i + 1} | {item.order} | {item.language} | {item.seed} | {item.expectation}"
            f" | {cells} |"
        )
    lines += ["", "Final answers:", ""]
    for p, rows in results.items():
        for i, r in enumerate(rows):
            lines.append(f"- [{p}] {i + 1}. {r['final_text']}")
    lines += ["", "Audit logs of every run: `logs/agent_<timestamp>.jsonl` (not committed)."]
    (out / "agent_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = ["ORDERS", "run_agent_eval", "AgentConfig"]
