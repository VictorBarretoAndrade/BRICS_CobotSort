"""Deterministic guardrails between the operator agent and the cell.

The language model decides *what* to do; this code decides *whether* it is allowed. Nothing
here depends on the prompt: a model that ignores its instructions still cannot move a part
from a stale scan, into a full or unknown bin, or past the tool-call and pick budgets.
"""

from __future__ import annotations

from dataclasses import dataclass

from cobotsort.config import AgentConfig
from cobotsort.perception.detector import Detection


class GuardrailError(Exception):
    """A refused tool call. ``code`` is stable and machine-readable."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class Guardrails:
    """Validates agent tool calls against the live cell state."""

    cell: object  # SortingCell (kept untyped to avoid an import cycle)
    cfg: AgentConfig
    tool_calls: int = 0
    picks: int = 0

    def reset_budget(self) -> None:
        self.tool_calls = 0
        self.picks = 0

    def count_tool_call(self) -> None:
        if self.tool_calls >= self.cfg.max_tool_calls:
            raise GuardrailError(
                "tool_limit", f"tool-call limit reached ({self.cfg.max_tool_calls} per order)"
            )
        self.tool_calls += 1

    def check_cell_ready(self) -> None:
        api = self.cell.api
        if api.estop_active:
            raise GuardrailError("estop", "E-stop is active: the cell will not move")
        if api.fault:
            raise GuardrailError("fault", f"cell fault: {api.fault}")

    def validate_pick(self, scan_id: str, object_id: int, bin_name: str) -> Detection:
        """Check a pick_and_place request; return the detection it refers to."""
        self.check_cell_ready()
        last = self.cell.last_scan
        if last is None:
            raise GuardrailError("no_scan", "no scan yet: call scan_workspace first")
        if scan_id != last.scan_id:
            raise GuardrailError(
                "stale_scan", f"scan_id {scan_id!r} is not the latest scan ({last.scan_id})"
            )
        if self.cell.placements_since_scan > 0:
            raise GuardrailError(
                "stale_scan", f"scan {scan_id} is stale after a placement: scan again"
            )
        if isinstance(object_id, bool) or not isinstance(object_id, int):
            raise GuardrailError(
                "unknown_object", f"object_id must be an integer, got {object_id!r}"
            )
        if not 0 <= object_id < len(last.detections):
            raise GuardrailError("unknown_object", f"object_id {object_id} is not in {scan_id}")
        det = last.detections[object_id]
        min_conf = self.cell.rules.min_confidence
        if det.confidence < min_conf:
            raise GuardrailError(
                "low_confidence",
                f"object {object_id} has confidence {det.confidence:.2f} < {min_conf}",
            )
        if bin_name not in self.cell.slots.bins:
            raise GuardrailError(
                "unknown_bin", f"bin {bin_name!r} does not exist (bins: {self.cell.slots.bins})"
            )
        if self.cell.slots.free(bin_name) == 0:
            raise GuardrailError("bin_full", f"bin {bin_name} is full")
        if self.picks >= self.cfg.max_picks:
            raise GuardrailError(
                "pick_limit", f"pick limit reached ({self.cfg.max_picks} per order)"
            )
        return det
