"""Tool schemas and their deterministic executors.

The model only chooses tools and arguments. Every call goes through the guardrails, and every
physical action still goes through the SafetyGuard inside the CobotAPI.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from cobotsort.agent.guardrails import GuardrailError, Guardrails
from cobotsort.cell.slots import BinFullError
from cobotsort.motion.trajectory import MotionError
from cobotsort.robot.api import SafetyViolation
from cobotsort.robot.backend import CobotError

log = logging.getLogger(__name__)

BINS = ["A", "B", "C", "REJECT"]

TOOLS: list[dict] = [
    {
        "name": "scan_workspace",
        "description": (
            "Park the arm out of the camera view and detect the parts in the pick area. Returns a "
            "scan_id and the detected objects (id, class_name, confidence, x_mm, y_mm). Object ids "
            "are only valid for this scan_id. Any placement makes the scan stale."
        ),
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "get_cell_status",
        "description": (
            "Read the cell state: engaged, signal lamp, held part, free slots per bin, safety "
            "events, E-stop and fault."
        ),
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "pick_and_place",
        "description": (
            "Pick one detected object and place it in the next free slot of a bin, then verify "
            "with a new scan. Requires the latest scan_id. One part per call. Returns success or "
            "the refusal reason; on success it also returns the verification scan, which becomes "
            "the latest scan."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "scan_id": {"type": "string", "description": "scan_id of the latest scan"},
                "object_id": {"type": "integer", "description": "object id from that scan"},
                "bin": {"type": "string", "enum": BINS},
            },
            "required": ["scan_id", "object_id", "bin"],
            "additionalProperties": False,
        },
    },
    {
        "name": "stop_cell",
        "description": "Protective stop of the cell (lamp RED). Use when asked to stop or pause.",
        "input_schema": {
            "type": "object",
            "properties": {"reason": {"type": "string"}},
            "required": ["reason"],
            "additionalProperties": False,
        },
    },
]


@dataclass
class ToolOutcome:
    """Result of one tool call as returned to the model."""

    content: dict
    is_error: bool
    guard: str  # "allowed", "refused:<code>" or "error:<type>"


class ToolExecutor:
    """Runs agent tool calls against a SortingCell."""

    def __init__(self, cell, guardrails: Guardrails) -> None:
        self.cell = cell
        self.rails = guardrails

    def execute(self, name: str, args: dict | None) -> ToolOutcome:
        args = args or {}
        try:
            self.rails.count_tool_call()
            if name == "scan_workspace":
                return self._scan()
            if name == "get_cell_status":
                return ToolOutcome(self.status(), False, "allowed")
            if name == "pick_and_place":
                return self._pick(args)
            if name == "stop_cell":
                return self._stop(str(args.get("reason", "operator request")))
            raise GuardrailError("unknown_tool", f"unknown tool {name!r}")
        except GuardrailError as exc:
            return ToolOutcome(
                {"error": exc.message, "code": exc.code}, True, f"refused:{exc.code}"
            )

    # ------------------------------------------------------------ tools
    def _scan(self) -> ToolOutcome:
        self.rails.check_cell_ready()
        try:
            scan = self.cell.scan()
        except (CobotError, MotionError) as exc:
            return ToolOutcome(
                {"error": f"scan failed: {exc}"}, True, f"error:{type(exc).__name__}"
            )
        return ToolOutcome(self.scan_payload(scan), False, "allowed")

    @staticmethod
    def scan_payload(scan) -> dict:
        objects = [
            {
                "id": i,
                "class_name": d.class_name,
                "confidence": round(d.confidence, 2),
                "x_mm": round(d.world_xy[0] * 1e3, 1),
                "y_mm": round(d.world_xy[1] * 1e3, 1),
            }
            for i, d in enumerate(scan.detections)
        ]
        return {"scan_id": scan.scan_id, "objects": objects}

    def status(self) -> dict:
        api = self.cell.api
        return {
            "engaged": api.engaged,
            "lamp": api.lamp.state.value,
            "held_part": api.backend.held_part is not None,
            "free_slots": self.cell.slots.free_by_bin(),
            "safety_events": len(api.safety_events),
            "estop": api.estop_active,
            "fault": api.fault,
            "latest_scan_id": self.cell.last_scan.scan_id if self.cell.last_scan else None,
        }

    def _pick(self, args: dict) -> ToolOutcome:
        det = self.rails.validate_pick(args.get("scan_id"), args.get("object_id"), args.get("bin"))
        bin_name = args["bin"]
        self.rails.picks += 1
        try:
            rec = self.cell.pick_to_bin(det, bin_name)
        except BinFullError as exc:
            raise GuardrailError("bin_full", str(exc)) from exc
        except (SafetyViolation, CobotError, MotionError) as exc:
            return ToolOutcome(
                {
                    "success": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "status": self.status(),
                },
                True,
                f"error:{type(exc).__name__}",
            )
        content = {
            "success": rec.verified,
            "class_name": rec.class_name,
            "bin": bin_name,
            "slot": rec.slot,
            "verified": rec.verified,
            "detail": rec.reason,
        }
        last = self.cell.last_scan
        if last is not None and self.cell.placements_since_scan == 0:
            # The verification scan was taken after the placement, so it is the latest scan.
            content["verification_scan"] = self.scan_payload(last)
            content["note"] = (
                f"the scan used for this pick is stale; {last.scan_id} (verification scan) is the "
                "latest one: use its scan_id and ids for the next pick"
            )
        return ToolOutcome(content, not rec.verified, "allowed" if rec.verified else "error:verify")

    def _stop(self, reason: str) -> ToolOutcome:
        self.cell.api.stop(f"agent stop: {reason}")
        return ToolOutcome({"ack": True, "lamp": self.cell.api.lamp.state.value}, False, "allowed")
