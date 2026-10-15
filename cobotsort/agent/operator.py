"""Operator agent: the tool-use loop, the fallback planner and the audit log.

The EduBot pattern: the language model decides *what* to do; deterministic code decides *how*
and *whether it is safe* (guardrails + SafetyGuard). A stop request always wins, whatever the
provider says.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from cobotsort.agent.guardrails import Guardrails
from cobotsort.agent.prompts import SYSTEM_PROMPT
from cobotsort.agent.providers.base import Provider, ProviderError
from cobotsort.agent.providers.mock import STOP_PATTERN, MockProvider, normalize
from cobotsort.agent.tools import TOOLS, ToolExecutor
from cobotsort.config import AgentConfig

log = logging.getLogger(__name__)

# Called before every tool call with the number of calls so far; returns operator input or None.
Interrupt = Callable[[int], str | None]


def make_provider(name: str, cfg: AgentConfig) -> Provider:
    if name == "mock":
        return MockProvider()
    if name == "anthropic":
        from cobotsort.agent.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(cfg.model, timeout_s=cfg.timeout_s)
    raise ProviderError(f"unknown provider {name!r}")


class AuditLog:
    """Append-only JSONL: time, order, provider, tool, input, result and guard decision."""

    def __init__(self, path: Path, order: str) -> None:
        self.path = path
        self.order = order
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        provider: str,
        event: str,
        tool: str | None = None,
        tool_input=None,
        result=None,
        guard: str | None = None,
    ) -> None:
        row = {
            "time": datetime.now().isoformat(timespec="milliseconds"),
            "order": self.order,
            "provider": provider,
            "event": event,
            "tool": tool,
            "input": tool_input,
            "result": result,
            "guard": guard,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


@dataclass
class OrderResult:
    order: str
    provider_requested: str
    provider_used: str
    fallback_reason: str | None
    final_text: str
    calls: list[dict] = field(default_factory=list)
    stopped_by_operator: bool = False
    audit_path: Path | None = None
    wall_time: float = 0.0


class OperatorAgent:
    """Runs one work order at a time against a SortingCell."""

    def __init__(
        self,
        cell,
        cfg: AgentConfig | None = None,
        provider: str | Provider | None = None,
        log_dir: Path | None = None,
    ) -> None:
        self.cell = cell
        self.cfg = cfg or AgentConfig()
        self._provider_obj = provider if isinstance(provider, Provider) else None
        self.provider_name = (
            provider.name if isinstance(provider, Provider) else (provider or self.cfg.provider)
        )
        self.rails = Guardrails(cell, self.cfg)
        self.executor = ToolExecutor(cell, self.rails)
        self.log_dir = log_dir or self.cfg.log_dir
        self.listeners: list[Callable[[str], None]] = []

    def _say(self, text: str) -> None:
        log.info("%s", text)
        for cb in self.listeners:
            cb(text)

    def run(self, order: str, interrupt: Interrupt | None = None) -> OrderResult:
        t0 = time.perf_counter()
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        audit = AuditLog(self.log_dir / f"agent_{stamp}.jsonl", order)
        self.rails.reset_budget()
        if not self.cell.api.engaged and not (self.cell.api.fault or self.cell.api.estop_active):
            self.cell.api.engage()
        result = OrderResult(order, self.provider_name, self.provider_name, None, "")
        result.audit_path = audit.path
        try:
            provider = self._provider_obj or make_provider(self.provider_name, self.cfg)
        except ProviderError as exc:
            provider = self._fallback(result, audit, str(exc))
        audit.write(provider.name, "order")
        provider.start(SYSTEM_PROMPT, order, TOOLS)
        max_turns = self.cfg.max_tool_calls + 5
        for _ in range(max_turns):
            if interrupt is not None:
                msg = interrupt(len(result.calls))
                if msg and re.search(STOP_PATTERN, normalize(msg)):
                    # The E-stop always wins: executed by code, not by the model.
                    outcome = self.executor.execute("stop_cell", {"reason": f"operator: {msg}"})
                    audit.write(
                        provider.name,
                        "operator_stop",
                        "stop_cell",
                        {"message": msg},
                        outcome.content,
                        "estop_override",
                    )
                    result.calls.append(
                        {
                            "tool": "stop_cell",
                            "input": {"message": msg},
                            "guard": "estop_override",
                            "result": outcome.content,
                        }
                    )
                    result.stopped_by_operator = True
                    result.final_text = self._stopped_text(order)
                    break
            try:
                turn = provider.next_turn()
            except ProviderError as exc:
                if provider.name == "mock":
                    raise
                provider = self._fallback(result, audit, str(exc))
                provider.start(SYSTEM_PROMPT, order, TOOLS)
                continue
            if turn.tool_call is None:
                result.final_text = turn.text
                break
            call = turn.tool_call
            self._say(f"{provider.name}: {call.name}({json.dumps(call.input)})")
            outcome = self.executor.execute(call.name, call.input)
            audit.write(
                provider.name, "tool", call.name, call.input, outcome.content, outcome.guard
            )
            result.calls.append(
                {
                    "tool": call.name,
                    "input": call.input,
                    "guard": outcome.guard,
                    "result": outcome.content,
                }
            )
            self._say(f"  -> {outcome.guard}: {json.dumps(outcome.content)[:160]}")
            provider.add_tool_result(call, outcome.content, outcome.is_error)
        else:
            result.final_text = "Stopped: the turn budget for this order is exhausted."
        audit.write(provider.name, "final", result=result.final_text)
        result.provider_used = provider.name
        result.wall_time = time.perf_counter() - t0
        self._say(result.final_text)
        return result

    def _fallback(self, result: OrderResult, audit: AuditLog, reason: str) -> Provider:
        log.warning(
            "provider %s unavailable (%s): using the rule-based fallback planner",
            self.provider_name,
            reason,
        )
        result.fallback_reason = reason
        audit.write("mock", "fallback", result={"reason": reason})
        return MockProvider()

    @staticmethod
    def _stopped_text(order: str) -> str:
        from cobotsort.agent.providers.mock import detect_language

        if detect_language(order) == "pt":
            return "Parado pelo operador. A célula está em parada de proteção (lâmpada vermelha)."
        return "Stopped by the operator. The cell is in a protective stop (lamp RED)."
