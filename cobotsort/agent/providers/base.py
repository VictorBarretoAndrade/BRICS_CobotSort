"""Interchangeable provider layer (mock / anthropic), as in EduBot."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field


class ProviderError(RuntimeError):
    """The provider cannot continue (API error, timeout, missing key, refusal)."""


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class AssistantTurn:
    """One model turn: some text and at most one tool call."""

    text: str = ""
    tool_call: ToolCall | None = None
    stop_reason: str = "end_turn"
    meta: dict = field(default_factory=dict)


class Provider(abc.ABC):
    """Decides *what* to do next; never executes anything itself."""

    name: str = "provider"

    @abc.abstractmethod
    def start(self, system: str, order: str, tools: list[dict]) -> None:
        """Begin a new work order."""

    @abc.abstractmethod
    def next_turn(self) -> AssistantTurn:
        """Produce the next assistant turn."""

    @abc.abstractmethod
    def add_tool_result(self, call: ToolCall, content: dict, is_error: bool) -> None:
        """Feed back the result of the last tool call."""
