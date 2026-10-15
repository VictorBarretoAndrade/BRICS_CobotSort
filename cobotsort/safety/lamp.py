"""Signal tower state machine (GREEN / YELLOW / RED)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

log = logging.getLogger(__name__)


class LampState(StrEnum):
    GREEN = "GREEN"  # engaged and executing
    YELLOW = "YELLOW"  # idle, waiting for the operator, or a warning
    RED = "RED"  # E-stop, protective stop or fault


@dataclass
class SignalLamp:
    """Derives the lamp from the cell state and forwards it to a sink (the sim geoms)."""

    sink: Callable[[str], None] | None = None
    clock: Callable[[], float] | None = None
    state: LampState = LampState.YELLOW
    history: list[tuple[float, str, str]] = field(default_factory=list)

    def update(
        self,
        *,
        engaged: bool,
        executing: bool,
        fault: bool,
        warning: bool = False,
        reason: str = "",
    ) -> LampState:
        if fault:
            new = LampState.RED
        elif engaged and executing and not warning:
            new = LampState.GREEN
        else:
            new = LampState.YELLOW
        return self.set(new, reason)

    def set(self, new: LampState, reason: str = "") -> LampState:
        if new != self.state:
            t = self.clock() if self.clock else 0.0
            self.history.append((t, new.value, reason))
            log.debug("lamp %s -> %s (%s)", self.state.value, new.value, reason)
        self.state = new
        if self.sink:
            self.sink(new.value)
        return new
