"""Bin slot allocation: each bin has a 2x2 grid; slots are used in order."""

from __future__ import annotations

from dataclasses import dataclass

from cobotsort.config import SceneConfig


class BinFullError(RuntimeError):
    """No free slot left in the requested bin."""


class UnknownBinError(KeyError):
    """The requested bin does not exist."""


@dataclass(frozen=True)
class Slot:
    bin: str
    index: int  # 1..4
    x: float
    y: float

    @property
    def label(self) -> str:
        return f"{self.bin}{self.index}"


class SlotAllocator:
    """Hands out slots in order and refuses moves into a full bin."""

    def __init__(self, scene: SceneConfig | None = None) -> None:
        scene = scene or SceneConfig()
        p = scene.slot_pitch / 2
        offsets = [(-p, p), (p, p), (-p, -p), (p, -p)]
        self.slots: dict[str, list[Slot]] = {
            b.name: [Slot(b.name, i + 1, b.x + dx, b.y + dy) for i, (dx, dy) in enumerate(offsets)]
            for b in scene.bins
        }
        self.used: dict[str, int] = {name: 0 for name in self.slots}

    @property
    def bins(self) -> list[str]:
        return list(self.slots)

    def _check(self, bin_name: str) -> None:
        if bin_name not in self.slots:
            raise UnknownBinError(bin_name)

    def free(self, bin_name: str) -> int:
        self._check(bin_name)
        return len(self.slots[bin_name]) - self.used[bin_name]

    def free_by_bin(self) -> dict[str, int]:
        return {b: self.free(b) for b in self.slots}

    def peek(self, bin_name: str) -> Slot:
        """Next slot of ``bin_name`` without reserving it."""
        if self.free(bin_name) <= 0:
            raise BinFullError(f"bin {bin_name} is full")
        return self.slots[bin_name][self.used[bin_name]]

    def allocate(self, bin_name: str) -> Slot:
        slot = self.peek(bin_name)
        self.used[bin_name] += 1
        return slot

    def release(self, slot: Slot) -> None:
        """Give back the last allocated slot of a bin (e.g. the pick failed)."""
        if self.used[slot.bin] and self.slots[slot.bin][self.used[slot.bin] - 1] == slot:
            self.used[slot.bin] -= 1
