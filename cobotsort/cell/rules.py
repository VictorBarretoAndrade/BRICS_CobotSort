"""Class -> bin mapping."""

from __future__ import annotations

from dataclasses import dataclass, field

from cobotsort.perception.detector import Detection

DEFAULT_RULES: dict[str, str] = {
    "red_cube": "A",
    "green_cube": "B",
    "blue_cube": "C",
    "defect": "REJECT",
}


@dataclass
class SortingRules:
    """Which bin each class goes to, and the confidence below which a part is left alone."""

    mapping: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_RULES))
    min_confidence: float = 0.6

    def bin_for(self, det: Detection) -> str | None:
        """Target bin, or None if the part must stay in place and be flagged."""
        if det.confidence < self.min_confidence:
            return None
        return self.mapping.get(det.class_name)

    def why_not(self, det: Detection) -> str:
        if det.confidence < self.min_confidence:
            return f"low confidence ({det.confidence:.2f} < {self.min_confidence})"
        if det.class_name not in self.mapping:
            return f"unknown class {det.class_name!r}"
        return ""
