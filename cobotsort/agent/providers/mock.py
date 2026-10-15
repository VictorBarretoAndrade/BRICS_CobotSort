"""Deterministic provider: a Portuguese/English keyword parser plus a rule-based planner.

It is the default provider (no API key needed) and also the fallback planner when the
Anthropic provider fails. It speaks the same tool protocol as a language model would.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from cobotsort.agent.providers.base import AssistantTurn, Provider, ToolCall
from cobotsort.cell.rules import DEFAULT_RULES

VALID_BINS = ("A", "B", "C", "REJECT")
CLASS_PATTERNS = {
    "red_cube": r"\b(vermelh\w*|red)\b",
    "green_cube": r"\b(verde\w*|green)\b",
    "blue_cube": r"\b(azu\w*|blue)\b",
    "defect": r"\b(defeituos\w*|defect\w*|roxo\w*|roxa\w*|purple)\b",
}
# "Everything by color" means every class with the default rules, even if a class is named
# ("sort everything by color, defects to reject"). Plain quantifiers and nouns ("all the red
# ones", "the defective parts") defer to the classes named.
BY_COLOR_PATTERN = r"\b(por cor|by colou?r|everything|tudo)\b"
ALL_PATTERN = r"\b(todos|todas|all|every|cubes|cubos|pecas|parts|pieces)\b"
ONLY_PATTERN = r"\b(only|just|somente|apenas|so)\b"
STOP_PATTERN = r"\b(pare|parar|para tudo|stop|halt|emergencia|emergency)\b"
BIN_PATTERN = r"\b(?:caixa|bin|box|cesto|cesta)\s+([a-z0-9]+)\b"
PT_HINTS = (
    "separe",
    "mande",
    "coloque",
    "caixa",
    "peca",
    "vermelh",
    "verde",
    "azu",
    "roxo",
    "defeituos",
    "todos",
    "todas",
    "tudo",
    "pare",
    "rejeito",
    "por cor",
    "cubos",
    "isso",
    "arruma",
    "so ",
    "apenas",
    "somente",
    "para o",
    "na ",
    "no ",
)
CLASS_LABELS = {
    # (singular, plural)
    "pt": {
        "red_cube": ("vermelha", "vermelhas"),
        "green_cube": ("verde", "verdes"),
        "blue_cube": ("azul", "azuis"),
        "defect": ("defeituosa", "defeituosas"),
    },
    "en": {
        "red_cube": ("red", "red"),
        "green_cube": ("green", "green"),
        "blue_cube": ("blue", "blue"),
        "defect": ("defective", "defective"),
    },
}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def detect_language(text: str) -> str:
    t = normalize(text) + " "
    return "pt" if sum(h in t for h in PT_HINTS) >= 1 else "en"


@dataclass
class OrderIntent:
    action: str  # sort | stop | ask | refuse
    language: str
    classes: set[str] | None = None  # None = every class
    bin_override: str | None = None
    reason: str = ""


def parse_order(order: str) -> OrderIntent:
    """Turn a work order into an intent with keyword rules (Portuguese and English)."""
    t = normalize(order)
    lang = detect_language(order)
    if re.search(STOP_PATTERN, t):
        return OrderIntent("stop", lang, reason="stop requested")
    bins = [b.upper() for b in re.findall(BIN_PATTERN, t)]
    bins = ["REJECT" if b in ("REJEITO", "REJECT") else b for b in bins]
    for b in bins:
        if b not in VALID_BINS:
            return OrderIntent("refuse", lang, bin_override=b, reason=f"bin {b} does not exist")
    if len(set(bins)) > 1:
        return OrderIntent("ask", lang, reason="more than one bin named")
    colors = {c for c, pat in CLASS_PATTERNS.items() if re.search(pat, t)}
    by_color = re.search(BY_COLOR_PATTERN, t) is not None
    every = re.search(ALL_PATTERN, t) is not None
    only = re.search(ONLY_PATTERN, t) is not None
    if colors and (only or not by_color):
        classes: set[str] | None = colors
    elif by_color or every:
        classes = None
    else:
        return OrderIntent("ask", lang, reason="no part class or 'all' in the order")
    return OrderIntent("sort", lang, classes, bins[0] if bins else None)


@dataclass
class _PlanState:
    scan_id: str | None = None
    objects: list[dict] = field(default_factory=list)
    need_scan: bool = True
    moved: list[tuple[str, str, str]] = field(default_factory=list)
    refused: list[str] = field(default_factory=list)
    full_bins: set[str] = field(default_factory=set)
    attempts: dict[tuple, int] = field(default_factory=dict)
    scan_errors: int = 0
    stopped: bool = False
    done_reason: str = ""


class MockProvider(Provider):
    """Rule-based planner. Never moves anything itself; it only proposes tool calls."""

    name = "mock"

    def __init__(self) -> None:
        self.intent: OrderIntent | None = None
        self.state = _PlanState()
        self._n = 0

    def start(self, system: str, order: str, tools: list[dict]) -> None:
        self.intent = parse_order(order)
        self.state = _PlanState()

    def _call(self, name: str, args: dict) -> AssistantTurn:
        self._n += 1
        return AssistantTurn("", ToolCall(f"mock_{self._n}", name, args), "tool_use")

    def _target_bin(self, cls: str) -> str | None:
        assert self.intent is not None
        return self.intent.bin_override or DEFAULT_RULES.get(cls)

    def next_turn(self) -> AssistantTurn:
        intent, st = self.intent, self.state
        assert intent is not None
        if intent.action == "ask":
            return AssistantTurn(self._question(), None)
        if intent.action == "refuse":
            return AssistantTurn(self._refusal(), None)
        if intent.action == "stop":
            if not st.stopped:
                return self._call("stop_cell", {"reason": "stop requested in the work order"})
            return AssistantTurn(self._summary(), None)
        if st.done_reason or st.stopped:
            return AssistantTurn(self._summary(), None)
        if st.need_scan:
            return self._call("scan_workspace", {})
        for obj in st.objects:
            if intent.classes is not None and obj["class_name"] not in intent.classes:
                continue
            if obj["confidence"] < 0.6:
                continue
            target = self._target_bin(obj["class_name"])
            if target is None or target in st.full_bins:
                continue
            key = (obj["class_name"], round(obj["x_mm"] / 20), round(obj["y_mm"] / 20))
            if st.attempts.get(key, 0) >= 3:
                continue
            st.attempts[key] = st.attempts.get(key, 0) + 1
            return self._call(
                "pick_and_place", {"scan_id": st.scan_id, "object_id": obj["id"], "bin": target}
            )
        return AssistantTurn(self._summary(), None)

    def add_tool_result(self, call: ToolCall, content: dict, is_error: bool) -> None:
        st = self.state
        code = content.get("code", "")
        if code in ("tool_limit", "pick_limit", "estop", "fault"):
            st.done_reason = content.get("error", code)
            st.refused.append(st.done_reason)
            return
        if call.name == "scan_workspace":
            if is_error:
                st.scan_errors += 1
                if st.scan_errors >= 2:
                    st.done_reason = content.get("error", "scan failed")
                return
            st.scan_id, st.objects, st.need_scan = content["scan_id"], content["objects"], False
        elif call.name == "pick_and_place":
            st.need_scan = True
            if not is_error and content.get("success"):
                st.moved.append((content["class_name"], content["bin"], content["slot"]))
                fresh = content.get("verification_scan")
                if fresh:
                    st.scan_id, st.objects, st.need_scan = fresh["scan_id"], fresh["objects"], False
                return
            if code == "bin_full":
                st.full_bins.add(call.input["bin"])
            st.refused.append(content.get("error") or content.get("detail") or "refused")
            if content.get("status", {}).get("fault"):
                st.done_reason = content["status"]["fault"]
        elif call.name == "stop_cell":
            st.stopped = True

    # ------------------------------------------------------------ texts
    def _question(self) -> str:
        if self.intent and self.intent.language == "pt":
            return (
                "Não entendi qual peça mover nem para onde. Quais peças (vermelhas, verdes, azuis, "
                "defeituosas ou todas) e para qual caixa (A, B, C ou REJECT)?"
            )
        return (
            "I could not tell which parts to move or where. Which parts (red, green, blue, "
            "defective or all) and which bin (A, B, C or REJECT)?"
        )

    def _refusal(self) -> str:
        assert self.intent is not None
        if self.intent.language == "pt":
            return (
                f"Não executei a ordem: a caixa {self.intent.bin_override} não existe. "
                "As caixas são A, B, C e REJECT. Para qual delas devo mandar as peças?"
            )
        return (
            f"I did not run the order: {self.intent.reason}. The bins are A, B, C and REJECT. "
            "Which one should I use?"
        )

    def _summary(self) -> str:
        assert self.intent is not None
        st, lang = self.state, self.intent.language
        labels = CLASS_LABELS[lang]
        counts: dict[tuple[str, str], int] = {}
        for cls, b, _ in st.moved:
            counts[(cls, b)] = counts.get((cls, b), 0) + 1
        if lang == "pt":
            moved = "; ".join(
                f"{n} {labels[c][n > 1]} para a caixa {b}" for (c, b), n in counts.items()
            )
            text = f"Movi {len(st.moved)} peça(s)" + (f": {moved}." if moved else ".")
            if st.stopped:
                text += " Célula parada a pedido (lâmpada vermelha)."
            if st.full_bins:
                full = ", ".join(sorted(st.full_bins))
                text += f" Caixa(s) cheia(s): {full}; peças restantes ficaram no lugar."
            if st.refused and not st.stopped:
                text += f" Recusado/falhou: {len(st.refused)} ({st.refused[-1]})."
            return text
        moved = "; ".join(f"{n} {labels[c][n > 1]} to bin {b}" for (c, b), n in counts.items())
        text = f"Moved {len(st.moved)} part(s)" + (f": {moved}." if moved else ".")
        if st.stopped:
            text += " Cell stopped on request (lamp RED)."
        if st.full_bins:
            full = ", ".join(sorted(st.full_bins))
            text += f" Full bin(s): {full}; remaining parts were left in place."
        if st.refused and not st.stopped:
            text += f" Refused/failed: {len(st.refused)} ({st.refused[-1]})."
        return text
