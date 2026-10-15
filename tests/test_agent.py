"""Operator agent: parser, guardrails, providers, fallback, audit log and the stop override."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from cobotsort.agent.guardrails import GuardrailError, Guardrails
from cobotsort.agent.operator import OperatorAgent
from cobotsort.agent.providers.anthropic_provider import AnthropicProvider
from cobotsort.agent.providers.mock import parse_order
from cobotsort.agent.tools import TOOLS, ToolExecutor
from cobotsort.cell.sorter import build_cell
from cobotsort.config import AgentConfig


@pytest.fixture
def cell():
    return build_cell("oracle", seed=11)  # 6 cubes: 3 blue, 2 red, 1 defect


@pytest.mark.parametrize(
    ("order", "action", "classes", "bin_"),
    [
        ("Separe as peças por cor e mande as defeituosas para o rejeito.", "sort", None, None),
        ("Sort everything by color, defects to reject.", "sort", None, None),
        ("Only the red ones to bin B.", "sort", {"red_cube"}, "B"),
        ("Coloque só as peças azuis na caixa C.", "sort", {"blue_cube"}, "C"),
        ("Mande os roxos para o rejeito", "sort", {"defect"}, None),
        ("all the green ones", "sort", {"green_cube"}, None),
        ("Put the cubes in bin D.", "refuse", None, "D"),
        ("Move that one over there.", "ask", None, None),
        ("pare", "stop", None, None),
    ],
)
def test_mock_parser(order, action, classes, bin_):
    intent = parse_order(order)
    assert intent.action == action
    assert intent.classes == classes
    assert intent.bin_override == bin_


def test_language_detection():
    assert parse_order("Separe todas as peças por cor.").language == "pt"
    assert parse_order("Sort everything by color.").language == "en"


def test_tool_schemas_are_small_and_explicit():
    names = [t["name"] for t in TOOLS]
    assert names == ["scan_workspace", "get_cell_status", "pick_and_place", "stop_cell"]
    pick = TOOLS[2]["input_schema"]
    assert pick["required"] == ["scan_id", "object_id", "bin"]
    assert pick["properties"]["bin"]["enum"] == ["A", "B", "C", "REJECT"]


def test_guardrails_validate_against_the_live_cell(cell):
    rails = Guardrails(cell, AgentConfig())
    with pytest.raises(GuardrailError, match="no scan"):
        rails.validate_pick("scan-1", 0, "A")
    scan = cell.scan()
    with pytest.raises(GuardrailError) as e:
        rails.validate_pick("scan-0", 0, "A")
    assert e.value.code == "stale_scan"
    with pytest.raises(GuardrailError) as e:
        rails.validate_pick(scan.scan_id, 42, "A")
    assert e.value.code == "unknown_object"
    with pytest.raises(GuardrailError) as e:
        rails.validate_pick(scan.scan_id, "0", "A")
    assert e.value.code == "unknown_object"
    with pytest.raises(GuardrailError) as e:
        rails.validate_pick(scan.scan_id, 0, "D")
    assert e.value.code == "unknown_bin"
    scan.detections[0].confidence = 0.3
    with pytest.raises(GuardrailError) as e:
        rails.validate_pick(scan.scan_id, 0, "A")
    assert e.value.code == "low_confidence"
    for _ in range(4):
        cell.slots.allocate("C")
    with pytest.raises(GuardrailError) as e:
        rails.validate_pick(scan.scan_id, 1, "C")
    assert e.value.code == "bin_full"
    assert rails.validate_pick(scan.scan_id, 1, "A") is scan.detections[1]


def test_tool_and_pick_budgets(cell):
    cfg = AgentConfig(max_tool_calls=3, max_picks=0)
    ex = ToolExecutor(cell, Guardrails(cell, cfg))
    scan = ex.execute("scan_workspace", {})
    assert scan.guard == "allowed"
    pick = ex.execute(
        "pick_and_place", {"scan_id": scan.content["scan_id"], "object_id": 0, "bin": "A"}
    )
    assert pick.guard == "refused:pick_limit"
    ex.execute("get_cell_status", {})
    assert ex.execute("get_cell_status", {}).guard == "refused:tool_limit"


def test_estop_always_wins(cell):
    ex = ToolExecutor(cell, Guardrails(cell, AgentConfig()))
    scan = ex.execute("scan_workspace", {})
    cell.api.emergency_stop()
    pick = ex.execute(
        "pick_and_place", {"scan_id": scan.content["scan_id"], "object_id": 0, "bin": "A"}
    )
    assert pick.is_error and pick.guard == "refused:estop"
    assert ex.execute("scan_workspace", {}).guard == "refused:estop"


def test_mock_order_end_state_and_audit_log(cell, tmp_path):
    agent = OperatorAgent(cell, AgentConfig(), provider="mock", log_dir=tmp_path)
    res = agent.run("Only the red ones to bin B.")
    for cube in cell.world.cubes():
        expected = "B" if cube.class_name == "red_cube" else None
        assert cell.world.bin_of(cube.pos) == expected
    assert "2 red to bin B" in res.final_text
    rows = [json.loads(line) for line in res.audit_path.read_text(encoding="utf-8").splitlines()]
    tools = [r for r in rows if r["event"] == "tool"]
    assert {"time", "order", "provider", "tool", "input", "result", "guard"} <= set(tools[0])
    assert [r["tool"] for r in tools][:2] == ["scan_workspace", "pick_and_place"]


def test_operator_stop_interrupt_wins(cell, tmp_path):
    agent = OperatorAgent(cell, AgentConfig(), provider="mock", log_dir=tmp_path)

    def interrupt(n_calls: int) -> str | None:
        return "pare!" if n_calls >= 2 else None

    res = agent.run("Separe todas as peças por cor.", interrupt=interrupt)
    assert res.stopped_by_operator
    assert cell.api.lamp.state.value == "RED" and cell.api.fault
    assert sum(1 for c in cell.world.cubes() if cell.world.bin_of(c.pos)) == 1


def test_missing_key_falls_back_to_rule_based_planner(cell, tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    agent = OperatorAgent(cell, AgentConfig(), provider="anthropic", log_dir=tmp_path)
    res = agent.run("Put the cubes in bin D.")
    assert res.provider_used == "mock"
    assert "ANTHROPIC_API_KEY" in res.fallback_reason
    events = [
        json.loads(x)["event"] for x in res.audit_path.read_text(encoding="utf-8").splitlines()
    ]
    assert "fallback" in events


# ---------------------------------------------------------------- anthropic loop (fake client)
def _block(**kw):
    return SimpleNamespace(**kw)


class FakeMessages:
    def __init__(self, script):
        self.script = list(script)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _response(stop_reason, *blocks):
    return SimpleNamespace(stop_reason=stop_reason, content=list(blocks))


def test_anthropic_tool_loop_returns_errors_to_the_model(cell, tmp_path):
    fake = FakeMessages(
        [
            _response(
                "tool_use",
                _block(type="text", text="Scanning."),
                _block(type="tool_use", id="tu_1", name="scan_workspace", input={}),
            ),
            _response(
                "tool_use",
                _block(
                    type="tool_use",
                    id="tu_2",
                    name="pick_and_place",
                    input={"scan_id": "scan-1", "object_id": 99, "bin": "A"},
                ),
            ),
            _response(
                "end_turn", _block(type="text", text="Object 99 does not exist; nothing moved.")
            ),
        ]
    )
    provider = AnthropicProvider(client=SimpleNamespace(beta=SimpleNamespace(messages=fake)))
    res = OperatorAgent(cell, AgentConfig(), provider=provider, log_dir=tmp_path).run(
        "move 99 to A"
    )
    assert res.final_text.startswith("Object 99")
    first = fake.calls[0]
    assert first["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
    assert first["model"] == "claude-sonnet-5-5"
    assert first["fallbacks"] == "default"
    assert first["betas"] == ["server-side-fallback-2026-07-01"]
    # The refused pick goes back as an error tool_result with the matching id.
    last_user = fake.calls[2]["messages"][-1]["content"][0]
    assert last_user["type"] == "tool_result" and last_user["tool_use_id"] == "tu_2"
    assert last_user["is_error"] is True
    assert json.loads(last_user["content"])["code"] == "unknown_object"
    # The assistant content (with its tool_use block) is appended unchanged.
    assert fake.calls[1]["messages"][1]["role"] == "assistant"
    assert all(cell.world.bin_of(c.pos) is None for c in cell.world.cubes())


def test_anthropic_api_error_mid_order_switches_to_fallback(cell, tmp_path):
    import anthropic
    import httpx2

    error = anthropic.APIConnectionError(
        request=httpx2.Request("POST", "https://api.anthropic.com")
    )
    fake = FakeMessages(
        [
            _response(
                "tool_use", _block(type="tool_use", id="tu_1", name="scan_workspace", input={})
            ),
            error,
        ]
    )
    provider = AnthropicProvider(client=SimpleNamespace(beta=SimpleNamespace(messages=fake)))
    res = OperatorAgent(cell, AgentConfig(), provider=provider, log_dir=tmp_path).run(
        "Only the red ones to bin B."
    )
    assert res.provider_used == "mock" and "connection" in res.fallback_reason.lower()
    assert sum(1 for c in cell.world.cubes() if cell.world.bin_of(c.pos) == "B") == 2


def test_anthropic_refusal_switches_to_fallback(cell, tmp_path, monkeypatch):
    monkeypatch.setenv("COBOTSORT_SERVER_FALLBACK", "0")
    fake = FakeMessages([_response("refusal", _block(type="text", text=""))])
    provider = AnthropicProvider(client=SimpleNamespace(messages=fake))
    res = OperatorAgent(cell, AgentConfig(), provider=provider, log_dir=tmp_path).run(
        "Put the cubes in bin D."
    )
    assert "fallbacks" not in fake.calls[0]
    assert res.provider_used == "mock" and "refusal" in res.fallback_reason
