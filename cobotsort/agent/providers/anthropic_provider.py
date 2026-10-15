"""Claude via the Anthropic Messages API, with a manual tool-use loop.

* One physical action per turn: ``tool_choice={"type": "auto", "disable_parallel_tool_use": True}``.
* The assistant content is appended back unchanged (it may contain thinking blocks), and each
  tool result goes back as a ``tool_result`` block with the matching ``tool_use_id`` and
  ``is_error: true`` when a guardrail refused the call.
* Server-side refusal fallbacks are on by default (``fallbacks: "default"``, beta
  ``server-side-fallback-2026-07-01``); set ``COBOTSORT_SERVER_FALLBACK=0`` to turn them off.
* Any API error, timeout, refusal or missing key raises :class:`ProviderError`; the operator
  then switches to the rule-based fallback planner and logs why.
"""

from __future__ import annotations

import json
import logging
import os

from cobotsort.agent.providers.base import AssistantTurn, Provider, ProviderError, ToolCall

log = logging.getLogger(__name__)


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(
        self,
        model: str = "claude-sonnet-5-5",
        max_tokens: int = 16000,
        timeout_s: float = 60.0,
        effort: str | None = None,
        client=None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort or os.getenv("COBOTSORT_EFFORT", "medium")
        self.server_fallback = os.getenv("COBOTSORT_SERVER_FALLBACK", "1") != "0"
        if client is None:
            if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
                raise ProviderError("ANTHROPIC_API_KEY is not set")
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - dependency is declared
                raise ProviderError(f"anthropic SDK not installed: {exc}") from exc
            client = anthropic.Anthropic(timeout=timeout_s, max_retries=1)
        self.client = client
        self.system = ""
        self.tools: list[dict] = []
        self.messages: list[dict] = []

    def start(self, system: str, order: str, tools: list[dict]) -> None:
        self.system = system
        self.tools = tools
        self.messages = [{"role": "user", "content": order}]

    def next_turn(self) -> AssistantTurn:
        import anthropic

        try:
            params = dict(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.system,
                tools=self.tools,
                tool_choice={"type": "auto", "disable_parallel_tool_use": True},
                output_config={"effort": self.effort},
                messages=self.messages,
            )
            if self.server_fallback:
                # Server-side refusal fallback: a declined request is re-run on a fallback
                # model inside the same call. Our rule-based planner is the last resort.
                response = self.client.beta.messages.create(
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default", **params
                )
            else:
                response = self.client.messages.create(**params)
        except anthropic.APITimeoutError as exc:
            raise ProviderError(f"API timeout: {exc}") from exc
        except anthropic.APIConnectionError as exc:
            raise ProviderError(f"API connection error: {exc}") from exc
        except anthropic.RateLimitError as exc:
            raise ProviderError(f"rate limited: {exc}") from exc
        except anthropic.APIStatusError as exc:
            raise ProviderError(f"API error {exc.status_code}: {exc.message}") from exc
        if response.stop_reason == "refusal":
            raise ProviderError("the model declined the request (stop_reason=refusal)")
        if response.stop_reason == "max_tokens":
            raise ProviderError("response hit max_tokens")
        # Keep the full content (thinking blocks included) for the next request.
        self.messages.append({"role": "assistant", "content": response.content})
        text = " ".join(b.text for b in response.content if b.type == "text").strip()
        call = next((b for b in response.content if b.type == "tool_use"), None)
        if response.stop_reason == "tool_use" and call is not None:
            tool_input = call.input if isinstance(call.input, dict) else json.loads(call.input)
            return AssistantTurn(text, ToolCall(call.id, call.name, tool_input), "tool_use")
        return AssistantTurn(text, None, response.stop_reason or "end_turn")

    def add_tool_result(self, call: ToolCall, content: dict, is_error: bool) -> None:
        block = {
            "type": "tool_result",
            "tool_use_id": call.id,
            "content": json.dumps(content),
        }
        if is_error:
            block["is_error"] = True
        self.messages.append({"role": "user", "content": [block]})
