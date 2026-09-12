"""One-shot provider-independent planner for `omarchy-voice say`.

Used by the turn-based pipeline and the typed CLI. The integrated Realtime
mode keeps its own transport. All modes share the same desktop policy gate.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

from . import capabilities
from .config import Config
from .persona import PERSONA
from .providers import ProviderError, brain_endpoint, chat_completion
from .tools import TOOL_SCHEMAS, Executor, tools_for

@dataclass
class Turn:
    """One request and everything that came of it."""
    text: str
    reply: str = ""
    actions: list[str] = field(default_factory=list)
    error: str = ""
    elapsed: float = 0.0
    tokens: dict = field(default_factory=dict)


def to_chat_tools(schemas: list[dict] | None = None) -> list[dict]:
    converted = []
    for schema in schemas if schemas is not None else TOOL_SCHEMAS:
        converted.append({
            "type": "function",
            "function": {
                "name": schema["name"],
                "description": schema["description"],
                "parameters": schema["input_schema"],
            },
        })
    return converted


def _system_prompt() -> str:
    return "\n\n".join([
        PERSONA,
        capabilities.manifest(),
        "# The desktop right now\n\n" + capabilities.live_state(),
    ])


class PlannerUnavailable(RuntimeError):
    """Something the one-shot planner needs is missing."""


class Planner:
    def __init__(self, config: Config, executor: Executor):
        self.config = config
        self.executor = executor

    def think(self, text: str, cancelled=None) -> Turn:
        turn = Turn(text=text)
        started = time.monotonic()
        try:
            turn.reply = self._loop(text, turn, cancelled)
        except (PlannerUnavailable, ProviderError) as exc:
            turn.error = str(exc)
            turn.reply = "My planner isn't configured yet."
        except Exception as exc:  # a voice tool must not die on one bad turn
            turn.error = f"{type(exc).__name__}: {exc}"
            turn.reply = "Something went wrong with that."
        turn.elapsed = time.monotonic() - started
        return turn

    def _loop(self, text: str, turn: Turn, cancelled=None) -> str:
        endpoint = brain_endpoint(self.config)
        if endpoint.key_env and not os.environ.get(endpoint.key_env):
            raise PlannerUnavailable(f"{endpoint.key_env} is not set — run omarchy-voice setup")
        cancelled = cancelled or (lambda: False)
        key = os.environ.get(endpoint.key_env, "")

        messages: list[dict] = [
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": text},
        ]
        tools = to_chat_tools(tools_for(self.config))
        reply = ""

        for _ in range(self.config.max_turns):
            if cancelled():
                return "Cancelled."
            data = _chat(messages, tools, self.config, key)
            if cancelled():
                return "Cancelled."
            usage = data.get("usage") or {}
            if usage:
                turn.tokens = {
                    "in": turn.tokens.get("in", 0) + usage.get("prompt_tokens", 0),
                    "out": turn.tokens.get("out", 0) + usage.get("completion_tokens", 0),
                }
            choices = data.get("choices")
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                raise ProviderError("Brain returned no completion choices")
            choice = choices[0]
            message = choice.get("message")
            if not isinstance(message, dict):
                raise ProviderError("Brain returned an invalid assistant message")
            content = message.get("content") or ""
            if not isinstance(content, str):
                raise ProviderError("Brain returned non-text assistant content")
            said = content.strip()
            if said:
                reply = said
            tool_calls = message.get("tool_calls") or []
            if not isinstance(tool_calls, list) or any(not isinstance(call, dict) for call in tool_calls):
                raise ProviderError("Brain returned invalid tool calls")
            if not tool_calls:
                return reply or "Done."

            # Keep opaque metadata (e.g. Gemini thought signatures) intact.
            messages.append({**message, "role": "assistant"})
            for call in tool_calls:
                if cancelled():
                    return "Cancelled."
                fn = call.get("function") or {}
                name = fn.get("name", "")
                raw_args = fn.get("arguments") or "{}"
                try:
                    args = json.loads(raw_args)
                    if not isinstance(args, dict):
                        raise ValueError("tool arguments must be an object")
                    if name not in {item["function"]["name"] for item in tools}:
                        raise ValueError("tool was not offered in this request")
                except (ValueError, TypeError) as exc:
                    outcome_text = f"ERROR: could not parse arguments: {exc}"
                else:
                    outcome = self.executor.call(name, args)
                    turn.actions.append(self.executor.describe(name, args))
                    outcome_text = outcome.as_tool_result()
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "content": outcome_text,
                })
                if self.executor.pending:
                    return "That needs local confirmation: omarchy-voice listen confirm."
            if self.executor.pending:
                return reply or "That needs confirmation."

        turn.error = "Maximum tool rounds reached; the task may be incomplete"
        return "I reached the step limit. Please check the result before continuing."


def _chat(messages: list[dict], tools: list[dict], config: Config, key: str = "") -> dict:
    """The key argument remains for existing callers; adapters own credentials."""
    return chat_completion(messages, tools, config)
