"""Provider seam.

The rest of the codebase talks to this module, not to Ollama. Swapping to a
hosted API (Anthropic, OpenAI) means reimplementing `chat()` and changing the
model constants in config.py — no changes to the agent loop, tools, or evals.

The brief specifies the Anthropic API; we run local models to meet a zero-cost
constraint. Keeping the boundary explicit is what makes that a reversible
decision rather than a fork.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import ollama

from config import AGENT_MODEL, AGENT_TEMPERATURE, OLLAMA_HOST


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


_client: ollama.Client | None = None


def _get_client() -> ollama.Client:
    global _client
    if _client is None:
        _client = ollama.Client(host=OLLAMA_HOST)
    return _client


def chat(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    model: str | None = None,
    temperature: float | None = None,
) -> LLMResponse:
    """One turn against the model. Returns text and any requested tool calls."""
    start = time.perf_counter()
    response = _get_client().chat(
        model=model or AGENT_MODEL,
        messages=messages,
        tools=tools or None,
        options={
            "temperature": AGENT_TEMPERATURE if temperature is None else temperature
        },
    )
    latency_ms = int((time.perf_counter() - start) * 1000)

    message = response.get("message", {}) or {}
    raw_calls = message.get("tool_calls") or []

    tool_calls = [
        ToolCall(
            name=call["function"]["name"],
            arguments=dict(call["function"].get("arguments") or {}),
        )
        for call in raw_calls
    ]

    return LLMResponse(
        text=(message.get("content") or "").strip(),
        tool_calls=tool_calls,
        prompt_tokens=response.get("prompt_eval_count", 0) or 0,
        completion_tokens=response.get("eval_count", 0) or 0,
        latency_ms=latency_ms,
    )


def generate_json(prompt: str, schema: dict, model: str, temperature: float = 0.0) -> str:
    """Constrained generation against a JSON schema.

    Used by the eval judge. Passing the schema to Ollama's `format` parameter
    constrains decoding, so a small local model cannot emit unparseable output.
    Without this, judge score extraction fails often enough to make LLM-scored
    metrics unusable.
    """
    response = _get_client().chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        format=schema,
        options={"temperature": temperature},
    )
    return (response.get("message", {}) or {}).get("content", "") or ""


def health_check() -> tuple[bool, str]:
    """Confirm Ollama is reachable and the agent model is present."""
    try:
        available = {m.get("model", "") for m in _get_client().list().get("models", [])}
    except Exception as exc:  # noqa: BLE001 - surfaced to the caller as a message
        return False, f"Cannot reach Ollama at {OLLAMA_HOST}: {exc}"

    if AGENT_MODEL not in available:
        return False, (
            f"Model {AGENT_MODEL!r} not found. Run: ollama pull {AGENT_MODEL}\n"
            f"Available: {sorted(available) or '(none)'}"
        )
    return True, f"Ollama OK · agent model {AGENT_MODEL}"
