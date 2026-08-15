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

from config import (
    AGENT_MODEL,
    AGENT_NUM_CTX,
    AGENT_SEED,
    AGENT_TEMPERATURE,
    LLM_BACKEND,
    OLLAMA_HOST,
    OPENAI_BASE_URL,
)


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


def _to_openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate the agent loop's message history into strict OpenAI shape.

    The loop writes tool calls in Ollama's dialect — `{"function": {...}}` with
    the arguments as a dict and no id. The OpenAI schema requires `"type":
    "function"`, a string `arguments`, an `id`, and a matching `tool_call_id` on
    every tool result. llama-server enforces this and returns a 500 otherwise:

        Failed to parse messages: Missing tool call type: {"function": ...}

    Translating here rather than changing the loop keeps the provider detail
    inside the provider seam. The loop stays in one dialect and knows nothing
    about which backend is serving it.
    """
    import json as _json

    out: list[dict[str, Any]] = []
    pending_ids: list[str] = []

    for message in messages:
        if message.get("role") == "assistant" and message.get("tool_calls"):
            pending_ids = []
            calls = []
            for i, call in enumerate(message["tool_calls"]):
                function = call.get("function", call)
                call_id = call.get("id") or f"call_{len(out)}_{i}"
                pending_ids.append(call_id)
                arguments = function.get("arguments") or {}
                calls.append({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": function.get("name", ""),
                        # OpenAI wants a JSON *string* here, Ollama wants a dict.
                        "arguments": arguments if isinstance(arguments, str)
                                     else _json.dumps(arguments),
                    },
                })
            out.append({
                "role": "assistant",
                "content": message.get("content") or "",
                "tool_calls": calls,
            })
        elif message.get("role") == "tool":
            # Results come back in the order the calls were made, so they pair
            # off against the ids minted just above.
            entry = dict(message)
            if "tool_call_id" not in entry:
                entry["tool_call_id"] = pending_ids.pop(0) if pending_ids else "call_0"
            out.append(entry)
        else:
            out.append(message)

    return out


def _chat_openai(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    model: str | None,
    temperature: float | None,
) -> LLMResponse:
    """One turn against an OpenAI-compatible endpoint (llama.cpp's llama-server).

    Exists because Ling 3.0 Tiny uses the `bailingmoe3` architecture, which
    Ollama cannot load — it needs a patched llama.cpp build. Rather than fork the
    agent loop, the second backend lands here, behind the same LLMResponse the
    rest of the codebase already consumes.

    The two differences from the Ollama path that matter:
      * tool-call arguments arrive as a JSON *string*, not a dict, so they are
        parsed here. Leaving that to the caller would push a provider detail into
        the agent loop.
      * `num_ctx` is fixed when the server starts, not per request, so it is a
        server flag rather than an option here. Temperature and seed still are.
    """
    import json as _json
    import urllib.request

    payload = {
        "model": model or AGENT_MODEL,
        "messages": _to_openai_messages(messages),
        "temperature": AGENT_TEMPERATURE if temperature is None else temperature,
        "seed": AGENT_SEED,
    }
    if tools:
        payload["tools"] = tools

    request = urllib.request.Request(
        f"{OPENAI_BASE_URL}/chat/completions",
        data=_json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )

    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=600) as response:
        body = _json.loads(response.read())
    latency_ms = int((time.perf_counter() - start) * 1000)

    message = body["choices"][0].get("message", {}) or {}
    tool_calls = []
    for call in message.get("tool_calls") or []:
        function = call.get("function", {})
        raw_args = function.get("arguments") or "{}"
        try:
            arguments = _json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
        except _json.JSONDecodeError:
            # A malformed argument blob is a real model failure and should be
            # scored as one, not crash the run 40 cases in.
            arguments = {}
        tool_calls.append(ToolCall(name=function.get("name", ""), arguments=arguments))

    usage = body.get("usage") or {}
    return LLMResponse(
        text=(message.get("content") or "").strip(),
        tool_calls=tool_calls,
        prompt_tokens=usage.get("prompt_tokens", 0) or 0,
        completion_tokens=usage.get("completion_tokens", 0) or 0,
        latency_ms=latency_ms,
    )


def chat(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    model: str | None = None,
    temperature: float | None = None,
) -> LLMResponse:
    """One turn against the model. Returns text and any requested tool calls."""
    if LLM_BACKEND == "openai":
        return _chat_openai(messages, tools, model, temperature)

    start = time.perf_counter()
    response = _get_client().chat(
        model=model or AGENT_MODEL,
        messages=messages,
        tools=tools or None,
        options={
            "temperature": AGENT_TEMPERATURE if temperature is None else temperature,
            # Explicit: Ollama silently truncates past its 4096 default rather
            # than erroring, which drops the system prompt and tool definitions
            # on long requests. See AGENT_NUM_CTX in config.py.
            "num_ctx": AGENT_NUM_CTX,
            # Fixed seed: without it, two runs of the same version differed by
            # up to 15pp on retrieval recall, which is larger than the effect
            # we are trying to measure. See AGENT_SEED in config.py.
            "seed": AGENT_SEED,
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

    Used by the eval judge and the escalation gate. Passing the schema to the
    backend constrains decoding, so a small local model cannot emit unparseable
    output. Without this, judge score extraction fails often enough to make
    LLM-scored metrics unusable — and the gate would be able to narrate instead
    of deciding, which is the exact failure it exists to remove.
    """
    if LLM_BACKEND == "openai":
        import json as _json
        import urllib.request

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "seed": AGENT_SEED,
            # Thinking mode stays ON here, which is a measured decision.
            #
            # Ling emits reasoning tokens before the JSON, and on harder
            # questions they exhausted a 3000-token budget, returning empty —
            # 17 of 90 gate calls, ~50s each. Turning thinking off fixes that
            # completely (1s per call, zero failures) and destroys the gate's
            # judgement: escalation recall goes to 5/5 but false alarms go to
            # 4/5, labelling document checklists HIGH_CONSEQUENCE. A gate that
            # escalates everything is worse than no gate.
            #
            # So the budget goes up instead. Thinking is what makes the
            # discrimination work; it just needs room to finish.
            # Bound runaway generation without starving the model's reasoning.
            #
            # One gate call ran 224 seconds for output that is a boolean, an
            # enum, and a sentence. The obvious fix — cap at 400 tokens, several
            # times what the schema can use — destroyed the gate: escalation
            # recall went to 0/5 with every call returning a uniform ~4.5s
            # `false`. Ling 3.0 is a reasoning model, and the visible JSON is
            # only the tail of its generation; capping near the schema's size
            # truncates the thinking that produces the answer, and what remains
            # is a well-formed but degenerate response.
            #
            # 3000 leaves room to reason while still bounding the outlier.
            "max_tokens": 6000,
            # llama-server enforces the schema via GBNF, the same guarantee
            # Ollama's `format` gives.
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "response", "strict": True, "schema": schema},
            },
        }
        request = urllib.request.Request(
            f"{OPENAI_BASE_URL}/chat/completions",
            data=_json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            body = _json.loads(response.read())
        return (body["choices"][0].get("message", {}) or {}).get("content", "") or ""

    response = _get_client().chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        format=schema,
        options={"temperature": temperature},
    )
    return (response.get("message", {}) or {}).get("content", "") or ""


def health_check() -> tuple[bool, str]:
    """Confirm the configured backend is reachable and serving a model."""
    if LLM_BACKEND == "openai":
        import json as _json
        import urllib.request
        try:
            with urllib.request.urlopen(f"{OPENAI_BASE_URL}/models", timeout=10) as r:
                served = [m.get("id") for m in _json.loads(r.read()).get("data", [])]
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller
            return False, (
                f"Cannot reach an OpenAI-compatible server at {OPENAI_BASE_URL}: {exc}\n"
                f"Start llama-server (see LLM_BACKEND in config.py)."
            )
        return True, f"llama-server OK · {OPENAI_BASE_URL} · serving {served or '(unnamed)'}"

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
