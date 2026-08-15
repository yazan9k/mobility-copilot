"""The agent loop.

Deliberately a plain while-loop rather than a framework. The whole control flow
is `run()` below: ask the model, run whatever tools it asked for, feed the
results back, repeat until it answers or we hit the turn limit.

`run()` returns the tool-call trajectory alongside the reply. That is not
instrumentation bolted on afterwards — trajectory-level evaluation is a primary
metric, so the trace is part of the return type.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

from agent import escalation_gate, llm, prompts, tools
from config import AGENT_VERSION, ESCALATION_GATE_ENABLED, MAX_AGENT_TURNS


@dataclass
class TraceEntry:
    """One executed tool call.

    `result` is the full tool output, not a preview. Faithfulness scoring needs
    the exact grounding the model was given; truncating here would make the
    metric measure our truncation rather than the agent. The API layer
    shortens it for display.
    """
    turn: int
    name: str
    arguments: dict[str, Any]
    result: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    reply: str
    tool_calls: list[TraceEntry]
    retrieved_docs: list[str]
    turns: int
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int
    version: str
    hit_turn_limit: bool = False
    gate_decision: dict[str, Any] | None = None

    def tool_names(self) -> list[str]:
        return [t.name for t in self.tool_calls]

    def grounding_context(self) -> list[str]:
        """Everything the tools returned — what the reply should be faithful to."""
        return [t.result for t in self.tool_calls]

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "tool_names": self.tool_names()}


def run(
    message: str,
    history: list[dict[str, str]] | None = None,
    version: str | None = None,
) -> AgentResult:
    """Answer one user message, using tools as needed."""
    version = version or AGENT_VERSION
    schemas = tools.build_tool_schemas(version)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": prompts.system_prompt(version)},
        *(history or []),
        {"role": "user", "content": message},
    ]

    trace: list[TraceEntry] = []
    retrieved_docs: list[str] = []
    prompt_tokens = completion_tokens = 0
    started = time.perf_counter()
    hit_limit = False
    reply = ""
    gate_decision = None

    # Escalation decision gate. Runs before the loop and calls the tool in code,
    # because four prompt formulations across two model architectures could not
    # get the model to reliably call it itself. See agent/escalation_gate.py.
    #
    # The escalation still enters the trace as a normal tool call, so trajectory
    # and escalation metrics score it exactly as they would a model-initiated
    # one — the gate changes how the decision is made, not how it is measured.
    if ESCALATION_GATE_ENABLED:
        decision = escalation_gate.decide(message, history)
        gate_decision = decision.as_dict()
        if decision.needs_human:
            result = tools.dispatch(
                "escalate_to_human",
                {"reason": decision.reason or f"Escalation rule: {decision.rule}"},
            )
            trace.append(
                TraceEntry(
                    turn=0,  # turn 0: decided before the model's first turn
                    name="escalate_to_human",
                    arguments={"reason": decision.reason},
                    result=result.display,
                    meta={**result.meta, "source": "gate", "rule": decision.rule},
                )
            )
            # Tell the loop what already happened, so it answers the answerable
            # part rather than re-escalating or refusing the whole message.
            messages.append({
                "role": "system",
                "content": (
                    "This question has already been escalated to a Global Mobility "
                    "adviser, and the employee has been told. Do not call "
                    "escalate_to_human. Answer any part of their message that "
                    "policy does cover, and say nothing about the part that was "
                    "escalated."
                ),
            })

    for turn in range(1, MAX_AGENT_TURNS + 1):
        response = llm.chat(messages, tools=schemas)
        prompt_tokens += response.prompt_tokens
        completion_tokens += response.completion_tokens

        # No tool calls means the model is answering.
        if not response.tool_calls:
            reply = response.text
            break

        messages.append(
            {
                "role": "assistant",
                "content": response.text,
                "tool_calls": [
                    {"function": {"name": c.name, "arguments": c.arguments}}
                    for c in response.tool_calls
                ],
            }
        )

        for call in response.tool_calls:
            result = tools.dispatch(call.name, call.arguments)
            retrieved_docs.extend(result.meta.get("retrieved_docs", []))
            trace.append(
                TraceEntry(
                    turn=turn,
                    name=call.name,
                    arguments=call.arguments,
                    result=result.display,
                    meta=result.meta,
                )
            )
            messages.append(
                {"role": "tool", "tool_name": call.name, "content": result.display}
            )
    else:
        # Turn limit reached with tools still being called. Ask once more with
        # no tools so the employee gets an answer from what we already have.
        hit_limit = True
        final = llm.chat(messages + [
            {
                "role": "user",
                "content": "Answer now using the information gathered so far.",
            }
        ])
        prompt_tokens += final.prompt_tokens
        completion_tokens += final.completion_tokens
        reply = final.text

    return AgentResult(
        reply=reply,
        tool_calls=trace,
        retrieved_docs=list(dict.fromkeys(retrieved_docs)),  # dedupe, keep order
        turns=min(turn, MAX_AGENT_TURNS),
        latency_ms=int((time.perf_counter() - started) * 1000),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        version=version,
        hit_turn_limit=hit_limit,
        gate_decision=gate_decision,
    )
