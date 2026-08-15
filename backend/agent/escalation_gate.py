"""A schema-constrained escalation decision, made outside the agent loop.

## Why this exists

Escalation recall sat at 20-40% across four prompt formulations (v2, v3, three
v4 arms, v5) and two model architectures (qwen2.5:7b dense, Ling 3.0 Tiny MoE).
Nothing in prompt-space moved it. The failure modes differ by model and neither
is fixable by describing the rule better:

  qwen2.5:7b   10 of 13 misses state in the reply that the question must go to
               an adviser, then never call the tool. esc-004 even asks "Would
               you like to proceed with escalating this request?" — permission
               seeking that every prompt since v2 explicitly forbids. The
               judgement is right; the action does not happen.

  Ling 3.0     only 1 of 7 held-out misses mentions escalation at all. It does
               not recognise the cases; it answers them.

One is an execution failure, the other a recognition failure, and a single
prompt asking one model to answer the question, choose among six tools, and
remember a safety rule fails at whichever of those it is weakest on.

## What this does instead

The decision becomes a separate call with a constrained output schema, and the
tool is invoked in code once the answer comes back.

Two properties follow, and they map onto the two failure modes:

  * Narration becomes impossible. The model must emit `needs_human: true` or
    `false`. There is nowhere for "I should escalate this" to go, so qwen's
    failure mode has no room to occur.

  * Recognition gets undivided attention. One question, five rules, no tool
    catalogue and no answer to compose alongside it. That targets Ling's.

This is the same technique that made the judge usable: constraining the output
rather than hoping for compliance took judge/human agreement from 80% to 95%.

## Known limitation

Invariant 5 ("cannot establish the answer") needs retrieval results to
evaluate — you cannot know policy fails to cover something before looking. The
gate runs before the loop, so it decides on invariants 1-4 only, and 5 stays in
the system prompt. Running it a second time post-retrieval is the obvious
extension and is deliberately not built until the first version is measured.

Source of truth for the rules: docs/escalation_invariants.md.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from agent import llm
from config import AGENT_MODEL

# Deliberately not the system prompt's wording. The gate is answering one
# closed question, so the rules are stated as classification criteria rather
# than as instructions to an assistant.
GATE_PROMPT = """You decide whether an employee's question must be handled by a human Global Mobility adviser, or can be answered from company policy.

Answer "yes" if ANY of these is true:

1. PERSON_SPECIFIC — answering depends on facts about this individual or their family that policy does not establish: their own tax liability, medical circumstances, pension or social security record, or personal financial position. This includes whether a spouse, partner, or dependent may WORK in the destination, because that depends on the dependent's own nationality rather than the employee's. Policy explains how a scheme works; it cannot decide one person's outcome.

2. UNDECIDED — answering depends on an approval, exception, or discretionary decision nobody has made yet: a request to exceed an allowance, extend a covered period, end an assignment early, or vary an entitlement.

3. NOT_AUTHORISED — the question asks about another person's package, pay, immigration status, or benefits. Being their manager or colleague does not by itself authorise it.

4. HIGH_CONSEQUENCE — being wrong causes serious or irreversible harm: a refused, expired, or at-risk visa or permit; a required document that cannot be obtained; a criminal record disclosure; or a significant back-dated tax liability. An assignment extension crossing six months counts, because tax residence can change retroactively.

Answer "no" for everything else. Standard entitlements, allowance amounts, timelines, document lists, what a policy covers or excludes, the difference between assignment types, and a dependent's right to RESIDE are all normal questions with policy answers. Note the split on dependents: whether they may LIVE there is answerable; whether they may WORK there is rule 1.

Do not answer "yes" merely because the question says "I" or "my". The test is whether policy settles it, not whether the topic is personal.

A question can contain both: if any part meets a rule above, answer "yes" — the answerable part is handled separately.

Employee question:
{question}

Reply with JSON only."""

# Field ORDER is load-bearing, and it was measured rather than reasoned about.
#
# Constrained decoding emits properties in schema order, so whatever comes first
# is decided before anything after it exists. The obvious design is reason ->
# rule -> verdict, letting the verdict be conditioned on the reasoning. That was
# tried and it is WORSE: on a 10-case dev set it lost the RSU-taxation and
# colleague's-package cases, both of which the verdict-first schema got right.
# Reasoning first appears to let the model hedge its way to "no".
#
# So the verdict comes first and the reason is a post-hoc label used for
# diagnostics. Not what theory predicts, but it is what the measurements say,
# and this project has already been wrong five times by predicting instead of
# measuring.
GATE_SCHEMA = {
    "type": "object",
    "properties": {
        "needs_human": {"type": "boolean"},
        "rule": {
            "type": "string",
            "enum": [
                "PERSON_SPECIFIC",
                "UNDECIDED",
                "NOT_AUTHORISED",
                "HIGH_CONSEQUENCE",
                "NONE",
            ],
        },
        # maxLength is not cosmetic — without it this field is the whole bug.
        #
        # An unbounded string under constrained decoding generates until it hits
        # the token budget, so `reason` ran to 3000 tokens and was cut mid-string,
        # leaving invalid JSON. 6 of the first 22 gate calls failed that way, each
        # taking ~45 seconds to produce nothing. Because a gate failure defaults
        # to needs_human=False, those cases silently skipped escalation — a 27%
        # failure rate that would have looked like the gate simply not firing.
        #
        # Bounding the field in the schema fixes it where it happens. Raising
        # max_tokens would only move the cliff further out.
        "reason": {"type": "string", "maxLength": 200},
    },
    "required": ["needs_human", "rule", "reason"],
}


@dataclass
class GateDecision:
    needs_human: bool
    rule: str
    reason: str
    latency_ms: int = 0
    failed: bool = False

    def as_dict(self) -> dict:
        return {
            "needs_human": self.needs_human,
            "rule": self.rule,
            "reason": self.reason,
            "latency_ms": self.latency_ms,
            "failed": self.failed,
        }


def _question_with_context(message: str, history: list[dict] | None) -> str:
    """Include prior turns, because follow-ups inherit what triggers a rule.

    "What about the tax side of that?" is only recognisable as a tax question
    when the preceding turn is visible. Multi-turn cases were among the worst
    performers on escalation, and this is the likely reason.
    """
    if not history:
        return message
    prior = "\n".join(
        f"{turn.get('role', 'user')}: {turn.get('content', '')}" for turn in history[-4:]
    )
    return f"Earlier in this conversation:\n{prior}\n\nCurrent question:\n{message}"


def decide(
    message: str,
    history: list[dict] | None = None,
    model: str | None = None,
) -> GateDecision:
    """Classify one question. Never raises — a gate failure must not kill a run.

    On failure it returns needs_human=False, i.e. it defers to the agent loop's
    own behaviour rather than escalating. That is the conservative choice for a
    metric where over-escalation is itself a tracked failure: a broken gate
    leaves the previous behaviour intact instead of routing everything to a
    human.
    """
    import time

    prompt = GATE_PROMPT.format(question=_question_with_context(message, history))
    started = time.perf_counter()
    try:
        raw = llm.generate_json(
            prompt=prompt,
            schema=GATE_SCHEMA,
            model=model or AGENT_MODEL,
            temperature=0.0,
        )
        data = json.loads(raw)
        rule = str(data.get("rule", "NONE"))
        # `needs_human` is authoritative and the rule is a diagnostic label.
        #
        # The first version derived the boolean from the rule instead, on the
        # theory that a named rule implies escalation. That converted a correct
        # decision into a false escalation on policy-001, where the model
        # answered needs_human=false with a well-reasoned justification and
        # simply mislabelled the enum. The verdict is the field the prompt is
        # actually about; the enum is the field it is worst at.
        #
        # With the schema reordered so the verdict comes last, it is also the
        # field with the most context behind it.
        needs_human = bool(data.get("needs_human"))
        if not needs_human and rule != "NONE":
            # Disagreement is worth keeping in the trace rather than smoothing
            # over — it is the signal that the enum is unreliable.
            rule = f"NONE (model labelled {rule})"
        return GateDecision(
            needs_human=needs_human,
            rule=rule,
            reason=str(data.get("reason", ""))[:300],
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
    except Exception as exc:  # noqa: BLE001 - a gate failure is data, not a crash
        return GateDecision(
            needs_human=False,
            rule="NONE",
            reason=f"gate failed: {type(exc).__name__}: {exc}"[:300],
            latency_ms=int((time.perf_counter() - started) * 1000),
            failed=True,
        )
