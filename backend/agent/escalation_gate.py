"""A schema-constrained escalation decision, made outside the agent loop.

## Why this exists

Escalation recall sat at 20-40% across four prompt formulations (v2, v3, three
v4 arms, v5) and two model architectures (qwen2.5:7b dense, Ling 3.0 Tiny MoE).
Nothing in prompt-space moved it.

The qwen evidence is solid and is what motivates this module:

  qwen2.5:7b   10 of 13 misses state in the reply that the question must go to
               an adviser, then never call the tool. esc-004 even asks "Would
               you like to proceed with escalating this request?" — permission
               seeking that every prompt since v2 explicitly forbids. The
               judgement is right; the action does not happen.

That is an execution failure: a single prompt asking one model to answer the
question, choose among six tools, and remember a safety rule fails at the part
it is weakest on. Moving the decision out of free-form generation addresses it.

## A retracted claim

This docstring previously also stated that Ling 3.0 had the opposite failure —
a *recognition* failure, "only 1 of 7 held-out misses mentions escalation at
all; it does not recognise the cases, it answers them." That was wrong, and it
was wrong because of a bug in agent/llm.py rather than a mistake in reading the
traces: 6 of those 7 replies were the empty string. llama-server had filed the
model's output as reasoning and the provider seam returned "". The claim
measured a parsing fault and reported it as a property of the model.

On the one held-out miss where text actually existed, the reply *did* raise
going to an adviser — pointing, as far as one case can point, at qwen's
execution failure rather than a distinct recognition failure.

Ling's real escalation behaviour is being remeasured. Until that lands, nothing
in this module is justified by Ling evidence.

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
import re
from dataclasses import dataclass

from agent import llm
from config import (
    AGENT_MODEL,
    GATE_DECISION_RULE,
    GATE_PROMPT_VERSION,
    GATE_REPEAT_PENALTY,
)

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


# v2 of the criteria. Targets over-escalation, which the v1 wording drove to 12%
# on golden and 30% on held-out — 7 of those 9 false alarms were rule 1 firing.
#
# Two changes, each aimed at a failure mode read off the traces:
#
#   1. "personal financial position" is gone. It was broad enough to justify
#      escalating almost any first-person question, and did: "how much furniture
#      can I ship" was escalated as involving the employee's personal financial
#      position. Rule 1 now names the specific facts policy cannot supply, and
#      states the test as whether a MISSING fact changes the answer.
#
#   2. An explicit dependent carve-out. The model escalated partner language
#      lessons and spousal career support as "another person's package" — but
#      those are benefits the company provides to a dependent, which are the
#      employee's own entitlement and are written down. On multi-002 it claimed
#      the question "asks whether the husband and kids may WORK", which it does
#      not. Four of nine false alarms were this.
#
# What is deliberately unchanged: rules 2-4, and every phrase that earned the
# 85%/80% recall. This edit is meant to cost recall as little as possible, and
# the sweep reports recall alongside false alarms so the trade is visible.
GATE_PROMPT_V2 = """You decide whether an employee's question must be handled by a human Global Mobility adviser, or can be answered from company policy.

Answer "yes" if ANY of these is true:

1. PERSON_SPECIFIC — answering requires a fact about this person that policy cannot supply AND that would change the answer: their own tax liability or tax residence, their medical circumstances, their pension or social security record, or a dependent's nationality or immigration status. This includes whether a spouse, partner, or dependent may WORK in the destination, because that turns on the dependent's own nationality rather than the employee's.

2. UNDECIDED — answering depends on an approval, exception, or discretionary decision nobody has made yet: a request to exceed an allowance, extend a covered period, end an assignment early, or vary an entitlement.

3. NOT_AUTHORISED — the question asks about a DIFFERENT EMPLOYEE's package, pay, immigration status, or benefits. Being their manager or colleague does not by itself authorise it. This rule is about other employees; it does not cover the asker's own family.

4. HIGH_CONSEQUENCE — being wrong causes serious or irreversible harm: a refused, expired, or at-risk visa or permit; a required document that cannot be obtained; a criminal record disclosure; or a significant back-dated tax liability. An assignment extension crossing six months counts, because tax residence can change retroactively.

Answer "no" for everything else. Standard entitlements, allowance amounts, timelines, document lists, what a policy covers or excludes, the difference between assignment types, and a dependent's right to RESIDE are all normal questions with policy answers.

Two things that are NOT reasons to answer "yes":

* **The question mentions the employee's family.** A benefit the company provides TO a partner, spouse, or child — language lessons, spousal career support, schooling, dependent relocation costs, dependent visa sponsorship — is the EMPLOYEE'S entitlement and is written down in policy. Only the dependent's legal right to WORK is rule 1.

* **The question says "I" or "my".** "How much can I ship", "does this apply to me", "what is my stipend", "does my allowance stop" are all asking what policy provides. Asking what policy says about your own situation is a policy question, not a personal one.

The test is whether policy settles the question, not whether the topic is personal.

A question can contain both: if any part meets a rule above, answer "yes" — the answerable part is handled separately.

Employee question:
{question}

Reply with JSON only."""

PROMPTS = {"v1": GATE_PROMPT, "v2": GATE_PROMPT_V2}


def active_prompt() -> str:
    """The criteria text in force, selected by GATE_PROMPT_VERSION."""
    return PROMPTS.get(GATE_PROMPT_VERSION, GATE_PROMPT_V2)

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


_BOOL_RE = re.compile(r'"needs_human"\s*:\s*(true|false)', re.I)
_RULE_RE = re.compile(
    r'"rule"\s*:\s*"(PERSON_SPECIFIC|UNDECIDED|NOT_AUTHORISED|HIGH_CONSEQUENCE|NONE)"'
)
_REASON_RE = re.compile(r'"reason"\s*:\s*"(.*?)(?:"|$)', re.S)


def _parse_gate_json(raw: str) -> dict:
    """Parse the gate's reply, salvaging the verdict from a truncated response.

    Gate calls fail at roughly a 20% rate, and the dominant mode is generation
    running long and being cut mid-string, leaving invalid JSON. Until now that
    raised, and a raised gate defaults to no escalation — so a truncated
    response and a genuine "no" were indistinguishable.

    They need not be. The schema emits `needs_human` and `rule` *before*
    `reason`, so a response cut inside `reason` still contains both decision
    fields in full:

        {"needs_human": true, "rule": "PERSON_SPECIFIC", "reason": "The user is

    Strict parsing is tried first and is authoritative. The regex path only
    runs when that fails, and only accepts the enum's real members, so it
    cannot invent a verdict that was never generated. If the boolean itself did
    not survive, this raises and the caller records a genuine failure.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    verdict = _BOOL_RE.search(raw)
    if not verdict:
        # Nothing decidable was produced. A real failure, not a "no".
        raise ValueError("gate response contains no needs_human verdict")

    rule = _RULE_RE.search(raw)
    reason = _REASON_RE.search(raw)
    return {
        "needs_human": verdict.group(1).lower() == "true",
        "rule": rule.group(1) if rule else "NONE",
        "reason": (reason.group(1) if reason else "") + " [recovered from truncated JSON]",
    }


def _apply_decision_rule(verdict: bool, rule: str) -> bool:
    """Combine the gate's two signals according to GATE_DECISION_RULE.

    See config.GATE_DECISION_RULE. Kept as one function so the three variants
    are visibly the same code path and the comparison is honest — the sweep
    scores exactly what production runs.
    """
    named = bool(rule) and rule != "NONE"
    if GATE_DECISION_RULE == "named":
        return named
    if GATE_DECISION_RULE == "either":
        return verdict or named
    return verdict


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

    prompt = active_prompt().format(question=_question_with_context(message, history))
    started = time.perf_counter()
    try:
        raw = llm.generate_json(
            prompt=prompt,
            schema=GATE_SCHEMA,
            model=model or AGENT_MODEL,
            temperature=0.0,
            repeat_penalty=GATE_REPEAT_PENALTY or None,
        )
        data = _parse_gate_json(raw)
        rule = str(data.get("rule", "NONE"))
        # `needs_human` is authoritative and the rule is a diagnostic label.
        #
        # The first version derived the boolean from the rule instead, on the
        # theory that a named rule implies escalation. That converted a correct
        # decision into a false escalation on policy-001, where the model
        # answered needs_human=false with a well-reasoned justification and
        # simply mislabelled the enum. The verdict is the field the prompt is
        # actually about; the enum is the field it is worst at.
        needs_human = _apply_decision_rule(bool(data.get("needs_human")), rule)
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
