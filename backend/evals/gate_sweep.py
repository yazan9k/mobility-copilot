"""Measure the escalation gate on its own, without running the agent loop.

A full eval run takes 25-45 minutes because it executes the whole agent. The
gate is one model call per case, so testing it in isolation is ~10x faster and
makes iteration on the escalation problem practical.

Two things this does that the full runner cannot:

  * **Captures the raw gate output**, including the failures. A gate failure
    defaults to no-escalation and disappears into the aggregate; here the
    exception and the raw text are kept so the failure mode can be read.

  * **Scores several decision rules from one pass.** `needs_human` and `rule`
    both come back in the same response, so "trust the boolean" and "escalate
    whenever a rule is named" can be compared without re-running anything.
    That turns a 40-minute A/B into a free one.

Run on the golden set; validate the winner once on held-out. Iterating against
held-out would destroy the only unbiased measurement in the project.

    python -m evals.gate_sweep --set golden --limit 40
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field

import yaml

from agent import escalation_gate, llm
from config import AGENT_MODEL, GATE_REPEAT_PENALTY, RUN_HISTORY_DIR
from evals.runner import CASE_SETS


@dataclass
class GateProbe:
    """One gate call, kept in full so failures stay readable."""
    case_id: str
    required: bool
    query: str
    raw: str = ""
    needs_human: bool | None = None
    rule: str = ""
    failed: bool = False
    error: str = ""
    latency_ms: int = 0


def probe(case: dict) -> GateProbe:
    """Call the gate once and record everything it returned."""
    question = escalation_gate._question_with_context(
        case["query"], case.get("history") or []
    )
    prompt = escalation_gate.active_prompt().format(question=question)
    result = GateProbe(
        case_id=case["id"],
        required=bool(case.get("requires_escalation")),
        query=case["query"],
    )
    started = time.perf_counter()
    try:
        result.raw = llm.generate_json(
            prompt=prompt,
            schema=escalation_gate.GATE_SCHEMA,
            model=AGENT_MODEL,
            temperature=0.0,
            # Must match production. An earlier version of this sweep called
            # generate_json with default settings and parsed with json.loads,
            # so it measured a gate that differed from the one that ships.
            repeat_penalty=GATE_REPEAT_PENALTY or None,
        )
        data = escalation_gate._parse_gate_json(result.raw)
        result.needs_human = bool(data.get("needs_human"))
        result.rule = str(data.get("rule", "NONE"))
    except Exception as exc:  # noqa: BLE001 - the failures are the point
        result.failed = True
        result.error = f"{type(exc).__name__}: {exc}"[:200]
    result.latency_ms = int((time.perf_counter() - started) * 1000)
    return result


# --- decision rules under test ---------------------------------------------
#
# Each takes a probe and returns whether to escalate. They read the SAME model
# output, so comparing them costs nothing beyond the single pass.

def rule_verdict(p: GateProbe) -> bool:
    """Current behaviour: trust `needs_human`, ignore the enum."""
    return bool(p.needs_human)


def rule_named(p: GateProbe) -> bool:
    """Escalate whenever a rule was named, whatever the boolean said.

    Motivated by the held-out failures: on 3 of 6 misses the model named the
    applicable rule (PERSON_SPECIFIC, NOT_AUTHORISED) and then answered
    needs_human=false. Under constrained decoding the boolean is emitted first,
    so it is decided before the justification exists.
    """
    return bool(p.rule and p.rule != "NONE")


def rule_either(p: GateProbe) -> bool:
    """Escalate if either signal fires."""
    return rule_verdict(p) or rule_named(p)


RULES = {
    "verdict (current)": rule_verdict,
    "rule-named": rule_named,
    "either": rule_either,
}


@dataclass
class Score:
    caught: int = 0
    missed: int = 0
    false_alarms: int = 0
    correct_quiet: int = 0
    missed_ids: list[str] = field(default_factory=list)
    false_ids: list[str] = field(default_factory=list)

    @property
    def recall(self) -> float:
        total = self.caught + self.missed
        return self.caught / total if total else 0.0

    @property
    def false_alarm_rate(self) -> float:
        total = self.false_alarms + self.correct_quiet
        return self.false_alarms / total if total else 0.0


def score(probes: list[GateProbe], decide) -> Score:
    s = Score()
    for p in probes:
        # A failed call escalates nothing — that is the production behaviour
        # and must be scored, not excluded.
        fired = False if p.failed else decide(p)
        if p.required:
            if fired:
                s.caught += 1
            else:
                s.missed += 1
                s.missed_ids.append(p.case_id)
        else:
            if fired:
                s.false_alarms += 1
                s.false_ids.append(p.case_id)
            else:
                s.correct_quiet += 1
    return s


def stratified(cases: list[dict], limit: int | None) -> list[dict]:
    """Keep every escalation case, sample the rest, so recall stays measurable."""
    required = [c for c in cases if c.get("requires_escalation")]
    other = [c for c in cases if not c.get("requires_escalation")]
    if limit is None or limit >= len(cases):
        return required + other
    room = max(limit - len(required), 0)
    step = max(len(other) // room, 1) if room else 1
    return required + other[::step][:room]


def main() -> int:
    ap = argparse.ArgumentParser(description="Test the escalation gate in isolation.")
    ap.add_argument("--set", dest="case_set", default="golden", choices=sorted(CASE_SETS))
    ap.add_argument("--limit", type=int, default=None, help="Cap cases (escalation cases always kept)")
    ap.add_argument("--out", help="Write raw probes here for later re-scoring")
    args = ap.parse_args()

    ok, detail = llm.health_check()
    if not ok:
        print(f"ERROR: {detail}", file=sys.stderr)
        return 1
    print(detail)

    cases = yaml.safe_load(CASE_SETS[args.case_set].read_text(encoding="utf-8"))["cases"]
    cases = stratified(cases, args.limit)
    n_req = sum(1 for c in cases if c.get("requires_escalation"))
    print(f"Probing {len(cases)} cases ({n_req} require escalation) from {args.case_set}\n")

    probes: list[GateProbe] = []
    for i, case in enumerate(cases, 1):
        p = probe(case)
        probes.append(p)
        mark = "FAIL" if p.failed else ("esc " if p.needs_human else "  - ")
        print(f"[{i:>2}/{len(cases)}] {p.case_id:<11} req={str(p.required):<5} "
              f"{mark} {p.rule[:20]:<20} {p.latency_ms:>7}ms", flush=True)

    failures = sum(1 for p in probes if p.failed)
    print(f"\n{'=' * 68}")
    print(f"Gate call failures: {failures}/{len(probes)} ({failures / len(probes):.0%})")
    print(f"{'=' * 68}")
    print(f"{'decision rule':<20}{'recall':>9}{'false alarm':>13}{'caught':>8}{'missed':>8}")
    for name, fn in RULES.items():
        s = score(probes, fn)
        print(f"{name:<20}{s.recall:>9.0%}{s.false_alarm_rate:>13.0%}"
              f"{s.caught:>8}{s.missed:>8}")

    print("\nPer-rule detail:")
    for name, fn in RULES.items():
        s = score(probes, fn)
        print(f"  {name}: missed={s.missed_ids or '-'}")
        print(f"  {' ' * len(name)}  false={s.false_ids or '-'}")

    if args.out:
        path = RUN_HISTORY_DIR / args.out if "/" not in args.out else args.out
        payload = {
            "case_set": args.case_set,
            "model": AGENT_MODEL,
            "probes": [p.__dict__ for p in probes],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
