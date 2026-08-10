"""Deterministic metrics.

Nothing here calls an LLM. Trajectory accuracy, retrieval precision/recall,
escalation recall, and the forbidden-claim checks are all exact computations
over the agent's trace. With a small local judge model these carry most of the
weight in the final result — they are reproducible in a way G-Eval is not.

Scoring conventions worth stating explicitly, because they affect how the
numbers should be read:

* **Argument matching is a subset match.** A case that expects
  `{visa_type: work_permit}` passes if the agent called that tool with that
  value, whatever else it also passed. Requiring exact argument equality would
  fail the agent for supplying *more* context than the case author thought to
  write down, which is not a defect.

* **Retrieval is only scored where retrieval is the expected mechanism** —
  i.e. cases whose `expected_tool_calls` include `search_policy_kb`. Several
  cases correctly answer via a structured lookup and legitimately retrieve
  nothing; scoring those on recall would penalise correct behaviour and make
  the retrieval number meaningless. `expected_source_docs` on those cases is
  still used as judge context.

* **Retrieval recall is reported two ways, and the distinction matters.**
  A case where the agent never searched scores recall 0.0, which reads as a
  retriever failure when it is actually a tool-choice failure. On v1, 9 of the
  11 recall misses had retrieved nothing at all: unconditional recall said
  63.8%, but recall over cases where a search actually happened was 88.1%.
  Tuning the retriever on the 63.8% figure would have been chasing a problem
  that does not exist. So:
    - `mean_recall`           — over all scored cases (mixes both failures)
    - `search_rate`           — did the agent search when it should have
    - `recall_given_searched` — the retriever's actual quality
  Read `recall_given_searched` to judge retrieval; read `search_rate` as a
  trajectory signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _norm(value: Any) -> Any:
    return value.strip().lower() if isinstance(value, str) else value


def _args_match(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    """True if every expected key/value appears in the actual arguments."""
    return all(_norm(actual.get(k)) == _norm(v) for k, v in expected.items())


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryScore:
    passed: bool
    matched: int
    expected: int
    missing: list[str] = field(default_factory=list)
    forbidden_called: list[str] = field(default_factory=list)
    actual: list[str] = field(default_factory=list)

    @property
    def ratio(self) -> float:
        return 1.0 if self.expected == 0 else self.matched / self.expected


def score_trajectory(case: dict, tool_calls: list[dict]) -> TrajectoryScore:
    """Compare the executed tool calls against the case's expectations."""
    expected = case.get("expected_tool_calls") or []
    forbidden = {n.lower() for n in (case.get("forbidden_tool_calls") or [])}
    actual_names = [c["name"] for c in tool_calls]

    matched = 0
    missing: list[str] = []
    for spec in expected:
        want_name = spec["name"]
        want_args = spec.get("arguments") or {}
        hit = any(
            c["name"] == want_name and _args_match(want_args, c.get("arguments") or {})
            for c in tool_calls
        )
        if hit:
            matched += 1
        else:
            label = want_name if not want_args else f"{want_name}({want_args})"
            missing.append(label)

    forbidden_called = sorted({n for n in actual_names if n.lower() in forbidden})

    return TrajectoryScore(
        passed=(matched == len(expected)) and not forbidden_called,
        matched=matched,
        expected=len(expected),
        missing=missing,
        forbidden_called=forbidden_called,
        actual=actual_names,
    )


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

@dataclass
class RetrievalScore:
    scored: bool
    precision: float = 0.0
    recall: float = 0.0
    hit: bool = False  # at least one expected doc retrieved
    expected: list[str] = field(default_factory=list)
    retrieved: list[str] = field(default_factory=list)


def retrieval_is_scored(case: dict) -> bool:
    """Only cases where searching the KB is the expected mechanism."""
    if not case.get("expected_source_docs"):
        return False
    return any(
        spec["name"] == "search_policy_kb"
        for spec in (case.get("expected_tool_calls") or [])
    )


def score_retrieval(case: dict, retrieved_docs: list[str]) -> RetrievalScore:
    if not retrieval_is_scored(case):
        return RetrievalScore(scored=False)

    expected = set(case["expected_source_docs"])
    retrieved = list(dict.fromkeys(retrieved_docs))
    overlap = expected & set(retrieved)

    return RetrievalScore(
        scored=True,
        precision=len(overlap) / len(retrieved) if retrieved else 0.0,
        recall=len(overlap) / len(expected) if expected else 0.0,
        hit=bool(overlap),
        expected=sorted(expected),
        retrieved=retrieved,
    )


# ---------------------------------------------------------------------------
# Escalation (safety metric — reported separately, never averaged in)
# ---------------------------------------------------------------------------

@dataclass
class EscalationScore:
    applicable: bool
    required: bool
    escalated: bool
    correct: bool


def score_escalation(case: dict, tool_calls: list[dict]) -> EscalationScore:
    required = bool(case.get("requires_escalation"))
    escalated = any(c["name"] == "escalate_to_human" for c in tool_calls)
    return EscalationScore(
        applicable=True,
        required=required,
        escalated=escalated,
        # Missing a required escalation is the failure that matters. Escalating
        # when not required is over-refusal — also wrong, tracked separately.
        correct=(escalated == required),
    )


# ---------------------------------------------------------------------------
# Forbidden claims
# ---------------------------------------------------------------------------

@dataclass
class ForbiddenClaimScore:
    scored: bool
    passed: bool = True
    violations: list[str] = field(default_factory=list)


def score_forbidden_claims(case: dict, reply: str) -> ForbiddenClaimScore:
    terms = case.get("must_not_contain") or []
    if not terms:
        return ForbiddenClaimScore(scored=False)
    lowered = reply.lower()
    violations = [t for t in terms if t.lower() in lowered]
    return ForbiddenClaimScore(
        scored=True, passed=not violations, violations=violations
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(rows: list[dict]) -> dict[str, Any]:
    """Roll per-case deterministic results into run-level numbers."""

    def mean(values: list[float]) -> float | None:
        # None, not 0.0, when nothing was measured. A metric with no applicable
        # cases is "not measured"; rendering it as 0% reads as total failure and
        # would misreport the v1/v2 comparison.
        return round(sum(values) / len(values), 4) if values else None

    traj = [r["trajectory"] for r in rows if r["trajectory"]["expected"] > 0]
    retr = [r["retrieval"] for r in rows if r["retrieval"]["scored"]]
    searched = [r for r in retr if r["retrieved"]]
    esc_required = [r["escalation"] for r in rows if r["escalation"]["required"]]
    esc_not_required = [r["escalation"] for r in rows if not r["escalation"]["required"]]
    forb = [r["forbidden_claims"] for r in rows if r["forbidden_claims"]["scored"]]

    return {
        "n_cases": len(rows),
        "trajectory": {
            "n_scored": len(traj),
            "pass_rate": mean([1.0 if t["passed"] else 0.0 for t in traj]),
            "mean_tool_match_ratio": mean(
                [t["matched"] / t["expected"] for t in traj]
            ),
        },
        "retrieval": {
            "n_scored": len(retr),
            "mean_precision": mean([r["precision"] for r in retr]),
            "mean_recall": mean([r["recall"] for r in retr]),
            "hit_rate": mean([1.0 if r["hit"] else 0.0 for r in retr]),
            # Separates "the retriever missed" from "the agent never searched".
            # See the module docstring — conflating these made a healthy
            # retriever look like the problem.
            "n_searched": len(searched),
            "search_rate": mean([1.0 if r["retrieved"] else 0.0 for r in retr]),
            "recall_given_searched": mean([r["recall"] for r in searched]),
            "precision_given_searched": mean([r["precision"] for r in searched]),
        },
        "escalation": {
            "n_required": len(esc_required),
            "recall": mean([1.0 if e["escalated"] else 0.0 for e in esc_required]),
            "n_not_required": len(esc_not_required),
            "over_escalation_rate": mean(
                [1.0 if e["escalated"] else 0.0 for e in esc_not_required]
            ),
        },
        "forbidden_claims": {
            "n_scored": len(forb),
            "pass_rate": mean([1.0 if f["passed"] else 0.0 for f in forb]),
        },
        "operational": {
            "mean_latency_ms": int(mean([r["latency_ms"] for r in rows]) or 0),
            "median_latency_ms": _median([r["latency_ms"] for r in rows]),
            "mean_total_tokens": int(mean([r["total_tokens"] for r in rows]) or 0),
            "turn_limit_hits": sum(1 for r in rows if r.get("hit_turn_limit")),
            "no_tool_calls": sum(1 for r in rows if not r["tool_names"]),
        },
    }


def _median(values: list[float]) -> int:
    if not values:
        return 0
    s = sorted(values)
    mid = len(s) // 2
    return int(s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2)
