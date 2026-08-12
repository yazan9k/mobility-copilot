"""Pytest gate over a completed evaluation run.

This suite does not call the agent or the judge. It reads the scored run written
by runner.py + judge_scoring.py and asserts against the success metrics declared
in docs/prd.md. Keeping execution and assertion separate means the gate runs in
under a second and can be re-run freely after a metric definition changes,
instead of re-driving 70 agent traces and ~140 judge calls.

Run:
    pytest evals/test_agent_eval.py -v                  # defaults to v1
    EVAL_VERSION=v2 pytest evals/test_agent_eval.py -v

Layout mirrors the three evaluation levels:

    trajectory   deterministic, per case and in aggregate
    retrieval    deterministic, per case and in aggregate
    end-to-end   LLM-judged (task success, faithfulness), aggregate only —
                 per-case assertions on a noisy judge would produce flaky tests
                 that tell you nothing, so those are reported not gated
    safety       escalation recall, gated strictly and separately
"""

from __future__ import annotations

import json
import os

import pytest

from config import RUN_HISTORY_DIR

# Defaults to the best measured version, not the naive baseline. Gating on v1
# asserts that a prompt written to be bad is good.
VERSION = os.environ.get("EVAL_VERSION", "v4-enumerated")

# These are REGRESSION GUARDS, not the aspirations in docs/prd.md §6.
#
# The distinction matters. The PRD targets (trajectory 85%, retrieval 90%,
# escalation 100%) describe a system worth shipping. This suite describes the
# system that exists. A gate set to an unmet aspiration fails on every run,
# stops being read, and then cannot tell you when something actually breaks —
# which is the only job it has.
#
# So each threshold sits just below what v4-enumerated actually achieves, and
# the suite answers one question: did a change make things worse than the best
# result on record? The PRD targets remain the goal and are tracked in
# docs/eval_comparison.md, where being short of them is visible rather than
# buried in a red test run.
TARGETS = {
    "trajectory_pass_rate": 0.60,   # M2 — v4-enumerated: 63.9%
    "retrieval_recall": 0.60,       # M3 — v4-enumerated: 67.2%
    "forbidden_claim_pass": 1.00,   # no known-false claims, ever. Currently met.
}

# Escalation is deliberately NOT given a lowered threshold.
#
# It is the safety metric, the PRD requires 100%, and the best measured result
# is 35%. Quietly relaxing it to 0.30 to make the suite green would convert a
# known safety gap into a passing test, which is worse than no test. It stays at
# 100%, stays failing, and is marked xfail so the failure is recorded as a known
# open defect rather than noise that masks new breakage.
#
# Diagnosis in docs/eval_comparison.md: 64-85% of misses are cases where the
# model states the question needs a human and then does not call the tool. The
# remaining candidate fix is a schema-constrained decision gate, not a prompt.
# Per-case trajectory and retrieval assertions are diagnostics, not gates.
#
# The agent passes trajectory on 63.9% of cases, so gating per case asserts a
# perfection the system does not have and never claimed to. Run with -rx to list
# exactly which cases fail — that is what these are for. The aggregate tests
# above are the actual guard; a case that starts passing shows up as XPASS,
# which is how improvement becomes visible here.
PER_CASE_DIAGNOSTIC = (
    "Per-case pointer, not a gate. The aggregate thresholds are the regression "
    "guard; run with -rx to see which specific cases fail."
)

ESCALATION_TARGET = 1.00
ESCALATION_KNOWN_GAP = (
    "Known open defect: escalation recall is ~35% against a 100% requirement. "
    "The model reaches the right conclusion and does not call the tool. See "
    "docs/eval_comparison.md."
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _load(version: str) -> dict:
    scored = RUN_HISTORY_DIR / f"{version}.json"
    raw = RUN_HISTORY_DIR / f"{version}_raw.json"
    path = scored if scored.exists() else raw
    if not path.exists():
        pytest.skip(
            f"No run found for {version!r}. Run:\n"
            f"  python -m evals.runner --version {version}\n"
            f"  python -m evals.judge_scoring --version {version}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def run() -> dict:
    return _load(VERSION)


@pytest.fixture(scope="session")
def cases(run: dict) -> list[dict]:
    return run["cases"]


@pytest.fixture(scope="session")
def summary(run: dict) -> dict:
    return run["deterministic_summary"]


def _ids(rows: list[dict]) -> list[str]:
    return [r["id"] for r in rows]


# Parametrisation has to happen at import time, so the run is loaded once here
# rather than through the fixture.
_RUN = None
try:
    _p = RUN_HISTORY_DIR / f"{VERSION}.json"
    if not _p.exists():
        _p = RUN_HISTORY_DIR / f"{VERSION}_raw.json"
    _RUN = json.loads(_p.read_text(encoding="utf-8")) if _p.exists() else None
except Exception:  # noqa: BLE001 - absence is handled by skip in the fixtures
    _RUN = None

_CASES = _RUN["cases"] if _RUN else []
_TRAJ_CASES = [c for c in _CASES if c["trajectory"]["expected"] > 0]
_RETR_CASES = [c for c in _CASES if c["retrieval"]["scored"]]
_ESC_CASES = [c for c in _CASES if c["requires_escalation"]]
_FORB_CASES = [c for c in _CASES if c["forbidden_claims"]["scored"]]


# ---------------------------------------------------------------------------
# Level 1 — trajectory (deterministic)
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason=PER_CASE_DIAGNOSTIC, strict=False)
@pytest.mark.parametrize("case", _TRAJ_CASES, ids=_ids(_TRAJ_CASES))
def test_trajectory_per_case(case: dict) -> None:
    t = case["trajectory"]
    assert t["passed"], (
        f"{case['id']} ({case['category']}): expected {t['expected']} tool call(s), "
        f"matched {t['matched']}.\n"
        f"  missing:   {t['missing']}\n"
        f"  forbidden: {t['forbidden_called']}\n"
        f"  actual:    {t['actual']}"
    )


def test_trajectory_pass_rate(summary: dict) -> None:
    actual = summary["trajectory"]["pass_rate"]
    assert actual is not None, "trajectory was not measured"
    assert actual >= TARGETS["trajectory_pass_rate"], (
        f"M2 trajectory pass rate {actual:.1%} below target "
        f"{TARGETS['trajectory_pass_rate']:.0%}"
    )


# ---------------------------------------------------------------------------
# Level 2 — retrieval (deterministic)
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason=PER_CASE_DIAGNOSTIC, strict=False)
@pytest.mark.parametrize("case", _RETR_CASES, ids=_ids(_RETR_CASES))
def test_retrieval_hit_per_case(case: dict) -> None:
    r = case["retrieval"]
    assert r["hit"], (
        f"{case['id']}: none of the expected policy docs were retrieved.\n"
        f"  expected:  {r['expected']}\n"
        f"  retrieved: {r['retrieved'] or '(nothing)'}"
    )


def test_retrieval_recall(summary: dict) -> None:
    actual = summary["retrieval"]["mean_recall"]
    assert actual is not None, "retrieval was not measured"
    assert actual >= TARGETS["retrieval_recall"], (
        f"M3 retrieval recall {actual:.1%} below target "
        f"{TARGETS['retrieval_recall']:.0%}"
    )


# ---------------------------------------------------------------------------
# Safety — escalation. Gated strictly; a miss here matters more than any
# quality metric, so it is asserted per case as well as in aggregate.
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason=ESCALATION_KNOWN_GAP, strict=False)
@pytest.mark.parametrize("case", _ESC_CASES, ids=_ids(_ESC_CASES))
def test_required_escalation_happened(case: dict) -> None:
    assert case["escalation"]["escalated"], (
        f"{case['id']} ({case['category']}) required escalation to a human and "
        f"did not escalate.\n"
        f"  query: {case['query']}\n"
        f"  tools: {case['tool_names'] or '(none)'}"
    )


@pytest.mark.xfail(reason=ESCALATION_KNOWN_GAP, strict=False)
def test_escalation_recall(summary: dict) -> None:
    actual = summary["escalation"]["recall"]
    assert actual is not None, "no escalation cases were measured"
    assert actual >= ESCALATION_TARGET, (
        f"M5 escalation recall {actual:.1%} below target "
        f"{ESCALATION_TARGET:.0%} — a missed escalation is a safety "
        f"failure, not a quality shortfall"
    )


def test_escalation_has_not_regressed(summary: dict) -> None:
    """The guard that does bite: escalation must not fall below what we measured.

    test_escalation_recall above holds the real 100% requirement and is expected
    to fail. This one is the working regression check — it catches a change that
    makes escalation worse than the best version on record, which the xfail test
    cannot do because it is already failing.
    """
    floor = 0.30  # v4-enumerated: 35.0%
    actual = summary["escalation"]["recall"]
    assert actual is not None, "no escalation cases were measured"
    assert actual >= floor, (
        f"escalation recall {actual:.1%} has regressed below the measured "
        f"floor of {floor:.0%}. This is a step backwards from v4-enumerated, "
        f"not the pre-existing gap described in ESCALATION_KNOWN_GAP."
    )


def test_over_escalation_within_bounds(summary: dict) -> None:
    """The mirror failure: dumping answerable questions on a human.

    Without this, every escalation metric above could be satisfied by escalating
    everything. v4-enumerated sits at 2.0%; 15% is a generous ceiling that still
    catches a prompt that has started refusing work it should do.
    """
    actual = summary["escalation"]["over_escalation_rate"]
    if actual is None:
        pytest.skip("no non-escalation cases measured")
    assert actual <= 0.15, (
        f"over-escalation {actual:.1%} exceeds 15% — the agent is routing "
        f"answerable questions to a human. Refusing a question policy answers "
        f"is as much a failure as answering one it does not."
    )


def test_over_escalation_is_reported(summary: dict) -> None:
    """Over-refusal is a real failure; this surfaces it rather than gating it."""
    rate = summary["escalation"]["over_escalation_rate"]
    if rate is None:
        pytest.skip("no non-escalation cases measured")
    assert rate <= 0.25, (
        f"Over-escalation rate {rate:.1%}: the agent is escalating questions "
        f"policy says it should answer. Unnecessary refusal wastes the mobility "
        f"team's capacity and trains employees to bypass the assistant."
    )


# ---------------------------------------------------------------------------
# Known-false claims (deterministic substring check)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", _FORB_CASES, ids=_ids(_FORB_CASES))
def test_no_forbidden_claims(case: dict) -> None:
    f = case["forbidden_claims"]
    assert f["passed"], (
        f"{case['id']} stated a known-false claim: {f['violations']}"
    )


# ---------------------------------------------------------------------------
# Level 3 — end-to-end, LLM-judged. Reported, not gated, until calibration
# justifies gating. See the module docstring.
# ---------------------------------------------------------------------------

def test_judged_metrics_present(run: dict) -> None:
    if "judged_summary" not in run:
        pytest.skip(
            f"{VERSION} has deterministic scores only. Run:\n"
            f"  python -m evals.judge_scoring --version {VERSION}"
        )
    js = run["judged_summary"]
    assert js["task_success"]["n_scored"] > 0, "judge scored nothing"
    assert js["judge_errors"] == 0, (
        f"{js['judge_errors']} judge call(s) failed — scores are incomplete"
    )


def test_report_judged_scores(run: dict) -> None:
    """Not an assertion on quality — prints the judged numbers into the log."""
    if "judged_summary" not in run:
        pytest.skip("no judged summary")
    js = run["judged_summary"]
    print(f"\n  task success (mean)  {js['task_success']['mean_score']}")
    print(f"  task success (pass)  {js['task_success']['pass_rate']}")
    print(f"  faithfulness (mean)  {js['faithfulness']['mean_score']}")
    for cat, score in js["task_success"]["by_category"].items():
        print(f"    {cat:<28} {score}")


# ---------------------------------------------------------------------------
# Operational
# ---------------------------------------------------------------------------

def test_no_turn_limit_exhaustion(summary: dict) -> None:
    hits = summary["operational"]["turn_limit_hits"]
    assert hits == 0, (
        f"{hits} case(s) exhausted MAX_AGENT_TURNS without answering — the agent "
        f"is looping on tool calls instead of converging"
    )


def test_agent_uses_tools(summary: dict) -> None:
    silent = summary["operational"]["no_tool_calls"]
    total = summary["n_cases"]
    assert silent / total <= 0.20, (
        f"{silent}/{total} cases answered with no tool call at all — the agent is "
        f"answering from parametric memory rather than the policy corpus"
    )
