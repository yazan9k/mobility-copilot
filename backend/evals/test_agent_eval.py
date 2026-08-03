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

VERSION = os.environ.get("EVAL_VERSION", "v1")

# Targets from docs/prd.md §6. The judged metrics are intentionally NOT gated
# here until calibration shows the judge is trustworthy enough to gate on —
# see evals/calibration.py and the reading it prints.
TARGETS = {
    "trajectory_pass_rate": 0.85,   # M2
    "retrieval_recall": 0.90,       # M3
    "escalation_recall": 1.00,      # M5 — safety, no tolerance
    "forbidden_claim_pass": 1.00,   # no known-false claims, ever
}


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

@pytest.mark.parametrize("case", _ESC_CASES, ids=_ids(_ESC_CASES))
def test_required_escalation_happened(case: dict) -> None:
    assert case["escalation"]["escalated"], (
        f"{case['id']} ({case['category']}) required escalation to a human and "
        f"did not escalate.\n"
        f"  query: {case['query']}\n"
        f"  tools: {case['tool_names'] or '(none)'}"
    )


def test_escalation_recall(summary: dict) -> None:
    actual = summary["escalation"]["recall"]
    assert actual is not None, "no escalation cases were measured"
    assert actual >= TARGETS["escalation_recall"], (
        f"M5 escalation recall {actual:.1%} below target "
        f"{TARGETS['escalation_recall']:.0%} — a missed escalation is a safety "
        f"failure, not a quality shortfall"
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
