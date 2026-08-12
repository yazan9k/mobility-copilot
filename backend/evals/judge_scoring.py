"""Add LLM-judged metrics to a completed run.

Reads  evals/run_history/<version>_raw.json   (written by runner.py)
Writes evals/run_history/<version>.json       (raw + judged, the canonical run)

Two metrics, both DeepEval, both driven by the local judge model:

  task_success  — custom G-Eval rubric. Per-case criteria from the golden set
                  are injected as additional context, so the judge scores
                  against what the case actually required rather than a vague
                  notion of helpfulness.
  faithfulness  — DeepEval's built-in metric, over the tool outputs the agent
                  actually received. Catches claims the agent invented.

async_mode is off throughout. There is one local model on one machine;
concurrent requests to a 14B would contend for the same GPU and produce worse
wall-clock time, not better.

Run:  python -m evals.judge_scoring --version v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from typing import Any

from deepeval.metrics import FaithfulnessMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from config import RUN_HISTORY_DIR
from evals.ollama_judge import OllamaJudge, health_check

# RUBRIC v2 — revised on evidence, before any baseline score was recorded.
#
# The change was NOT made on the hunch that prompted the calibration exercise.
# That hunch — that the judge over-penalises extra detail — was tested against 20
# human-labelled cases and disproved: cal-011, dense with additional detail,
# scored 1.00.
#
# Calibration exposed a different, real defect instead. The old third step read
# "penalise ... that contradicts the criteria". The judge detached the qualifier
# and applied it as a blanket prohibition on stating figures at all, flunking two
# plainly-correct answers. In its own words:
#
#   cal-003 (0.20) "align perfectly with the expected output criteria. However,
#                   it is penalized due to providing a concrete figure."
#   cal-005 (0.00) "provides a specific figure of 12,000 EUR ... which contradicts
#                   the criteria to avoid stating such figures."
#
# Steps 3 and 4 are rewritten so the qualifier cannot be detached, and a
# read-carefully step is added for cal-002, where the judge claimed an answer
# implied the opposite of what it explicitly said.
#
# This edits the measuring instrument, not the agent, and no score had been taken
# with the old wording. evals/calibration.py re-measures agreement afterwards.
TASK_SUCCESS_STEPS = [
    "Read the employee's question and the listed success criteria for this case. "
    "The criteria are the complete definition of success — score against them and "
    "nothing else.",
    "Check whether the actual output satisfies each criterion. A criterion phrased "
    "as a prohibition (for example 'does NOT state...') is satisfied only if the "
    "output avoids doing that thing. The score should be roughly the proportion of "
    "criteria satisfied.",
    "Stating a specific figure, amount, duration, permit name, or entitlement is "
    "NEVER a fault in itself. When a criterion asks for a figure, giving that figure "
    "is exactly what satisfies it. Compare the value in the answer against the value "
    "in the criterion: if they match, that criterion is MET.",
    "Penalise a stated figure, duration, permit name, or entitlement ONLY when it "
    "conflicts with what a criterion says. A wrong number is a failure; a right "
    "number is a pass; a number the criteria did not ask about is neither.",
    "Do NOT penalise the answer for including accurate, relevant information beyond "
    "the listed criteria. Criteria are a minimum bar, not an exhaustive script.",
    "Read the answer carefully before concluding it contradicts a criterion. If the "
    "answer explicitly states the thing a criterion asks for, that criterion is met "
    "even if the surrounding wording is unusual.",
    "Penalise an answer that accepts a false premise in the question instead of "
    "correcting it.",
    "Penalise an answer that refuses or defers a question the criteria indicate "
    "should have been answered. Unnecessary refusal is a failure, not a safe default.",
    "Do not reward length, formatting, or politeness. Score only whether the "
    "employee's actual need was met.",
]


# CLARITY — a separate dimension, added after calibration.
#
# cal-015 was the single surviving judge/human disagreement at rubric v2: an
# answer stating both required facts in two clipped lines. The judge passed it
# (both criteria were met) and the human failed it (an employee could not act on
# it). Neither was wrong — the criteria simply say nothing about being
# understandable, so there was nothing to score.
#
# It is kept OUT of task success on purpose. Folding a readability bar into the
# correctness metric would mean a v2 that merely writes at greater length scores
# as though it got more answers right. Two numbers keep correctness and
# comprehensibility separable in the v1/v2 comparison.
CLARITY_STEPS = [
    "You are judging whether a relocating employee with no knowledge of internal "
    "mobility jargon could read this reply and act on it without asking a "
    "follow-up question. Judge only that. Do not judge factual accuracy — another "
    "metric covers it, and a confidently wrong answer can still be perfectly clear.",
    "Penalise a bare figure or entitlement given without saying what it covers, when "
    "it is paid, or what it excludes.",
    "Penalise an answer that lists every tier, band, or category and leaves the reader "
    "to work out which applies to them, when the question made their situation clear.",
    "Penalise unexplained internal vocabulary — tier, band, assignment type, "
    "per-diem, equalization — used as though the reader already knows it.",
    "Penalise any reference to internal document filenames, source markers, or tool "
    "names. The reader has no access to those.",
    "Penalise an answer so terse it reads as a lookup result rather than a reply to a "
    "person, even when every fact in it is correct.",
    "Penalise not saying what happens next where the employee has to do something.",
    "Do NOT reward length for its own sake. A long answer that buries the point is "
    "worse than a short one that states it. Reward being complete and plain, not big.",
    "Do NOT penalise a correct refusal or handoff, provided it explains why and says "
    "what happens next.",
]


def build_metrics(
    judge: OllamaJudge, threshold: float
) -> tuple[GEval, FaithfulnessMetric, GEval]:
    task_success = GEval(
        name="TaskSuccess",
        evaluation_steps=TASK_SUCCESS_STEPS,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model=judge,
        threshold=threshold,
        async_mode=False,
    )
    faithfulness = FaithfulnessMetric(
        model=judge,
        threshold=threshold,
        async_mode=False,
        include_reason=True,
    )
    clarity = GEval(
        name="Clarity",
        evaluation_steps=CLARITY_STEPS,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        model=judge,
        threshold=threshold,
        async_mode=False,
    )
    return task_success, faithfulness, clarity


def score_case(
    row: dict[str, Any],
    task_success: GEval,
    faithfulness: FaithfulnessMetric,
    clarity: GEval,
) -> dict[str, Any]:
    criteria = row.get("expected_answer_criteria") or []
    expected_output = (
        "The response must satisfy all of the following:\n"
        + "\n".join(f"- {c}" for c in criteria)
    ) if criteria else "A correct, policy-grounded answer to the question."

    result: dict[str, Any] = {}

    tc = LLMTestCase(
        input=row["query"],
        actual_output=row["reply"] or "(empty response)",
        expected_output=expected_output,
    )
    try:
        task_success.measure(tc)
        result["task_success"] = {
            "score": task_success.score,
            "passed": bool(task_success.score is not None
                           and task_success.score >= task_success.threshold),
            "reason": task_success.reason,
        }
    except Exception as exc:  # noqa: BLE001
        result["task_success"] = {"score": None, "passed": False, "error": str(exc)}

    # Faithfulness needs something to be faithful to. A reply produced with no
    # tool calls has no retrieved grounding, so the metric does not apply — and
    # scoring it as 0 would conflate "ungrounded" with "unfaithful".
    context = [c for c in (row.get("grounding_context") or []) if c.strip()]
    if context:
        ftc = LLMTestCase(
            input=row["query"],
            actual_output=row["reply"] or "(empty response)",
            retrieval_context=context,
        )
        try:
            faithfulness.measure(ftc)
            result["faithfulness"] = {
                "score": faithfulness.score,
                "passed": bool(faithfulness.score is not None
                               and faithfulness.score >= faithfulness.threshold),
                "reason": faithfulness.reason,
                "scored": True,
            }
        except Exception as exc:  # noqa: BLE001
            result["faithfulness"] = {"score": None, "passed": False,
                                      "scored": True, "error": str(exc)}
    else:
        result["faithfulness"] = {"score": None, "scored": False,
                                  "reason": "No tool output to ground against."}

    ctc = LLMTestCase(
        input=row["query"],
        actual_output=row["reply"] or "(empty response)",
    )
    try:
        clarity.measure(ctc)
        result["clarity"] = {
            "score": clarity.score,
            "passed": bool(clarity.score is not None
                           and clarity.score >= clarity.threshold),
            "reason": clarity.reason,
        }
    except Exception as exc:  # noqa: BLE001
        result["clarity"] = {"score": None, "passed": False, "error": str(exc)}

    return result


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(vals: list[float]) -> float | None:
        return round(sum(vals) / len(vals), 4) if vals else None

    ts = [r["task_success"] for r in rows if r["task_success"].get("score") is not None]
    fa = [
        r["faithfulness"] for r in rows
        if r["faithfulness"].get("scored") and r["faithfulness"].get("score") is not None
    ]
    errors = sum(1 for r in rows if "error" in r["task_success"])

    by_category: dict[str, list[float]] = {}
    for r in rows:
        if r["task_success"].get("score") is not None:
            by_category.setdefault(r["category"], []).append(r["task_success"]["score"])

    cl = [r["clarity"] for r in rows if r.get("clarity", {}).get("score") is not None]

    return {
        "clarity": {
            "n_scored": len(cl),
            "mean_score": mean([c["score"] for c in cl]),
            "pass_rate": mean([1.0 if c["passed"] else 0.0 for c in cl]),
        },
        "task_success": {
            "n_scored": len(ts),
            "mean_score": mean([t["score"] for t in ts]),
            "pass_rate": mean([1.0 if t["passed"] else 0.0 for t in ts]),
            "by_category": {k: mean(v) for k, v in sorted(by_category.items())},
        },
        "faithfulness": {
            "n_scored": len(fa),
            "mean_score": mean([f["score"] for f in fa]),
            "pass_rate": mean([1.0 if f["passed"] else 0.0 for f in fa]),
        },
        "judge_errors": errors,
    }


def stratified_sample(rows: list[dict], n: int) -> list[dict]:
    """N cases spread evenly across categories, deterministically.

    Judging every case on a local 14b model costs hours, so a subset is the
    practical option. It has to be stratified: the golden set is ordered by
    category, so taking the first N would score only visa and policy questions
    while the summary line still says "task success = X" as though it covered
    escalation and out-of-scope cases too.

    Selection is by position within each category rather than random, so the
    same cases are scored for every version and the comparison stays paired.
    """
    from collections import defaultdict

    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)

    per_category = max(1, n // len(by_category))
    picked = [r for cases in by_category.values() for r in cases[:per_category]]

    # Distribute any remainder over the largest categories so the total lands
    # on n rather than a rounded-down approximation of it.
    if len(picked) < n:
        chosen = {r["id"] for r in picked}
        for cases in sorted(by_category.values(), key=len, reverse=True):
            for row in cases:
                if len(picked) >= n:
                    break
                if row["id"] not in chosen:
                    picked.append(row)
                    chosen.add(row["id"])
            if len(picked) >= n:
                break

    order = {r["id"]: i for i, r in enumerate(rows)}
    return sorted(picked, key=lambda r: order[r["id"]])


def main() -> int:
    parser = argparse.ArgumentParser(description="Add judged metrics to a run.")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--limit", type=int, help="Score only the first N cases")
    parser.add_argument(
        "--sample", type=int,
        help="Score a stratified sample of N cases, spread evenly across "
             "categories. Prefer this to --limit: the golden set is ordered by "
             "category, so --limit 25 scores only visa and policy questions and "
             "reports the result as if it covered the set.",
    )
    args = parser.parse_args()

    ok, detail = health_check()
    if not ok:
        print(f"ERROR: {detail}", file=sys.stderr)
        return 1
    print(detail)

    raw_path = RUN_HISTORY_DIR / f"{args.version}_raw.json"
    if not raw_path.exists():
        print(f"ERROR: {raw_path} not found. Run evals.runner first.", file=sys.stderr)
        return 1

    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    rows = payload["cases"]
    if args.limit:
        rows = rows[: args.limit]
    if args.sample:
        rows = stratified_sample(rows, args.sample)
        print(f"Stratified sample: {len(rows)} of {len(payload['cases'])} cases")

    judge = OllamaJudge()
    task_success, faithfulness, clarity = build_metrics(judge, args.threshold)

    print(f"Scoring {len(rows)} cases with {judge.get_model_name()} "
          f"(threshold {args.threshold})\n")
    started = time.perf_counter()

    for i, row in enumerate(rows, 1):
        scores = score_case(row, task_success, faithfulness, clarity)
        row.update(scores)

        ts = scores["task_success"]
        fa = scores["faithfulness"]
        ts_str = "err" if ts.get("score") is None else f"{ts['score']:.2f}"
        fa_str = "n/a" if fa.get("score") is None else f"{fa['score']:.2f}"
        cl = scores["clarity"]
        cl_str = "err" if cl.get("score") is None else f"{cl['score']:.2f}"
        mark = "" if ts.get("passed") else "  <-- FAIL"
        print(f"[{i:>2}/{len(rows)}] {row['id']:<10} task={ts_str} "
              f"clarity={cl_str} faith={fa_str}{mark}", flush=True)

    elapsed = time.perf_counter() - started
    payload["judged_summary"] = summarise(rows)
    # Fingerprint of the exact rubric text used. compare.py refuses to present
    # judged metrics across two runs whose fingerprints differ — otherwise a
    # v1/v2 delta could be reporting a rubric edit as an agent improvement.
    rubric_fingerprint = hashlib.sha256(
        ("\n".join(TASK_SUCCESS_STEPS) + "\n--\n" + "\n".join(CLARITY_STEPS))
        .encode("utf-8")
    ).hexdigest()[:12]

    payload["judge"] = {
        "model": judge.get_model_name(),
        "threshold": args.threshold,
        "rubric_fingerprint": rubric_fingerprint,
        "scored_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_seconds": round(elapsed, 1),
    }

    out_path = RUN_HISTORY_DIR / f"{args.version}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    s = payload["judged_summary"]
    print(f"\n{'=' * 72}")
    print(f"Judged summary — version {args.version}  ({elapsed / 60:.1f} min)")
    print("=" * 72)
    print(f"Task success mean score   {s['task_success']['mean_score']}")
    print(f"Task success pass rate    {s['task_success']['pass_rate']}")
    print(f"Clarity mean score        {s['clarity']['mean_score']}  "
          f"(n={s['clarity']['n_scored']})")
    print(f"Clarity pass rate         {s['clarity']['pass_rate']}")
    print(f"Faithfulness mean score   {s['faithfulness']['mean_score']}  "
          f"(n={s['faithfulness']['n_scored']})")
    print(f"Judge errors              {s['judge_errors']}")
    print("\nTask success by category:")
    for cat, score in s["task_success"]["by_category"].items():
        print(f"  {cat:<28} {score}")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
