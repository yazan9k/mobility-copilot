"""Measure how much the judge can be trusted.

Runs the same TaskSuccess metric used in judge_scoring.py over the hand-labelled
calibration set and compares its verdicts to the human labels. Reports raw
agreement, Cohen's kappa, and — the useful part — a breakdown by probe group,
so a failure points at *which* judge defect is real rather than just saying the
judge is unreliable.

Reading the output:

  false negative   judge failed an answer a human passed. Clustering in the
                   over_penalisation group means the judge marks down correct
                   answers for extra detail, unfamiliar names, terseness, or
                   correct refusal.
  false positive   judge passed an answer a human failed. Clustering in the
                   under_penalisation group means fluent, well-shaped answers
                   slip through with wrong figures — the more dangerous defect,
                   because it inflates the headline score.

Cohen's kappa is reported alongside raw agreement because raw agreement is
flattered by chance. On a balanced 10/10 set, chance agreement is ~50%, so a
judge that agreed 60% of the time would look mediocre-but-usable at first
glance and is in fact barely better than a coin flip (kappa ~0.2).

Run:  python -m evals.calibration --label rubric-v1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from typing import Any

import yaml
from deepeval.test_case import LLMTestCase

from config import EVALS_DIR, RUN_HISTORY_DIR
from evals.judge_scoring import build_metrics
from evals.ollama_judge import OllamaJudge, health_check

CALIBRATION_PATH = EVALS_DIR / "calibration_set.yaml"

# Kappa interpretation bands (Landis & Koch, the conventional reading).
KAPPA_BANDS = [
    (0.81, "almost perfect"),
    (0.61, "substantial"),
    (0.41, "moderate"),
    (0.21, "fair"),
    (0.01, "slight"),
    (-1.0, "none / worse than chance"),
]


def interpret_kappa(k: float) -> str:
    for floor, label in KAPPA_BANDS:
        if k >= floor:
            return label
    return "none"


def cohens_kappa(pairs: list[tuple[bool, bool]]) -> float:
    """Agreement between two raters on a binary label, corrected for chance."""
    n = len(pairs)
    if n == 0:
        return 0.0

    observed = sum(1 for h, j in pairs if h == j) / n

    human_pass = sum(1 for h, _ in pairs if h) / n
    judge_pass = sum(1 for _, j in pairs if j) / n
    expected = (human_pass * judge_pass) + ((1 - human_pass) * (1 - judge_pass))

    if expected == 1.0:  # degenerate: both raters unanimous the same way
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1 - expected)


def run(threshold: float, limit: int | None) -> dict[str, Any]:
    data = yaml.safe_load(CALIBRATION_PATH.read_text(encoding="utf-8"))
    cases = data["cases"][:limit] if limit else data["cases"]

    judge = OllamaJudge()
    task_success, _ = build_metrics(judge, threshold)

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    print(f"Calibrating {judge.get_model_name()} over {len(cases)} labelled cases "
          f"(pass threshold {threshold})\n")
    print(f"{'case':<10} {'probe':<20} {'human':<6} {'judge':<6} {'score':<6} verdict")
    print("-" * 72)

    for case in cases:
        expected_output = (
            "The response must satisfy all of the following:\n"
            + "\n".join(f"- {c}" for c in case["criteria"])
        )
        tc = LLMTestCase(
            input=case["query"],
            actual_output=case["candidate_answer"],
            expected_output=expected_output,
        )

        try:
            task_success.measure(tc)
            score = task_success.score
            judge_pass = bool(score is not None and score >= threshold)
            error = None
        except Exception as exc:  # noqa: BLE001
            score, judge_pass, error = None, False, str(exc)

        human_pass = case["human_label"] == "pass"
        agree = human_pass == judge_pass

        if agree:
            verdict = "agree"
        elif human_pass and not judge_pass:
            verdict = "FALSE NEGATIVE"
        else:
            verdict = "FALSE POSITIVE"

        rows.append({
            "id": case["id"],
            "probe": case["probe"],
            "human_label": case["human_label"],
            "human_pass": human_pass,
            # The assistant's independent label is kept as a second rater, so the
            # report can show judge-vs-human (authoritative) alongside
            # human-vs-assistant (how stable the labels themselves are).
            "assistant_label": case.get("assistant_label"),
            "human_note": case.get("assistant_note", ""),
            "judge_pass": judge_pass,
            "judge_score": score,
            "judge_reason": getattr(task_success, "reason", None),
            "agree": agree,
            "verdict": verdict,
            "error": error,
        })

        score_str = "err" if score is None else f"{score:.2f}"
        print(f"{case['id']:<10} {case['probe']:<20} "
              f"{case['human_label']:<6} {'pass' if judge_pass else 'fail':<6} "
              f"{score_str:<6} {verdict}")

    elapsed = time.perf_counter() - started
    pairs = [(r["human_pass"], r["judge_pass"]) for r in rows]

    n = len(rows)
    agreements = sum(1 for r in rows if r["agree"])
    false_neg = [r for r in rows if r["verdict"] == "FALSE NEGATIVE"]
    false_pos = [r for r in rows if r["verdict"] == "FALSE POSITIVE"]

    by_probe: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "agree": 0, "false_negative": 0, "false_positive": 0}
    )
    for r in rows:
        g = by_probe[r["probe"]]
        g["n"] += 1
        if r["agree"]:
            g["agree"] += 1
        elif r["verdict"] == "FALSE NEGATIVE":
            g["false_negative"] += 1
        else:
            g["false_positive"] += 1

    kappa = cohens_kappa(pairs)

    # Second-rater check: how far apart were the two humans labelling this set?
    rater_pairs = [
        (r["human_pass"], r["assistant_label"] == "pass")
        for r in rows if r["assistant_label"]
    ]
    rater_agreement = (
        round(sum(1 for a, b in rater_pairs if a == b) / len(rater_pairs), 4)
        if rater_pairs else None
    )

    return {
        "inter_rater_agreement": rater_agreement,
        "inter_rater_disagreements": [
            r["id"] for r in rows
            if r["assistant_label"] and (r["human_pass"] != (r["assistant_label"] == "pass"))
        ],
        "n_cases": n,
        "threshold": threshold,
        "judge_model": judge.get_model_name(),
        "elapsed_seconds": round(elapsed, 1),
        "raw_agreement": round(agreements / n, 4) if n else 0.0,
        "cohens_kappa": round(kappa, 4),
        "kappa_reading": interpret_kappa(kappa),
        "false_negatives": len(false_neg),
        "false_positives": len(false_pos),
        "by_probe": dict(by_probe),
        "judge_errors": sum(1 for r in rows if r["error"]),
        "cases": rows,
    }


def report(result: dict[str, Any]) -> None:
    print(f"\n{'=' * 72}")
    print(f"Judge calibration — {result['judge_model']}  "
          f"({result['elapsed_seconds'] / 60:.1f} min)")
    print("=" * 72)
    print(f"Raw agreement       {result['raw_agreement']:.1%}  "
          f"({result['n_cases']} cases)")
    print(f"Cohen's kappa       {result['cohens_kappa']:.3f}  "
          f"({result['kappa_reading']})")
    print(f"False negatives     {result['false_negatives']}  "
          f"(judge failed what a human passed)")
    print(f"False positives     {result['false_positives']}  "
          f"(judge passed what a human failed)")
    if result["judge_errors"]:
        print(f"Judge errors        {result['judge_errors']}")

    print("\nBy probe group:")
    print(f"  {'group':<20} {'n':>2}  {'agree':>5}  {'FN':>3}  {'FP':>3}")
    for group, stats in sorted(result["by_probe"].items()):
        print(f"  {group:<20} {stats['n']:>2}  {stats['agree']:>5}  "
              f"{stats['false_negative']:>3}  {stats['false_positive']:>3}")

    disagreements = [r for r in result["cases"] if not r["agree"]]
    if disagreements:
        print(f"\nDisagreements ({len(disagreements)}):")
        for r in disagreements:
            print(f"\n  {r['id']} [{r['probe']}] — {r['verdict']}")
            print(f"    human said {r['human_label']}: {r['human_note'].strip()[:150]}")
            reason = (r["judge_reason"] or r["error"] or "")
            print(f"    judge scored {r['judge_score']}: {reason.strip()[:250]}")

    print("\n" + "-" * 72)
    kappa = result["cohens_kappa"]
    if kappa >= 0.61:
        print("READING: agreement is substantial. G-Eval scores can be reported as a")
        print("headline metric, with the deterministic metrics alongside them.")
    elif kappa >= 0.41:
        print("READING: agreement is moderate. Report G-Eval as directional only, and")
        print("lead with the deterministic trajectory and retrieval metrics.")
    else:
        print("READING: agreement is weak. Do NOT report G-Eval as a headline number.")
        print("The deterministic metrics become the result; G-Eval is commentary.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure judge/human agreement.")
    parser.add_argument("--label", default="rubric-v1",
                        help="Tag for the output file, e.g. rubric-v1")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    ok, detail = health_check()
    if not ok:
        print(f"ERROR: {detail}", file=sys.stderr)
        return 1
    print(detail)

    result = run(args.threshold, args.limit)
    report(result)

    RUN_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    out = RUN_HISTORY_DIR / f"calibration_{args.label}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
