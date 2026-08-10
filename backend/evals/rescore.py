"""Recompute deterministic metrics over an existing run, without the agent.

The runner deliberately stores the full trace — every tool call with its
arguments, the retrieved source docs, the reply text — so that a change to a
metric definition does not require re-running 70 agent traces. This is the
script that cashes that in.

It re-derives per-case scores from the stored trace and the golden set, then
re-aggregates, so a corrected metric applies to runs that were captured before
the correction existed. Nothing here calls an LLM or the agent, so it is exact
and takes about a second.

Run:  python -m evals.rescore                # every *_raw.json
      python -m evals.rescore --version v1   # just one
"""

from __future__ import annotations

import argparse
import json
import sys

import yaml

from config import RUN_HISTORY_DIR
from evals import metrics
from evals.runner import CASE_SETS


def load_cases(case_set: str) -> dict[str, dict]:
    """Expectations for one case set, keyed by id.

    A run records which set it came from. Rescoring a held-out run against the
    golden set would compare answers to different questions, so the set is read
    from the run rather than assumed.
    """
    data = yaml.safe_load(CASE_SETS[case_set].read_text(encoding="utf-8"))
    return {c["id"]: c for c in data["cases"]}


def rescore_row(row: dict, case: dict) -> dict:
    """Re-derive one case's deterministic scores from its stored trace.

    Scores come from the golden set rather than the copy embedded in the run,
    so that a fix to a case's expectations also propagates on rescore.
    """
    tool_calls = row["tool_calls"]
    row["trajectory"] = metrics.score_trajectory(case, tool_calls).__dict__
    row["retrieval"] = metrics.score_retrieval(case, row["retrieved_docs"]).__dict__
    row["escalation"] = metrics.score_escalation(case, tool_calls).__dict__
    row["forbidden_claims"] = metrics.score_forbidden_claims(case, row["reply"]).__dict__
    return row


def rescore_file(path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    # Runs written before --set existed carry no case_set and are all golden.
    case_set = payload.get("case_set", "golden")
    golden = load_cases(case_set)

    missing = [r["id"] for r in payload["cases"] if r["id"] not in golden]
    if missing:
        raise SystemExit(
            f"{path.name}: {len(missing)} case id(s) are not in the current "
            f"{case_set} set ({', '.join(missing[:5])}...). The run predates a "
            f"change to that set; rescoring it would compare against different "
            f"questions."
        )

    before = payload["deterministic_summary"]
    for row in payload["cases"]:
        rescore_row(row, golden[row["id"]])
    payload["deterministic_summary"] = metrics.aggregate(payload["cases"])
    payload["rescored"] = True

    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"path": path, "before": before, "after": payload["deterministic_summary"]}


def pct(v: float | None) -> str:
    return "  n/a" if v is None else f"{v:.1%}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", help="Rescore only this version, e.g. v1")
    args = parser.parse_args()

    pattern = f"{args.version}_raw.json" if args.version else "*_raw.json"
    paths = sorted(RUN_HISTORY_DIR.glob(pattern))
    if not paths:
        print(f"No runs matching {pattern}", file=sys.stderr)
        return 1

    for path in paths:
        r = rescore_file(path)
        b, a = r["before"], r["after"]
        print(f"\n{path.name}")
        for label, key, sub in (
            ("trajectory pass", "trajectory", "pass_rate"),
            ("retrieval recall", "retrieval", "mean_recall"),
            ("  recall|searched", "retrieval", "recall_given_searched"),
            ("  search rate", "retrieval", "search_rate"),
            ("escalation recall", "escalation", "recall"),
        ):
            old = b.get(key, {}).get(sub)
            new = a.get(key, {}).get(sub)
            mark = "" if old == new else "   <- changed"
            print(f"  {label:<18} {pct(old):>7} -> {pct(new):>7}{mark}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
