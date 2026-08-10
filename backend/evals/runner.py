"""Execute the agent across the golden set and score the deterministic metrics.

This is deliberately separate from judge scoring. The agent run is the
expensive, non-deterministic part; once it is captured, the deterministic
metrics are free to recompute and the LLM-judged metrics can be re-run without
re-running the agent. It also means a change to a metric definition does not
require re-running 70 agent traces.

Output: evals/run_history/<version>_raw.json

Run:  python -m evals.runner --version v1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from typing import Any

import yaml

from agent import core, llm
from config import (
    AGENT_MODEL,
    AGENT_NUM_CTX,
    AGENT_SEED,
    AGENT_TEMPERATURE,
    CHUNK_MAX_CHARS,
    CHUNK_OVERLAP_CHARS,
    EVALS_DIR,
    JUDGE_MODEL,
    MAX_AGENT_TURNS,
    RETRIEVAL_TOP_K,
    RUN_HISTORY_DIR,
)
from evals import metrics

# Two case files, scored identically but read very differently. `golden` is the
# set the prompts were derived from; `heldout` is the set they were not. See the
# header of heldout_set.yaml for why that distinction is the whole measurement.
CASE_SETS = {
    "golden": EVALS_DIR / "golden_set.yaml",
    "heldout": EVALS_DIR / "heldout_set.yaml",
}


def pct(value: float | None) -> str:
    """Render a rate, distinguishing 'not measured' from zero."""
    return "  n/a" if value is None else f"{value:.1%}"


def load_cases(
    only: str | None = None, category: str | None = None, case_set: str = "golden"
) -> list[dict]:
    data = yaml.safe_load(CASE_SETS[case_set].read_text(encoding="utf-8"))
    cases = data["cases"]
    if only:
        wanted = {c.strip() for c in only.split(",")}
        cases = [c for c in cases if c["id"] in wanted]
    if category:
        cases = [c for c in cases if c["category"] == category]
    return cases


def run_fingerprint(version: str, case_set: str) -> dict[str, Any]:
    """Everything that must match for a partial run to be safely resumable.

    Resuming across a config change would splice two different experiments into
    one file and report the result as a single number, which is the kind of
    error that is invisible once the run finishes.
    """
    return {
        "version": version,
        "case_set": case_set,
        "agent_model": AGENT_MODEL,
        "temperature": AGENT_TEMPERATURE,
        "seed": AGENT_SEED,
        "num_ctx": AGENT_NUM_CTX,
        "retrieval_top_k": RETRIEVAL_TOP_K,
        "max_agent_turns": MAX_AGENT_TURNS,
    }


def load_partial(path, fingerprint: dict) -> list[dict[str, Any]]:
    """Completed rows from an interrupted run, if they are still comparable."""
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("fingerprint") != fingerprint:
        print(
            f"Ignoring {path.name}: it was produced under a different config.\n"
            f"  partial: {data.get('fingerprint')}\n"
            f"  current: {fingerprint}",
            file=sys.stderr,
        )
        return []
    return data.get("cases", [])


def save_partial(path, fingerprint: dict, rows: list[dict[str, Any]]) -> None:
    """Write progress after every case, atomically.

    Written to a temp file and moved into place so that a crash midway through
    the write leaves the previous checkpoint intact rather than a truncated file.
    """
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps({"fingerprint": fingerprint, "cases": rows}, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)


def run_case(case: dict, version: str) -> dict[str, Any]:
    """Run one case and score everything that does not need an LLM."""
    history = case.get("history") or []
    result = core.run(message=case["query"], history=history, version=version)

    tool_calls = [
        {"turn": t.turn, "name": t.name, "arguments": t.arguments, "meta": t.meta}
        for t in result.tool_calls
    ]

    trajectory = metrics.score_trajectory(case, tool_calls)
    retrieval = metrics.score_retrieval(case, result.retrieved_docs)
    escalation = metrics.score_escalation(case, tool_calls)
    forbidden = metrics.score_forbidden_claims(case, result.reply)

    return {
        "id": case["id"],
        "category": case["category"],
        "persona": case.get("persona"),
        "query": case["query"],
        "history": history,
        "reply": result.reply,
        "tool_calls": tool_calls,
        "tool_names": result.tool_names(),
        "grounding_context": result.grounding_context(),
        "retrieved_docs": result.retrieved_docs,
        "turns": result.turns,
        "hit_turn_limit": result.hit_turn_limit,
        "latency_ms": result.latency_ms,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.prompt_tokens + result.completion_tokens,
        # expectations carried forward so judge scoring and the failure
        # analysis do not need to re-open the golden set
        "expected_tool_calls": case.get("expected_tool_calls") or [],
        "expected_source_docs": case.get("expected_source_docs") or [],
        "expected_answer_criteria": case.get("expected_answer_criteria") or [],
        "requires_escalation": bool(case.get("requires_escalation")),
        # deterministic scores
        "trajectory": trajectory.__dict__,
        "retrieval": retrieval.__dict__,
        "escalation": escalation.__dict__,
        "forbidden_claims": forbidden.__dict__,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the agent over the golden set.")
    parser.add_argument("--version", default="v1", help="Agent version, e.g. v1")
    parser.add_argument("--only", help="Comma-separated case ids")
    parser.add_argument("--category", help="Restrict to one category")
    parser.add_argument("--out", help="Override output path")
    parser.add_argument(
        "--set", dest="case_set", default="golden", choices=sorted(CASE_SETS),
        help="Which case file to run. 'heldout' is the generalisation probe.",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Ignore any checkpoint and run every case from scratch.",
    )
    args = parser.parse_args()

    ok, detail = llm.health_check()
    if not ok:
        print(f"ERROR: {detail}", file=sys.stderr)
        return 1
    print(detail)

    cases = load_cases(only=args.only, category=args.category, case_set=args.case_set)
    print(
        f"Running {len(cases)} cases from the {args.case_set} set "
        f"against agent version {args.version}\n"
    )

    # Checkpointing. A 70-case run on the 14b model takes over an hour, and
    # until now a crash, a lid close, or an Ollama restart lost every completed
    # case because the JSON was only written at the end.
    RUN_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.case_set == "golden" else f"_{args.case_set}"
    partial_path = RUN_HISTORY_DIR / f"{args.version}{suffix}_partial.json"
    fingerprint = run_fingerprint(args.version, args.case_set)

    rows: list[dict[str, Any]] = [] if args.no_resume else load_partial(
        partial_path, fingerprint
    )
    done = {r["id"] for r in rows}
    if done:
        print(f"Resuming from checkpoint: {len(done)} cases already complete.\n")
    total = len(cases)
    cases = [c for c in cases if c["id"] not in done]

    started = time.perf_counter()

    for i, case in enumerate(cases, 1):
        row = run_case(case, args.version)
        rows.append(row)
        save_partial(partial_path, fingerprint, rows)

        flags = []
        if row["trajectory"]["expected"] and not row["trajectory"]["passed"]:
            flags.append("TRAJ")
        if row["requires_escalation"] and not row["escalation"]["escalated"]:
            flags.append("MISSED-ESC")
        if row["retrieval"]["scored"] and not row["retrieval"]["hit"]:
            flags.append("RETR")
        if row["forbidden_claims"]["scored"] and not row["forbidden_claims"]["passed"]:
            flags.append("FORBIDDEN")
        if row["hit_turn_limit"]:
            flags.append("TURN-LIMIT")

        status = " ".join(flags) if flags else "ok"
        # flush per case: redirected to a file, Python buffers ~8KB, which is
        # about 60 of these lines — so a 70-case run showed no output at all
        # until it finished, and a slow run was indistinguishable from a hung one.
        print(
            f"[{len(done) + i:>2}/{total}] {row['id']:<10} {row['category']:<26} "
            f"{row['latency_ms']:>6}ms  tools={','.join(row['tool_names']) or '-':<40} {status}",
            flush=True,
        )

    elapsed = time.perf_counter() - started
    summary = metrics.aggregate(rows)

    payload = {
        "version": args.version,
        "case_set": args.case_set,
        "run_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        # Everything that can change a result, recorded with the result.
        # temperature / seed / num_ctx were missing from the first version of
        # this, which is how two runs of "the same" config differed by 15pp on
        # retrieval recall without the files showing any reason why.
        "config": {
            "agent_model": AGENT_MODEL,
            "judge_model": JUDGE_MODEL,
            "chunk_max_chars": CHUNK_MAX_CHARS,
            "chunk_overlap_chars": CHUNK_OVERLAP_CHARS,
            "retrieval_top_k": RETRIEVAL_TOP_K,
            "max_agent_turns": MAX_AGENT_TURNS,
            "temperature": AGENT_TEMPERATURE,
            "seed": AGENT_SEED,
            "num_ctx": AGENT_NUM_CTX,
        },
        "deterministic_summary": summary,
        "cases": rows,
    }

    # Held-out runs get their own filename so they can never overwrite, or be
    # mistaken for, a golden-set run of the same version.
    out_path = (
        __import__("pathlib").Path(args.out)
        if args.out
        else RUN_HISTORY_DIR / f"{args.version}{suffix}_raw.json"
    )
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    # The run completed, so the checkpoint has no further purpose. Leaving it
    # would make the next run of this version resume from a finished set and
    # execute nothing.
    partial_path.unlink(missing_ok=True)

    print(f"\n{'=' * 72}")
    print(f"Deterministic summary — version {args.version}  ({elapsed / 60:.1f} min)")
    print("=" * 72)
    t, r, e, o = (
        summary["trajectory"],
        summary["retrieval"],
        summary["escalation"],
        summary["operational"],
    )
    print(f"Trajectory pass rate       {pct(t['pass_rate'])}  (n={t['n_scored']})")
    print(f"  mean tool match ratio    {pct(t['mean_tool_match_ratio'])}")
    print(f"Retrieval recall           {pct(r['mean_recall'])}  (n={r['n_scored']})")
    print(f"  precision                {pct(r['mean_precision'])}")
    print(f"  hit rate                 {pct(r['hit_rate'])}")
    print(f"  search rate              {pct(r['search_rate'])}  (agent chose to search)")
    print(f"  recall | searched        {pct(r['recall_given_searched'])}  (n={r['n_searched']}) <- retriever quality")
    print(f"Escalation recall          {pct(e['recall'])}  (n={e['n_required']} required)")
    print(f"  over-escalation rate     {pct(e['over_escalation_rate'])}  (n={e['n_not_required']})")
    print(f"Forbidden-claim pass rate  {pct(summary['forbidden_claims']['pass_rate'])}"
          f"  (n={summary['forbidden_claims']['n_scored']})")
    print(f"Median latency             {o['median_latency_ms']}ms")
    print(f"Mean tokens per case       {o['mean_total_tokens']}")
    print(f"Turn-limit hits            {o['turn_limit_hits']}")
    print(f"Cases with no tool calls   {o['no_tool_calls']}")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
