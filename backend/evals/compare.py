"""Compare two evaluation runs.

The brief calls the before/after the single most important deliverable, so this
does more than diff averages. An aggregate that improves can hide cases that got
worse, and a change that fixes six cases while breaking three is a different
result from one that fixes three and breaks none — the means are identical.

So alongside per-metric deltas this reports **case-level flips**: which cases
went failing -> passing (fixed), which went passing -> failing (regressed), and
which never worked at all. Regressions are printed even when every headline
number went up.

Run:  python -m evals.compare --base v1 --new v2
      python -m evals.compare --base v1 --new v2 --markdown docs/eval_comparison.md
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from config import RUN_HISTORY_DIR


def load(version: str) -> dict[str, Any]:
    scored = RUN_HISTORY_DIR / f"{version}.json"
    raw = RUN_HISTORY_DIR / f"{version}_raw.json"
    path = scored if scored.exists() else raw
    if not path.exists():
        raise SystemExit(f"No run found for {version!r} ({scored} / {raw})")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["_source"] = path.name
    return data


def pct(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.1%}"


def delta(base: float | None, new: float | None) -> str:
    """Signed change in percentage points, or a marker when not comparable."""
    if base is None or new is None:
        return "—"
    d = (new - base) * 100
    if abs(d) < 0.05:
        return "0.0"
    return f"{d:+.1f}"


# Metrics to compare: (label, path into the summary, higher_is_better)
DETERMINISTIC = [
    ("Trajectory pass rate", ("deterministic_summary", "trajectory", "pass_rate"), True),
    ("Tool match ratio", ("deterministic_summary", "trajectory", "mean_tool_match_ratio"), True),
    ("Retrieval recall", ("deterministic_summary", "retrieval", "mean_recall"), True),
    ("Retrieval precision", ("deterministic_summary", "retrieval", "mean_precision"), True),
    ("Retrieval hit rate", ("deterministic_summary", "retrieval", "hit_rate"), True),
    ("Escalation recall", ("deterministic_summary", "escalation", "recall"), True),
    ("Over-escalation rate", ("deterministic_summary", "escalation", "over_escalation_rate"), False),
    ("Forbidden-claim pass", ("deterministic_summary", "forbidden_claims", "pass_rate"), True),
]

JUDGED = [
    ("Task success (mean)", ("judged_summary", "task_success", "mean_score"), True),
    ("Task success (pass rate)", ("judged_summary", "task_success", "pass_rate"), True),
    ("Clarity (mean)", ("judged_summary", "clarity", "mean_score"), True),
    ("Clarity (pass rate)", ("judged_summary", "clarity", "pass_rate"), True),
    ("Faithfulness (mean)", ("judged_summary", "faithfulness", "mean_score"), True),
]


def dig(data: dict, path: tuple[str, ...]) -> Any:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def flips(base: dict, new: dict, predicate) -> dict[str, list[str]]:
    """Case-level movement on a boolean predicate."""
    b = {c["id"]: c for c in base["cases"]}
    n = {c["id"]: c for c in new["cases"]}
    fixed, regressed, still, kept = [], [], [], []
    for cid in sorted(set(b) & set(n)):
        bp, np_ = predicate(b[cid]), predicate(n[cid])
        if bp is None or np_ is None:
            continue
        if not bp and np_:
            fixed.append(cid)
        elif bp and not np_:
            regressed.append(cid)
        elif not bp and not np_:
            still.append(cid)
        else:
            kept.append(cid)
    return {"fixed": fixed, "regressed": regressed, "still_failing": still, "still_passing": kept}


def p_trajectory(c: dict):
    return c["trajectory"]["passed"] if c["trajectory"]["expected"] else None


def p_retrieval(c: dict):
    return c["retrieval"]["hit"] if c["retrieval"]["scored"] else None


def p_escalation(c: dict):
    return c["escalation"]["escalated"] if c["requires_escalation"] else None


def build(base: dict, new: dict) -> dict[str, Any]:
    rows = []
    for label, path, hib in DETERMINISTIC + JUDGED:
        b, n = dig(base, path), dig(new, path)
        if b is None and n is None:
            continue
        rows.append({
            "metric": label, "base": b, "new": n, "delta": delta(b, n),
            "higher_is_better": hib,
            # None when unchanged or not comparable — an unchanged metric is
            # neither an improvement nor a regression, and marking it "DOWN"
            # (the first version of this) misreads a flat result as a loss.
            "improved": (
                None if (b is None or n is None or b == n)
                else ((n > b) == hib)
            ),
        })

    return {
        "base_version": base["version"],
        "new_version": new["version"],
        "metrics": rows,
        "trajectory_flips": flips(base, new, p_trajectory),
        "retrieval_flips": flips(base, new, p_retrieval),
        "escalation_flips": flips(base, new, p_escalation),
        "config": {"base": base.get("config", {}), "new": new.get("config", {})},
    }


def report(cmp: dict, base: dict, new: dict) -> None:
    b, n = cmp["base_version"], cmp["new_version"]
    print(f"\n{'=' * 74}")
    print(f"{b} -> {n}   ({base['_source']} vs {new['_source']})")
    print("=" * 74)
    print(f"{'metric':<28} {b:>10} {n:>10} {'delta (pp)':>12}")
    print("-" * 74)
    for r in cmp["metrics"]:
        fmt = pct if "mean" not in r["metric"].lower() or "rate" in r["metric"].lower() else str
        bs = pct(r["base"]) if r["base"] is None or r["base"] <= 1 else str(r["base"])
        ns = pct(r["new"]) if r["new"] is None or r["new"] <= 1 else str(r["new"])
        mark = ""
        if r["improved"] is True:
            mark = "  up"
        elif r["improved"] is False:
            mark = "  DOWN"
        print(f"{r['metric']:<28} {bs:>10} {ns:>10} {r['delta']:>12}{mark}")

    for name, key in (("Trajectory", "trajectory_flips"),
                      ("Retrieval", "retrieval_flips"),
                      ("Escalation", "escalation_flips")):
        f = cmp[key]
        print(f"\n{name} — case movement")
        print(f"  fixed ({len(f['fixed'])}):        {', '.join(f['fixed']) or '—'}")
        print(f"  REGRESSED ({len(f['regressed'])}):    {', '.join(f['regressed']) or '—'}")
        print(f"  still failing ({len(f['still_failing'])}): {', '.join(f['still_failing']) or '—'}")

    cb, cn = cmp["config"]["base"], cmp["config"]["new"]
    changed = {k: (cb.get(k), cn.get(k)) for k in set(cb) | set(cn) if cb.get(k) != cn.get(k)}
    print(f"\nConfig differences: {changed or 'none'}")

    total_reg = sum(len(cmp[k]["regressed"]) for k in
                    ("trajectory_flips", "retrieval_flips", "escalation_flips"))
    print("\n" + "-" * 74)
    if total_reg:
        print(f"NOTE: {total_reg} case-level regression(s) across all metrics. Aggregate")
        print("improvement does not mean nothing got worse — see the lists above.")
    else:
        print("No case-level regressions on any deterministic metric.")


def to_markdown(cmp: dict) -> str:
    b, n = cmp["base_version"], cmp["new_version"]
    out = [f"# Evaluation comparison — {b} vs {n}", ""]
    out += ["| Metric | " + b + " | " + n + " | Δ (pp) |", "|---|---|---|---|"]
    for r in cmp["metrics"]:
        bs = pct(r["base"]) if r["base"] is None or r["base"] <= 1 else str(r["base"])
        ns = pct(r["new"]) if r["new"] is None or r["new"] <= 1 else str(r["new"])
        out.append(f"| {r['metric']} | {bs} | {ns} | **{r['delta']}** |")

    for name, key in (("Trajectory", "trajectory_flips"),
                      ("Retrieval", "retrieval_flips"),
                      ("Escalation", "escalation_flips")):
        f = cmp[key]
        out += ["", f"## {name} — case movement", "",
                f"- **Fixed ({len(f['fixed'])}):** {', '.join(f'`{c}`' for c in f['fixed']) or '—'}",
                f"- **Regressed ({len(f['regressed'])}):** {', '.join(f'`{c}`' for c in f['regressed']) or '—'}",
                f"- **Still failing ({len(f['still_failing'])}):** {', '.join(f'`{c}`' for c in f['still_failing']) or '—'}"]

    cb, cn = cmp["config"]["base"], cmp["config"]["new"]
    changed = {k: (cb.get(k), cn.get(k)) for k in set(cb) | set(cn) if cb.get(k) != cn.get(k)}
    out += ["", "## Configuration changed between runs", ""]
    out += [f"- `{k}`: {v[0]} → {v[1]}" for k, v in sorted(changed.items())] or ["- none"]
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare two eval runs.")
    ap.add_argument("--base", default="v1")
    ap.add_argument("--new", default="v2")
    ap.add_argument("--markdown", help="Also write a markdown table to this path")
    ap.add_argument("--json", dest="json_out", help="Write the comparison as JSON")
    args = ap.parse_args()

    base, new = load(args.base), load(args.new)
    cmp = build(base, new)
    report(cmp, base, new)

    if args.markdown:
        from pathlib import Path
        Path(args.markdown).write_text(to_markdown(cmp), encoding="utf-8")
        print(f"\nWrote {args.markdown}")
    if args.json_out:
        from pathlib import Path
        Path(args.json_out).write_text(json.dumps(cmp, indent=2), encoding="utf-8")
        print(f"Wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
