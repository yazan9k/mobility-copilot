# Global Mobility Copilot

An internal relocation assistant for **Meridian Systems, a fictional company** — and, more to the point, the evaluation system built around it.

The chatbot is not the deliverable. The deliverable is a measurement harness and a documented, numbers-backed improvement where **every change traces to a specific measured failure**.

> **All content is synthetic.** Meridian Systems does not exist. Every policy document, visa record, employee, and persona is fabricated. Nothing here is scraped from real company policy or any government immigration source, and none of it is usable as immigration or tax advice.

---

## What it does

Answers relocation questions grounded in a policy corpus, looks up permit requirements, generates document checklists and timelines, files HR tickets — and **hands off to a human** when a question falls into one of five categories policy says it must.

That last one is a first-class feature. An assistant that confidently invents a visa requirement is worse than no assistant.

## How it's evaluated

Three levels, because a single "is it good?" score tells you nothing actionable:

| Level | Question | Method | Reliability |
|---|---|---|---|
| **Trajectory** | Right tools, right arguments? | Exact match against labels | Deterministic |
| **Retrieval** | Right policy documents returned? | Precision/recall against labels | Deterministic |
| **End-to-end** | Did it meet the employee's need? | LLM-as-judge (G-Eval) | Calibrated — see below |
| **Clarity** | Could a non-expert act on it? | LLM-as-judge, separate dimension | Calibrated |

Two of these need no model at all, which is deliberate: with a local judge, the reproducible metrics carry the weight.

### Two case sets, because one cannot detect overfitting

`golden_set.yaml` (70 cases) is what every prompt was derived from. Scoring a prompt on the cases that shaped it cannot distinguish *"the rules are good"* from *"the rules absorbed these cases."*

`heldout_set.yaml` (20 cases) was written **before** the v4 prompts existed, drawn from corpus documents the failure analysis never opened, and phrased without the vocabulary of the rule each case instantiates. The gap between a prompt's golden score and its held-out score is the overfitting measurement.

It earned its place immediately: **v4-principled scored best on the golden set (72.1%) and worst on held-out (40.0%)**. Measuring only the golden set would have shipped the worst-generalising version as the winner.

---

## Results

`qwen2.5:7b`, temperature 0, fixed seed. Deterministic metrics — no LLM scores these.

| Version | Traj (golden) | Traj (held-out) | Gap | Escalation | Search rate | Recall \| searched |
|---|---:|---:|---:|---:|---:|---:|
| v1 (naive baseline) | 50.8% | 55.0% | +4.2 | 0.0% | 72.4% | 88.1% |
| v2 | 31.1% | — | — | 40.0% | 20.7% | 100% |
| v3 | 63.9% | 55.0% | −8.9 | 30.0% | 82.8% | 91.7% |
| **v4-enumerated** | **63.9%** | **60.0%** | **−3.9** | 35.0% | 75.9% | 88.6% |
| v4-principled | 72.1% | 40.0% | −32.1 | 35.0% | 93.1% | 100% |
| v4-verbatim | 65.6% | 60.0% | −5.6 | 20.0% | 93.1% | 96.3% |
| v5-tooldesc | 63.9% | 55.0% | −8.9 | 25.0% | 82.8% | 93.8% |

**What works:** retrieval (88–100% recall whenever a search happens, including 100% on unseen questions), and tool selection — excluding escalation cases, **v4-principled reaches 90.2%** trajectory and v4-enumerated 78.0%.

**What doesn't:** escalation, 20–40% across every version. It accounts for **59%** of v4-enumerated's remaining trajectory failures (13 of 22) and **76%** of v4-principled's (13 of 17).

**Answer quality barely moved.** Judged on 24 paired cases: task success 29.2% → 33.3% (one case, i.e. noise), clarity 25.0% → **41.7%**, faithfulness 91.3% → 95.0%. The deterministic metrics improved substantially and the answers did not follow. Tracking trajectory alone would have reported a 13-point win and called it a success — the three-level design is the only reason that claim wasn't made.

**Every PRD success target was missed.** M2 trajectory 63.9% against >85%, M3 retrieval recall 67.2% against >90%, M5 escalation 35% against 100%. The project's stated deliverable was a documented, evidence-driven iteration loop rather than a passing scorecard, and that is what it produced — but the targets in [docs/prd.md](docs/prd.md) are not met and are not on track to be met by prompt changes.

**On unseen questions, the improvement is about a third of what the golden set reports.**

| v1 → v4-enumerated | Golden | Held-out |
|---|---:|---:|
| Trajectory | +13.1pp | **+5.0pp** |
| Escalation | +35.0pp | **+10.0pp** |

Most of the measured gain sits on the cases that shaped the prompts. v1 also scores *higher* on held-out than golden (55.0% vs 50.8%), suggesting those cases are somewhat easier — which makes v4's 60.0% weaker still. This is exactly what the held-out set was built to catch, and it caught the project's own headline.

One caveat remains: **v3's golden figure is the higher of two conflicting runs** of identical configuration (62.3% and 63.9%). Unexplained; see Known open problems.

**Reproducibility is verified, not assumed.** v1 and v2 were re-run under fully recorded configuration and came back identical — same metrics, **0 of 70 cases** taking a different tool path, eight days apart.

Three of the maintainer's own hypotheses were falsified by measurement — prompt length as the cause of the v2 collapse, principles generalising better than an enumerated list, and the tool description being the lever for escalation. Full write-ups in [docs/eval_comparison.md](docs/eval_comparison.md) and [docs/case_study.md](docs/case_study.md).

### The judge is calibrated, not assumed

An LLM-judged score is an unverified number until you know how often the judge agrees with a human. `evals/calibration_set.yaml` holds 20 hand-labelled cases; `evals/calibration.py` measures agreement and reports **Cohen's kappa** (raw agreement is flattered by chance — a judge that fails everything scores 50% raw agreement and kappa 0.000).

Calibration **disproved the hypothesis that prompted it**, then found a real defect: the judge had detached a qualifier from a rubric step and was penalising answers for stating figures at all. Fixing that moved agreement from **80% / κ 0.588** to **95% / κ 0.900**. The reasoning is recorded in `evals/judge_scoring.py`.

---

## Running it

### Prerequisites

```bash
brew install ollama            # or see ollama.com
ollama pull qwen2.5:7b         # agent
ollama pull qwen2.5:14b        # judge — deliberately larger than the agent
```

Everything runs locally. No API keys, no spend. Provider is isolated behind `agent/llm.py`, so swapping to a hosted API means reimplementing one function.

**The machine must stay awake** for any eval run — inference is local, so a sleeping laptop stops the run.

### Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv -r backend/requirements.txt

cd backend
../.venv/bin/python -m rag.ingest      # chunk + embed the corpus -> Chroma
../.venv/bin/python -m data.init_db    # create the mock HR database
```

`rag.ingest` downloads a ~79 MB ONNX embedding model on first run, then works offline.

### Serve

```bash
cd backend
../.venv/bin/uvicorn api.main:app --port 8010
```

```bash
curl -s localhost:8010/health
curl -s -X POST localhost:8010/chat -H 'Content-Type: application/json' \
  -d '{"message":"I am relocating from Dubai to Singapore, what visa do I need and what does the company cover?"}'
```

`/chat` returns the reply **and the tool-call trajectory** — the trajectory is part of the contract, because trajectory-level evaluation depends on it.

### Evaluate

```bash
cd backend

# 1. run the agent -> deterministic metrics. ~25 min for 70 cases.
../.venv/bin/python -m evals.runner --version v4-enumerated
../.venv/bin/python -m evals.runner --version v4-enumerated --set heldout

# 2. add the judged metrics (task success, clarity, faithfulness).
#    Much slower — 14B judge, several calls per case. --sample takes a
#    stratified subset, the same cases for every version so it stays paired.
../.venv/bin/python -m evals.judge_scoring --version v1 --sample 24
../.venv/bin/python -m evals.judge_scoring --version v4-enumerated --sample 24

# 3. compare, with case-level regressions
../.venv/bin/python -m evals.compare --base v1 --new v4-enumerated \
    --markdown ../docs/eval_comparison.md

# 4. check the judge itself
../.venv/bin/python -m evals.calibration --label rubric-v2

# 5. regression gate (reads completed runs; no model calls, runs in <1s)
../.venv/bin/python -m pytest evals/test_agent_eval.py -q
EVAL_VERSION=v2 ../.venv/bin/python -m pytest evals/test_agent_eval.py -q  # fails, as it should

# 6. recompute deterministic metrics after a metric change, without re-running
#    the agent — reads the stored trace
../.venv/bin/python -m evals.rescore
```

Agent runs and judge scoring are split on purpose: the agent pass is the expensive, non-deterministic half, so changing a metric definition or re-judging with a revised rubric never means re-running 70 agent traces. `rescore` is what cashes that in.

**Runs checkpoint after every case.** An interrupted run resumes from where it stopped rather than starting over, and refuses to resume if the model, seed, or retrieval config changed — otherwise two different experiments would be spliced into one file and reported as a single number.

### About the pytest gate

It holds **regression guards, not the aspirations in `docs/prd.md`.** A gate set to an unmet target fails on every run, stops being read, and can no longer tell you when something actually breaks. Thresholds sit just below what the best version achieves, and the suite answers one question: *did this change make things worse than the best result on record?*

Escalation is the exception. It is the safety metric, the requirement is 100%, and the measured result is 35%. Rather than relax it to something passing — which would convert a known safety gap into a green test — it stays at 100%, is marked `xfail` with the diagnosis attached, and a separate floor test catches genuine regression below the measured level.

---

## Layout

```
backend/
  config.py               every tunable constant in one file
  rag/
    corpus/               10 fabricated policy documents (~9.5k words)
    ingest.py             section-aware chunking -> Chroma (local ONNX embeddings)
    retrieve.py           top-k search, returns source filenames for scoring
  agent/
    llm.py                provider seam — the only Ollama-aware module
    tools.py              6 tools; descriptions versioned as an eval lever
    prompts.py            v1 naive baseline through v5; v4 arms share a
                          byte-identical preamble and tail so only the
                          escalation block varies
    core.py               the agent loop, ~110 lines
  data/                   fabricated visa dataset + mock HR SQLite
  api/main.py             POST /chat, GET /health
  evals/
    golden_set.yaml       70 labelled cases across 6 categories
    heldout_set.yaml      20 cases the prompts were never derived from
    calibration_set.yaml  20 hand-labelled cases for judging the judge
    metrics.py            deterministic scoring, no LLM
    ollama_judge.py       DeepEval judge over a local model
    runner.py             agent pass -> deterministic metrics; checkpointed
    rescore.py            recompute metrics over a stored trace, no agent
    judge_scoring.py      adds task success, clarity, faithfulness
    calibration.py        judge/human agreement + Cohen's kappa
    compare.py            per-version comparison incl. case-level regressions
    test_agent_eval.py    regression gate over completed runs
    run_history/          versioned results, golden and held-out
docs/
  prd.md                     scope and success metrics
  personas.md                5 synthetic personas (eval design scaffolding)
  corpus_outline.md          the shared fabricated facts the corpus agrees on
  failure_analysis_v1.md     8 numbered findings -> 3 changes
  escalation_invariants.md   the escalation spec the prompts derive from
  eval_comparison.md         all versions per metric, and what got falsified
  case_study.md              the write-up, with architecture diagram
```

---

## Deviations from the original brief

Recorded here rather than buried, since a reviewer should be able to see what changed and why.

- **Local models instead of the Anthropic API.** A zero-spend constraint. The cost is real: weaker tool-calling, and a judge that needed calibrating before its scores meant anything. `agent/llm.py` keeps the swap to one function.
- **Frontend and deployment deferred.** The brief's own MVP cut line puts the agent, the golden set, and one documented iteration cycle above the UI. The eval dashboard lives in `docs/eval_comparison.md` as tables rather than as a React page.
- **A `Clarity` metric was added** after a human labeller failed an answer the judge passed — correctly, on the criteria as written. The criteria said nothing about being understandable. It is scored separately from task success so that writing at greater length cannot masquerade as getting more answers right.
- **A held-out set was added**, which the brief does not call for. Without it there is no way to tell a prompt that generalises from one that has absorbed the golden set — and the version that looked best on the golden set turned out to be the one that generalised worst.
- **Judged metrics are scored on a stratified subset** (24 of 70, four per category, the same cases for every version). Judging everything on a local 14B model costs hours of sustained load. The deterministic metrics cover all 70.

## Known open problems

Documented rather than smoothed over.

- **Escalation recall is ~35% against a 100% requirement.** Diagnosed, not fixed: in 64–85% of misses the model states in its own reply that the question needs a human and then does not call the tool. Four prompt formulations across three surfaces — criteria, structure, tool descriptions — all land in the same band. The remaining candidate is a schema-constrained decision gate invoked in code, which is the technique that made the judge reliable.
- **Two runs of identical v3 configuration differed** (62.3% vs 63.9% trajectory). They overlapped in time on one Ollama instance, which is the likely cause, but this contradicts the determinism established elsewhere and is unexplained.
- **Instruction blocks are less separable than assumed.** All three v4 arms share a byte-identical search-first preamble, yet search rate ranges 75.9%–93.1%. The escalation block changes behaviour on questions unrelated to escalation.
- **The held-out set is 20 cases**, so each case moves a score 5 points. Conclusions drawn from it are directional.
