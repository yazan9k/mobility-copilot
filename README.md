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

# 1. run the agent over the golden set -> deterministic metrics
../.venv/bin/python -m evals.runner --version v1        # ~15 min
../.venv/bin/python -m evals.runner --version v2

# 2. add the judged metrics (task success, clarity, faithfulness)
../.venv/bin/python -m evals.judge_scoring --version v1  # slower; 14B model
../.venv/bin/python -m evals.judge_scoring --version v2

# 3. compare, with case-level regressions
../.venv/bin/python -m evals.compare --base v1 --new v2 \
    --markdown ../docs/eval_comparison.md

# 4. check the judge itself
../.venv/bin/python -m evals.calibration --label rubric-v2

# 5. pytest gate against the PRD targets (reads completed runs; no model calls)
../.venv/bin/python -m pytest evals/test_agent_eval.py -v
```

Agent runs and judge scoring are split on purpose: the agent pass is the expensive, non-deterministic half, so changing a metric definition or re-judging with a revised rubric never means re-running 70 agent traces.

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
    tools.py              6 tools; descriptions versioned as a Phase 3 lever
    prompts.py            v1 (deliberately naive baseline) and v2
    core.py               the agent loop, ~110 lines
  data/                   fabricated visa dataset + mock HR SQLite
  api/main.py             POST /chat, GET /health
  evals/
    golden_set.yaml       70 labelled cases across 6 categories
    calibration_set.yaml  20 hand-labelled cases for judging the judge
    metrics.py            deterministic scoring, no LLM
    ollama_judge.py       DeepEval judge over a local model
    runner.py             agent pass -> deterministic metrics
    judge_scoring.py      adds task success, clarity, faithfulness
    calibration.py        judge/human agreement + Cohen's kappa
    compare.py            v1 vs v2, including case-level regressions
    test_agent_eval.py    pytest gate against the PRD targets
    run_history/          versioned results
docs/
  prd.md                  scope and success metrics
  personas.md             5 synthetic personas (eval design scaffolding)
  corpus_outline.md       the shared fabricated facts the corpus agrees on
  failure_analysis_v1.md  8 numbered findings -> 3 changes
  eval_comparison.md      v1 vs v2, per metric
  case_study.md           the write-up
```

---

## Deviations from the original brief

Recorded here rather than buried, since a reviewer should be able to see what changed and why.

- **Local models instead of the Anthropic API.** A zero-spend constraint. The cost is real: weaker tool-calling, and a judge that needed calibrating before its scores meant anything. `agent/llm.py` keeps the swap to one function.
- **Frontend and deployment deferred.** The brief's own MVP cut line puts the agent, the golden set, and one documented iteration cycle above the UI. The eval dashboard lives in `docs/eval_comparison.md` as tables rather than as a React page.
- **A `Clarity` metric was added** after a human labeller failed an answer the judge passed — correctly, on the criteria as written. The criteria said nothing about being understandable. It is scored separately from task success so that writing at greater length cannot masquerade as getting more answers right.
