"""Central configuration.

Everything tunable lives here so that Phase 3 changes are visible as a diff
on one file rather than scattered across the codebase. The eval run history
records these values alongside the scores, which is what makes a v1 vs v2
comparison attributable to a specific change.
"""

import os
from pathlib import Path

BACKEND_DIR = Path(__file__).parent
CORPUS_DIR = BACKEND_DIR / "rag" / "corpus"
CHROMA_DIR = BACKEND_DIR / "rag" / "chroma_db"
DATA_DIR = BACKEND_DIR / "data"
VISA_DATA_PATH = DATA_DIR / "mock_visa_requirements.json"
DB_PATH = DATA_DIR / "mock_hr_db.sqlite"
EVALS_DIR = BACKEND_DIR / "evals"
RUN_HISTORY_DIR = EVALS_DIR / "run_history"

COLLECTION_NAME = "mobility_policies"

# --- Models -----------------------------------------------------------------
# Local via Ollama. Provider is abstracted in agent/llm.py; swapping to a
# hosted API is a change here plus one client call, not a rewrite.
# AGENT_MODEL is overridable by environment variable so a capacity diagnostic
# (does escalation fail because the instruction is unclear, or because a 7B
# model cannot make the inference?) can be run without editing this file and
# risking the change being left in place. The runner records the model that
# actually ran, and the checkpoint refuses to resume across a model change.
AGENT_MODEL = os.environ.get("AGENT_MODEL", "qwen2.5:7b")
JUDGE_MODEL = "qwen2.5:14b"  # larger than the agent, deliberately

# Which backend serves the agent. "ollama" is the default and everything from
# v1 to v5 was measured on it.
#
# "openai" points at an OpenAI-compatible endpoint — in practice llama.cpp's
# llama-server. It exists because Ling 3.0 Tiny uses the `bailingmoe3`
# architecture, which Ollama cannot load; it needs a patched llama.cpp build
# (aetherbird/llama.cpp:bailingmoe3-support, upstream PR pending).
#
#   ~/llama.cpp-bailing/build/bin/llama-server \
#       -m ~/models/ling3-tiny/Ling-3.0-tiny-Q4_K_M.gguf \
#       --host 127.0.0.1 --port 8080 -c 16384 -ngl 99 --jinja
#
#   LLM_BACKEND=openai AGENT_MODEL=ling-3.0-tiny \
#       python -m evals.runner --version v4-enumerated
#
# The judge stays on Ollama regardless — its 0.900 kappa calibration was
# measured against qwen2.5:14b, and changing the judge would invalidate it.
LLM_BACKEND = os.environ.get("LLM_BACKEND", "ollama")

# Escalation decision gate (agent/escalation_gate.py). Off by default so every
# result up to this point stays reproducible; enable per run to A/B it:
#
#   ESCALATION_GATE=1 python -m evals.runner --version v4-enumerated
#
# Four prompt formulations and two model architectures all landed at 20-40%
# escalation recall, so this moves the decision out of free-form generation
# into a schema-constrained call whose result is acted on in code.
# ON by default as of the v8 measurement. It was off through v6-v7, when the
# gate cost more than it returned: on held-out it gained one correct handoff and
# caused two wrong ones, and dropped trajectory 65.0% -> 60.0%.
#
# Fixing the reasoning loop (GATE_REPEAT_PENALTY below) changed that verdict on
# unseen questions:
#
#                        no gate    gate    gate+repeat_penalty
#   trajectory            65.0%    60.0%          75.0%
#   escalation recall     30.0%    40.0%          80.0%
#   over-escalation        0.0%    20.0%          30.0%
#   gate call failures        -    3/20            1/20
#
# Trajectory already charges for the false escalations — every answerable
# held-out case lists escalate_to_human in forbidden_tool_calls — so +10pp is
# net of that cost.
#
# Set ESCALATION_GATE=0 to reproduce any run recorded before this default
# changed. The runner records the setting with each result and the checkpoint
# refuses to resume across a change to it.
ESCALATION_GATE_ENABLED = os.environ.get("ESCALATION_GATE", "1") not in ("", "0", "false")

# How the gate turns its response into a decision. The response carries two
# signals — a boolean `needs_human` and a `rule` enum — and they disagree often
# enough that which one is authoritative is a real choice, not a detail.
#
#   verdict  trust the boolean, ignore the enum (original behaviour)
#   named    escalate whenever a rule was named, whatever the boolean said
#   either   escalate if either fires
#
# `named` exists because of the held-out failures: on 3 of 6 misses the model
# named the applicable rule and then answered needs_human=false. Constrained
# decoding emits the boolean first, so the verdict is committed before the
# justification for it exists.
#
# Compared on the golden set via `python -m evals.gate_sweep`; see
# docs/eval_comparison.md for the numbers behind the default.
GATE_DECISION_RULE = os.environ.get("GATE_DECISION_RULE", "verdict")

# Repetition penalty on the gate call. 0 disables it.
#
# 22% of gate calls produced no JSON at all — not truncated JSON, none. The
# model looped: on esc-007 it repeated "But wait: the question is..." 24 times
# across 23 unique lines, burning all 6000 tokens without reaching the answer.
# The JSON grammar constrains the answer but not the thinking in front of it,
# so nothing terminated the loop.
#
# Six of the nine looping cases required escalation, and every one defaulted to
# no-escalation, so a third of the safety-relevant cases were scored on a
# component that never answered.
#
# 1.15 was chosen on three known-looping cases: 3/3 completed, one dropping
# from 96s to 14s. Alternatives measured and rejected — frequency_penalty=0.4
# fixed 1 of 3; a different seed fixed 2 of 3 but flipped both to the wrong
# verdict, which is evidence the loop is not simply unlucky sampling.
GATE_REPEAT_PENALTY = float(os.environ.get("GATE_REPEAT_PENALTY", "1.15"))

# Which version of the gate's classification criteria to use.
#
# "v1" is the default and is what every reported result was measured on.
#
# "v2" narrows rule 1 and adds explicit carve-outs for dependant benefits, aimed
# at the 12%/30% over-escalation rate. It is kept, unused, because abandoning it
# produced a more useful finding than adopting it would have: v2 is 26% longer
# than v1, and on the first 12 golden cases its gate call failure rate was 4/12
# against v1's 1/40.
#
# The failures are the reasoning loop returning. GATE_REPEAT_PENALTY stops the
# model circling one thought; it does not reduce how much there is to weigh.
# Two extra paragraphs of exclusions to check every question against is more
# deliberation, so the loop comes back by another route.
#
# The lesson generalises past this project: for a small reasoning model, the
# instinct to fix a wrong answer by adding a clarifying rule can make the
# component less reliable overall, and the reliability cost does not show up in
# any accuracy metric. Exclusions are better encoded structurally — a
# deterministic pre-filter, or showing the model the retrieved policy instead of
# describing what policy covers.
GATE_PROMPT_VERSION = os.environ.get("GATE_PROMPT_VERSION", "v1")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8080/v1")
OLLAMA_HOST = "http://localhost:11434"

# Determinism controls. These exist because of a measured problem, not caution.
#
# The first two v1 runs used temperature=0.1 with no seed. Re-running the SAME
# version moved trajectory 59.0% -> 52.5%, retrieval recall 72.4% -> 56.9%, and
# escalation recall 15% -> 5%, with 19 of 70 cases taking a different tool path.
# That run-to-run spread was as large as any v1/v2 difference worth reporting,
# which makes an unseeded comparison uninterpretable: a delta could be the change
# or could be the dice.
#
# temperature=0 plus a fixed seed makes a run reproducible, so a difference
# between two versions is attributable to the version. Sampling variation is a
# real property of the system, but it should be measured deliberately by varying
# the seed — not leaked into every comparison by accident.
AGENT_TEMPERATURE = 0.0
AGENT_SEED = 20260804  # vary deliberately to sample variance; fixed for A/B runs

# Context window, set EXPLICITLY. This is not a tuning knob — leaving it unset
# was a bug that invalidated an entire eval run.
#
# Ollama defaults to a 4096-token context and, when a request exceeds it,
# silently slides the window instead of erroring. The v1 prompt averaged 1,783
# tokens and fit. The v2 prompt averaged 3,050 and 12 cases exceeded 4,096 —
# so the oldest content, meaning the system prompt and the tool definitions,
# was dropped before the model ever saw it. The agent then stopped calling
# tools on 34 of 70 cases and answered from memory, inventing figures that
# appear nowhere in the corpus.
#
# The failure is silent by construction: no error, no warning, just a shorter
# prompt and worse answers. Any prompt-length change must be checked against
# this number.
AGENT_NUM_CTX = 16384  # qwen2.5:7b supports 32k; 16k leaves headroom for tool output

# --- Retrieval (Phase 3 tuning levers) --------------------------------------
CHUNK_MAX_CHARS = 1200  # a chunk larger than this gets split
CHUNK_OVERLAP_CHARS = 150  # carried between splits of the same section

# v2 (change C3): lowered 4 -> 3 on a measured sweep over the 29 retrieval-scored
# golden cases. k=3 holds recall identical to k=4/5/6 (98.3%) while lifting
# precision 60.1% -> 75.3%; k=2 starts costing hit rate. Sweep is reproducible —
# see docs/failure_analysis_v1.md §5 F8.
#
# The sweep also showed the retriever reaching 98.3% recall on well-formed
# queries while the agent scored only 72.4% in the v1 run. The shortfall is in
# how the agent phrases its search, not in k, so C1's context instruction is
# doing more of the work here than this constant is.
RETRIEVAL_TOP_K = 3

# --- Agent loop -------------------------------------------------------------
MAX_AGENT_TURNS = 6  # tool-call rounds before we stop and answer with what we have

# --- Agent version ----------------------------------------------------------
# Selects both the system prompt and the tool-description set. The eval runner
# passes a version explicitly per run; this constant is what the API serves.
#
# Defaults to the best measured version rather than the baseline. It sat at "v1"
# through development, which meant anyone starting the server to try the product
# was talking to the deliberately naive prompt — 50.8% trajectory, escalates
# nothing — and would reasonably conclude the whole thing was broken.
#
# Override to compare versions live without editing this file:
#     AGENT_VERSION=v1 uvicorn api.main:app --port 8010
#
# Known versions: v1 (naive baseline), v2 (regression — see eval_comparison.md),
# v3, v4-enumerated (best generaliser), v4-principled (best on golden set, worst
# on held-out), v4-verbatim, v5-tooldesc.
AGENT_VERSION = os.environ.get("AGENT_VERSION", "v4-enumerated")
