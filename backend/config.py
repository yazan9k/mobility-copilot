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
