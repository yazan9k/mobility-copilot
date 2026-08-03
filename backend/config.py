"""Central configuration.

Everything tunable lives here so that Phase 3 changes are visible as a diff
on one file rather than scattered across the codebase. The eval run history
records these values alongside the scores, which is what makes a v1 vs v2
comparison attributable to a specific change.
"""

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
AGENT_MODEL = "qwen2.5:7b"
JUDGE_MODEL = "qwen2.5:14b"  # larger than the agent, deliberately
OLLAMA_HOST = "http://localhost:11434"

# Low temperature: we want the agent's tool selection to be as reproducible as
# possible so that eval deltas reflect our changes, not sampling noise.
AGENT_TEMPERATURE = 0.1

# --- Retrieval (Phase 3 tuning levers) --------------------------------------
CHUNK_MAX_CHARS = 1200  # a chunk larger than this gets split
CHUNK_OVERLAP_CHARS = 150  # carried between splits of the same section
RETRIEVAL_TOP_K = 4

# --- Agent loop -------------------------------------------------------------
MAX_AGENT_TURNS = 6  # tool-call rounds before we stop and answer with what we have

# --- Agent version ----------------------------------------------------------
# Selects both the system prompt and the tool-description set, which are the
# two things Phase 3 changes together. Switched per eval run; "v1" is the
# deliberately naive baseline we are trying to beat.
AGENT_VERSION = "v1"
