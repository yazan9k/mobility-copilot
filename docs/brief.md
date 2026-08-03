# Global Mobility Copilot — Project Brief for Claude Code

## Purpose of this document

This is a project brief meant to be pasted directly into Claude Code as the first message, or saved as `CLAUDE.md` in the repo root. It defines what to build, in what order, and what "done" looks like at each stage. Goal: a portfolio project demonstrating AI agent design, RAG, tool use, and an evaluation system, targeting enterprise AI product roles (agent building, RAG, evaluation-driven iteration).

## Product framing (read first, this is not just a coding task)

The product is a "Global Mobility Copilot": an internal assistant that helps employees at a fictional company navigate international relocation. It answers visa and policy questions, generates document checklists, tracks relocation status, and escalates to a human HR contact when it's out of its depth.

The point of the project is not the chatbot. The point is the evaluation system built around it, and the documented before/after improvement that evaluation drove. Build accordingly: every phase should produce something measurable, not just something that runs.

## Tech stack

1. Backend: Python, FastAPI. Agent orchestration in plain Python (function calling against the Anthropic API), not a heavy framework, so the architecture stays legible.
2. Vector store: Chroma, local, embedded. No external infra needed.
3. Eval framework: DeepEval (pytest-based). Golden dataset stored as JSON/YAML fixtures.
4. Frontend: Next.js + TypeScript. Chat interface plus a separate eval dashboard page.
5. Persistence: SQLite for mock HR tickets, relocation status records, and eval run history.
6. Deployment target: Fly.io or Render for the API, Vercel for the frontend. Both have generous free tiers.

## Repository structure

```
mobility-copilot/
  backend/
    agent/
      core.py           # agent loop: plan, call tools, respond
      tools.py           # tool implementations
      prompts.py          # system prompt(s), versioned (v1, v2, v3)
      subagents/          # stretch goal: router + specialist agents
    rag/
      ingest.py           # chunk + embed policy docs into Chroma
      retrieve.py
      corpus/              # the 30-50 mock policy markdown files
    data/
      mock_visa_requirements.json
      mock_hr_db.sqlite
    api/
      main.py             # FastAPI routes
    evals/
      golden_set.yaml     # the 60-100 test cases
      test_agent_eval.py  # DeepEval test suite
      run_history/        # JSON logs of each eval run, versioned
  frontend/
    app/
      chat/                # chat UI
      dashboard/           # eval metrics over time
  docs/
    prd.md
    personas.md
    case_study.md
    architecture_diagram.png
```

## Setup: repository and version control

Do this before Phase 0, every time you start this project in a new session:

1. Check whether a GitHub MCP server is currently connected and available. If it is not connected, connect it before doing anything else, and confirm the connection is working before proceeding.
2. Once connected, create a new private GitHub repository named `mobility-copilot` (or ask the user to confirm the name if they have a preference).
3. Initialize the local repo, add a `.gitignore` appropriate for a Python/Node monorepo (node_modules, __pycache__, .env, *.sqlite, chroma persistence directories), commit this brief itself as `docs/brief.md`, and push the initial commit to the new private repo.
4. From this point on, commit and push at the end of every phase below, not just at the end of the project. Each commit message should name the phase and what changed (e.g. "Phase 1: agent core, RAG pipeline, and tool implementations"). This gives you a real commit history showing incremental, evaluation-driven progress, which is itself part of the portfolio artifact — a reviewer can read the git log and see the project being built the way the brief describes, not dropped in as one giant commit.
5. Never commit secrets (API keys, .env files). Use `.env.example` with placeholder values instead.

## Build order

### Phase 0: Product foundation (do this before any code)

1. Write `docs/prd.md`: problem statement, target user, MVP scope, out-of-scope, success metrics (should include at least one eval-derived metric, e.g. "task success rate over 80% on golden set").
2. Write `docs/personas.md`: 3-5 synthetic employee personas with distinct relocation scenarios (e.g. first-time expat with dependents, short-term assignee, visa renewal case). Label these explicitly as synthetic/illustrative, not real user research.
3. Draft the mock policy corpus outline: list the 8-10 policy documents needed (relocation allowance, visa sponsorship, housing stipend, shipping/moving allowance, tax equalization overview, timeline/milestones, document requirements by country tier, escalation criteria). Each doc should be 300-800 words, self-authored.

### Phase 1: Agent core, RAG, and tools

1. Write the mock policy corpus (Phase 0 outline) as markdown files in `rag/corpus/`.
2. Build the ingestion pipeline: chunk, embed, store in Chroma.
3. Implement tools:
   - `search_policy_kb(query: str)` — RAG retrieval, returns top-k chunks with source doc
   - `lookup_visa_requirements(from_country, to_country, visa_type)` — structured lookup against `mock_visa_requirements.json` (fabricate realistic but clearly synthetic data for 15-20 country pairs, do not scrape real government sources)
   - `generate_document_checklist(destination, employee_type)` — deterministic, returns a task list
   - `get_relocation_timeline(destination)` — returns milestone plan
   - `create_hr_ticket(category, summary, urgency)` — writes to SQLite mock HR table
   - `escalate_to_human(reason)` — explicit handoff path, should trigger when the agent's confidence is low or the query is out of scope
4. Implement the agent loop in `agent/core.py`: system prompt (v1, kept intentionally simple/naive as the baseline), tool-calling loop, response formatting.
5. Expose via FastAPI: a `/chat` endpoint that takes a message and conversation history, returns the agent's response plus the tool calls it made (needed later for trajectory-level eval).
6. Definition of done for this phase: you can hit `/chat` with a query like "I'm relocating from Dubai to Singapore, what visa do I need and what does the company cover?" and get a coherent, tool-grounded answer.

### Phase 2: Baseline evaluation

1. Write `evals/golden_set.yaml`: 60-100 test cases. Each case needs: the user query, category (visa question / policy question / document request / escalation-needed / ambiguous or out-of-scope / multi-turn), expected tool calls, and expected answer criteria (key facts that must appear, or a rubric for a judge to check against).
2. Build `evals/test_agent_eval.py` using DeepEval. Implement metrics at three levels:
   - End-to-end: did the agent's final answer satisfy the user's actual need (LLM-as-judge, custom G-Eval rubric)
   - Trajectory-level: did it call the correct tool(s) with correct parameters (deterministic check against expected_tool_calls)
   - Component-level: RAG retrieval precision/recall against a labeled set of "which policy doc(s) should this query retrieve"
   - Also track: hallucination/faithfulness (DeepEval's built-in faithfulness metric), latency, cost per query
3. Run the full suite against the v1 (naive) agent. Log results to `evals/run_history/v1.json`.
4. Do a real failure analysis: read through the failures, categorize them (wrong tool chosen, hallucinated visa detail, missed escalation, poor retrieval, etc). Write this up in a short markdown note. This failure analysis is what justifies every change you make in Phase 3, don't skip it.

### Phase 3: Iterate using eval findings

1. Based on the Phase 2 failure analysis, make specific, justified changes: e.g. rewrite the system prompt (v2), improve tool descriptions so the model picks the right one more reliably, tune retrieval (top-k, chunk size, add reranking if retrieval precision was the problem), add explicit escalation triggers.
2. Rerun the full eval suite. Log to `evals/run_history/v2.json`.
3. Compare v1 vs v2 numerically, per metric. If time allows, do one more iteration cycle (v3) targeting whatever's still weakest.
4. This before/after comparison, with real numbers, is the single most important deliverable of the whole project. Do not skip or rush this phase to get to the frontend sooner.

### Phase 4: Frontend

1. Chat interface: simple, functional, employee-facing. Persona switcher (pick one of the personas from Phase 0) is a nice touch, not required.
2. Eval dashboard: a page showing metric scores per version (v1, v2, v3) as a simple chart, plus a drill-down into a couple of specific failure-to-fix examples (before answer, after answer, what changed).
3. Deploy both, get a live URL.

### Phase 5: Case study write-up

1. Write `docs/case_study.md`: problem, approach, architecture (include a simple diagram), eval methodology, before/after results with numbers, what you'd build next if you kept going (e.g. multi-agent split, production vector DB, human-in-the-loop eval labeling).
2. Record a 2-3 minute demo walkthrough (screen recording is fine).
3. Optional stretch: rebuild a scoped-down version of the same agent in Coze (coze.com) using its knowledge base and workflow features, to demonstrate hands-on familiarity with that specific tool.

## Definition of done (MVP cut line)

If time runs short, the non-negotiable core is: a working tool-using agent (Phase 1), a real golden eval set with at least end-to-end and trajectory-level scoring (Phase 2), and one documented iteration cycle showing scores improve (Phase 3, steps 1-3). The frontend, dashboard, and Coze stretch goal can be cut or simplified without losing the thing that makes this project different from a standard chatbot demo.

## Constraints and notes for Claude Code

1. All policy documents, visa data, and personas must be clearly fabricated/synthetic. Do not scrape or reproduce real company policies or real government visa pages.
2. Keep the agent architecture legible over clever. A reviewer (technical or not) should be able to look at `agent/core.py` and understand the loop in under a minute.
3. Every phase should end with something that can be demoed or measured, not just code that runs.
4. Favor small, well-named functions over abstraction layers. This is a portfolio project meant to be read by a human, not a production system meant to scale.
5. Commit and push to the private GitHub repo at the end of every phase, per the Setup section above. Don't wait until the end of the project to start version control.
