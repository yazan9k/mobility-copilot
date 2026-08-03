# Global Mobility Copilot — Product Requirements

> **Fictional context.** Meridian Systems, the company described here, does not exist. All policies, allowances, country tiers, and visa data in this project are fabricated for demonstration purposes.

## 1. Problem statement

At Meridian Systems (~4,000 employees, 14 countries), roughly 180 employees relocate internationally each year. Each relocation generates an average of **11 questions to the Global Mobility team**, and internal survey data — also fabricated for this exercise — suggests around 70% of those questions are already answered somewhere in the policy library.

The failure is not missing information. It's that the information is:

- **Fragmented.** Ten separate policy documents, plus a visa requirements matrix maintained in a spreadsheet.
- **Conditional.** Nearly every answer depends on destination country tier, employee band, assignment type, and whether dependents are relocating. A single document rarely answers a single employee's question.
- **Time-sensitive.** Employees ask at the point of anxiety — often outside the mobility team's working hours, and often across timezones.

The result is a Global Mobility team of four spending the majority of its capacity re-answering resolvable questions, which delays the cases that genuinely need human judgment: tax equalization edge cases, visa refusals, and compassionate exceptions.

## 2. Target user

**Primary:** A Meridian employee who has just been offered or assigned an international move, from the moment of offer through their first month in-country. They are not policy experts, they do not know the vocabulary ("tier 2 destination", "tax equalization", "shipping allowance band"), and they do not know which of their questions are answerable by policy versus which need a human.

**Secondary:** The Global Mobility team itself, who need low-value queries deflected *and* need genuine escalations routed to them with context attached rather than dropped.

**Explicit non-user:** Recruiters, hiring managers, and finance. Their questions overlap superficially but carry different authority and confidentiality requirements.

## 3. What the product does

An internal chat assistant that:

1. Answers visa and policy questions grounded in the Meridian policy corpus, citing which document each claim came from.
2. Looks up structured visa requirements for a given origin/destination/visa-type combination.
3. Generates a document checklist tailored to destination and employee type.
4. Returns a relocation timeline with milestones for a destination.
5. Files an HR ticket when the employee needs something actioned.
6. **Escalates to a human** when it is out of scope, low-confidence, or the query touches a category that policy requires a human to handle.

Point 6 is a first-class feature, not a fallback. A mobility assistant that confidently invents a visa requirement is worse than no assistant.

## 4. MVP scope

| In scope | Notes |
|---|---|
| Single-turn and short multi-turn Q&A | Conversation history passed to the agent; no long-term memory |
| RAG over 10 fabricated policy documents | Local embeddings, local vector store |
| 6 tools, including explicit escalation | Per §3 |
| `POST /chat` API returning reply **and tool-call trajectory** | The trajectory is required for evaluation, not decoration |
| Evaluation system: golden set, three metric levels, versioned run history | The actual deliverable |
| Documented v1 → v2 iteration driven by failure analysis | The actual deliverable |

## 5. Out of scope

- **Authentication, authorization, per-employee data access.** The agent answers policy questions; it does not read anyone's HR record. Relocation status lookups run against a mock table.
- **Real visa or immigration data.** All visa data is fabricated. The product is explicitly not immigration advice, and the system prompt must say so.
- **Write access to real systems.** HR tickets are written to a local SQLite mock.
- **Multi-language.** English only.
- **Production concerns** — rate limiting, PII redaction, audit logging, SSO. Named here because a reviewer should see they were considered and deliberately deferred, not overlooked.
- **Frontend and deployment.** Deferred; see `case_study.md` for rationale.

## 6. Success metrics

The first three are eval-derived and measured on the golden set. They are the ones that matter.

| # | Metric | Target | How measured |
|---|---|---|---|
| M1 | **Task success rate** — final answer satisfies the user's actual need | **> 80%** | LLM-as-judge (G-Eval) against a per-case rubric, with judge/human agreement reported alongside |
| M2 | **Tool trajectory accuracy** — correct tool(s) called with correct parameters | **> 85%** | Deterministic comparison against `expected_tool_calls`. No LLM involved |
| M3 | **Retrieval recall@k** — the policy doc(s) that should have been retrieved were retrieved | **> 90%** | Deterministic comparison against labelled `expected_source_docs` |
| M4 | **Faithfulness** — no claims unsupported by retrieved context | **> 0.85** | DeepEval faithfulness metric |
| M5 | **Escalation recall** — cases that require a human actually escalate | **100% on the escalation subset** | Deterministic: did `escalate_to_human` fire |
| M6 | Median latency per query | Tracked, not gated | Measured per case in the eval runner |

**M5 is treated as a safety metric rather than a quality metric.** A missed escalation on an immigration question is a materially worse outcome than a mediocre answer, so it is reported separately and never averaged into an overall score.

### The metric that defines the project

> **A documented improvement between v1 and v2 on M1–M5, where every change made in v2 traces to a numbered finding in the v1 failure analysis.**

A high v1 score would actually be a worse outcome for this project than a low one. The deliverable is the iteration loop, not the chatbot.

## 7. Key design decisions

| Decision | Rationale |
|---|---|
| Hand-written agent loop rather than a framework | The brief requires a reviewer to understand `agent/core.py` in under a minute. Frameworks obscure exactly the control flow being evaluated |
| Local models via Ollama instead of a hosted API | Zero-cost constraint. Provider sits behind one interface (`agent/llm.py`) so the swap is config, not rewrite. Costs are documented honestly in the case study |
| Trajectory and retrieval metrics are deterministic | With a small local judge model, LLM-scored metrics are noisy. Two of three metric levels need no LLM and carry the weight |
| Judge model is larger than the agent model, with schema-constrained output | Reduces the two dominant small-model judging failures: bad calibration and unparseable output |
| Judge is calibrated against 20 human-labelled cases | Without this, an LLM-judged score is an unverified number. Agreement is reported, not assumed |
| v1 system prompt is deliberately naive | It is the baseline. Tuning it before measuring would destroy the before/after story |

## 8. Risks

| Risk | Mitigation |
|---|---|
| Small model tool-calling is unreliable | This is expected and is itself the primary Phase 3 finding. Tool descriptions are isolated as a tuning lever |
| Judge scores are noisy | Calibration set quantifies it. If agreement is poor, deterministic metrics become the headline and G-Eval is reported as directional |
| Overfitting v2 to the golden set | Acknowledged limitation. A held-out split is listed as next-step work rather than claimed as done |
| Synthetic data flatters the system | Corpus and golden set are written to include genuine ambiguity, conflicting-looking policies, and out-of-scope queries |
