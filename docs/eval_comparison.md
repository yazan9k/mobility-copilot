# Evaluation Results — v1 through v4

Every number here is **deterministic**. Trajectory, search rate, retrieval, and
escalation are exact computations over the agent's trace — no LLM scores any of
them, and reruns of the same configuration reproduce them. LLM-judged answer
quality is a separate measurement and is not in this document.

Agent: `qwen2.5:7b`, temperature 0, fixed seed, `num_ctx` 16384. Retrieval top-k 3.
Everything except the system prompt is held constant across every row.

## Results

| Version | Prompt chars | Trajectory (golden) | Trajectory (held-out) | Search rate | Recall \| searched | Escalation (golden) | Escalation (held-out) | Over-escalation | No-tool cases |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v1 | 459 | 50.8% | — | 72.4% | 88.1% | 0.0% | — | 0.0% | 3 |
| v2 | 4,163 | 31.1% | — | 20.7% | 100.0% | 40.0% | — | 4.0% | **37** |
| v3 | 1,856 | 63.9% | 55.0% | 82.8% | 91.7% | 30.0% | 30.0% | 4.0% | 11 |
| v4-enumerated | 2,540 | 63.9% | **60.0%** | 75.9% | 88.6% | 35.0% | 30.0% | 2.0% | 12 |
| v4-principled | 2,779 | **72.1%** | 40.0% | **93.1%** | 100.0% | 35.0% | 10.0% | 6.0% | 10 |
| v4-verbatim | 5,522 | 65.6% | 60.0% | **93.1%** | 96.3% | 20.0% | 20.0% | 4.0% | 11 |
| v5-tooldesc | 2,540 | 63.9% | 55.0% | 82.8% | 93.8% | 25.0% | 20.0% | 2.0% | 10 |

**Golden set:** 70 cases, the set every prompt was derived from.
**Held-out set:** 20 cases written before the v4 prompts existed, drawn from corpus
documents the failure analysis never opened, phrased without the vocabulary of the
rule they instantiate. See `backend/evals/heldout_set.yaml`.

## What each metric measures

| Metric | Question it answers |
|---|---|
| Trajectory pass rate | Did it call the right tools with the right arguments? |
| Search rate | When it should have consulted policy, did it? |
| Recall \| searched | When it *did* search, did it find the right document? |
| Escalation recall | Of questions that must go to a human, how many were handed off? |
| Over-escalation | How often was an answerable question dumped on a human? |

## Three measurement bugs found before any result was trusted

These were fixed first, because every number above depends on them.

**Non-determinism.** The first runs used temperature 0.1 with no seed. Two runs of
identical configuration differed by 6.5pp on trajectory and 15.5pp on retrieval
recall, with 19 of 70 cases taking a different tool path. Fixed with temperature 0
and a fixed seed; verified by re-running and getting byte-identical replies.

**Retrieval recall conflated two different failures.** A case where the agent never
searched scored recall 0.0, indistinguishable from a case where the retriever
returned the wrong document. On v1, 9 of 11 recall misses had retrieved *nothing*.
Unconditional recall read 63.8%; recall over cases where a search actually happened
was 88.1%. Splitting the metric showed there was never a retrieval problem to solve
— tuning chunk size or top-k against the 63.8% figure would have been chasing a
defect that did not exist.

**Silent context truncation.** `num_ctx` was unset, so Ollama defaulted to 4096 and
slid the window without warning. Fixed at 16384. Note this did *not* explain the v2
regression: 0 of the 12 over-limit cases regained tool use after the fix.

## Two hypotheses this exercise falsified

Both were mine. Both were wrong, and the runs that disproved them were worth more
than the runs that confirmed anything.

### 1. "Prompt length suppresses tool use on a 7B model"

v2 ran to 4,163 characters and 37 of 70 cases called no tools at all. v3 cut it to
1,856 and tool use recovered. I attributed the recovery to length.

`v4-verbatim` tests this directly: 5,522 characters, 33% longer than v2. It did not
collapse — 11 no-tool cases (identical to v3) and the highest search rate measured,
93.1%.

**Length was not the cause.** v2 buried its tool instruction in the middle, beneath a
long section on how to write to employees; v3 and v4 put it first and in capitals.
The lever was position and emphasis, not volume. v3 changed both at once and I
credited the wrong one.

### 2. "Principles generalise; a list cannot handle the unnamed case"

The argument was that enumerating triggers overfits, because a list only covers what
is on it, while a principle extends to situations nobody wrote down.

The held-out set says otherwise:

| Arm | Golden | Held-out | Gap |
|---|---:|---:|---:|
| v4-principled | 72.1% | 40.0% | **−32.1** |
| v4-enumerated | 63.9% | 60.0% | **−3.9** |
| v4-verbatim | 65.6% | 60.0% | −5.6 |
| v3 | 63.9% | 55.0% | −8.9 |

The principled prompt scores highest on the questions it was derived from and worst
on the questions it was not. That is the overfitting signature, in the arm predicted
to be most robust. The enumerated prompt is nearly flat.

Applying an abstract rule requires recognising that an unnamed situation instantiates
a category. A 7B model can match a list; it does not reliably make that inference.
Specific rules appear to be the correct starting point, with generalisation deferred
until there is enough evidence to know what the categories actually are.

**Caveat stated plainly:** the held-out set is 20 cases, so each case moves the score
5 points, and 40% vs 60% is four cases. The direction is consistent across two
independent metrics (trajectory and escalation) and the 32-point gap is large
relative to that granularity, but this is a small sample and should be read as such.

### 3. "The tool description is the lever, since that is what it reads when deciding to call"

Diagnosis first: across all six runs, **64–85% of missed escalations are cases where
the model states in its own reply that the question must go to an adviser, and then
does not call the tool.**

> *"I need to escalate this question to a Global Mobility adviser"* — esc-001
> *"...are discretionary and must be escalated to a Global Mobility adviser"* — esc-004
> *"Would you like to proceed with escalating this request?"* — esc-004

The judgement is correct. The call never happens. Escalation's real ceiling is ~85%,
not 35% — the gap is entirely narration-instead-of-action, including permission-seeking
that every prompt since v2 explicitly forbids.

`v5-tooldesc` tests the obvious fix: system prompt byte-identical to v4-enumerated,
only the `escalate_to_human` description rewritten to say that calling the tool *is*
the escalation and that writing about it notifies no one.

| | Trajectory G | Trajectory H | Escalation G | Escalation H | Said-but-didn't-call |
|---|---:|---:|---:|---:|---:|
| v4-enumerated | 63.9% | 60.0% | 35.0% | 30.0% | 10 + 5 |
| v5-tooldesc | 63.9% | 55.0% | 25.0% | 20.0% | 11 + 6 |

**It got worse on both sets, and the targeted pathology got more common.** A two-case
move on the golden set, so the magnitude is not load-bearing, but the direction is
consistent across both sets and both metrics.

Three prompt surfaces have now been tried — escalation criteria, prompt structure and
length, and tool descriptions. All land in the same 20–40% band. The lever is not in
prompt-space.

The remaining candidate is architectural: a schema-constrained decision gate that asks
"does this need a human?" as a separate constrained call and invokes escalation in
code, rather than relying on the model to remember to call a tool inside free-form
generation. This is the same technique that made the judge reliable at 0.900 kappa.
Not built.

## What is fixed, and what is not

**Retrieval is solved.** 88–100% recall whenever a search happens, including 100% on
the held-out set — 20 unseen questions against corpus sections the analysis never
read. The RAG pipeline is not the bottleneck and never was.

**Tool selection is good.** On v4-principled, excluding escalation cases, trajectory
is **90.2%**. The agent picks the right tool nine times out of ten.

**Search-first works.** Adding a mandatory search instruction lifted search rate from
82.8% to 93.1% and retrieval recall to 100% on the golden set.

**Escalation is not solved.** Four prompt versions, two case sets, best result 35%.
It is 76% of all remaining trajectory failures (13 of 17 on v4-principled). More
instruction about escalation made it worse, not better: `v4-verbatim` contains the
complete specification and scores lowest at 20%.

The conclusion the data supports is that this is not a prompt problem. Four
substantially different formulations — naive, exhaustive, principled, enumerated —
moved the metric between 20% and 40%. The next test is capacity: running the same
prompt on `qwen2.5:14b` to separate what the instruction cannot express from what
the model cannot infer.

## Answer quality: the process metrics improved and the answers did not

Judged by `qwen2.5:14b` (κ 0.900 against human labels) on a stratified sample of 24
cases — four per category, the same cases for both versions so the comparison is paired.

| Metric | v1 | v4-enumerated | Change |
|---|---:|---:|---:|
| Task success (mean) | 0.404 | 0.375 | −0.029 |
| Task success (pass rate) | 29.2% | 33.3% | +1 case |
| **Clarity (mean)** | 0.421 | **0.558** | **+0.137** |
| **Clarity (pass rate)** | 25.0% | **41.7%** | **+4 cases** |
| Faithfulness (mean) | 0.913 | 0.937 | +0.024 |

**Task success did not measurably improve.** The mean fell slightly while the pass rate
rose by a single case. At n=24 one case is 4.2pp, so this is noise. Meanwhile the
deterministic metrics moved a long way: trajectory 50.8% → 63.9%, escalation 0% → 35%,
search rate 72.4% → 93.1%.

Better tool use did not produce better answers. Tracking trajectory alone would have
reported a 13-point improvement and called the project a success.

**Clarity is the one genuine gain** — 25.0% → 41.7%, four cases, and attributable: v2
through v4 all carry explicit writing guidance, and this metric exists to measure
exactly that. It was added after a human labeller failed an answer the judge passed on
criteria that said nothing about being understandable.

### The aggregate was hiding a redistribution

| Category (n=4 each) | v1 | v4-enumerated | Δ |
|---|---:|---:|---:|
| ambiguous / out-of-scope | 0.07 | 0.33 | **+0.26** |
| visa | 0.38 | 0.47 | +0.09 |
| document_request | 0.62 | 0.62 | 0.00 |
| escalation_needed | 0.25 | 0.15 | −0.10 |
| policy | 0.62 | 0.45 | **−0.17** |
| multi_turn | 0.47 | 0.23 | **−0.24** |

v4 improved most on the category it was written for and regressed on two it was not.
Policy questions — the most common real use — got *worse* despite the agent searching
more often. Escalation quality fell even though escalation recall rose, which fits the
narration-vs-action diagnosis: calling the tool more often does not help if the
surrounding answer is weaker.

Each case is 0.25 of a category mean here, so these are one- and two-case movements and
should be read as directional. The pattern is still worth recording: a flat headline
concealed offsetting movements in both directions, and the fixes went where the process
metrics pointed rather than where answer quality was actually lost.

### Coverage limit

Only **two of seven** versions are judged — v1 and v4-enumerated. Judging a version
costs ~35 minutes of sustained load on a local 14b model. v2, v3, v4-principled,
v4-verbatim and v5-tooldesc have deterministic scores only. In particular there is no
answer-quality reading on v4-principled, which would sharpen the overfitting finding.

## Open items

**Two runs of identical v3 configuration differed.** 62.3% and 63.9% trajectory. They
overlapped in time and contended for the same Ollama instance, which is the likely
cause, but this contradicts the determinism established above and is unexplained.
The v3 figures in this document are from the second run.

**Instruction blocks are not as separable as assumed.** All three v4 arms share a
byte-identical search-first preamble, yet search rate ranges 75.9% to 93.1%. The
escalation block is changing behaviour on questions that have nothing to do with
escalation, so "one variable at a time" holds for what was edited, not for what was
affected.

**Answer quality is unmeasured.** Everything here is process, not correctness. Task
success, clarity, and faithfulness are judged by `qwen2.5:14b`, calibrated at 0.900
Cohen's kappa against human labels, and have not yet been run against v1–v4.
