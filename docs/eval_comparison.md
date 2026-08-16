# Evaluation Results

Every number here is **deterministic**. Trajectory, search rate, retrieval, and
escalation are exact computations over the agent's trace — no LLM scores any of
them, and reruns of the same configuration reproduce them. LLM-judged answer
quality is a separate measurement and is not in this document.

## Headline: baseline to final

The number that counts is the held-out one — 20 questions written before the prompts
that answer them, never used to derive any rule or fix.

| Held-out (20 unseen) | v1 baseline | final | Δ |
|---|---:|---:|---:|
| **Trajectory** | 55.0% | **75.0%** | **+20.0** |
| **Escalation recall** | 20.0% | **80.0%** | **+60.0** |
| Search rate | 90.0% | **100.0%** | +10.0 |
| Retrieval recall | 90.0% | **100.0%** | +10.0 |
| Retrieval precision | 48.3% | 58.3% | +10.0 |
| Over-escalation | 0.0% | 30.0% | **+30.0** |
| Median latency | 17,139ms | 24,151ms | +41% |

| Golden (70, the derivation set) | v1 baseline | final | Δ |
|---|---:|---:|---:|
| Trajectory | 50.8% | **88.5%** | +37.7 |
| Escalation recall | 0.0% | **85.0%** | +85.0 |
| Search rate | 72.4% | 96.5% | +24.1 |
| Retrieval recall | 63.8% | 96.5% | +32.8 |
| Over-escalation | 0.0% | 12.0% | +12.0 |

Final configuration: `Ling-3.0-tiny` via llama-server, prompt `v4-enumerated`,
escalation gate ON with `repeat_penalty` 1.15, temperature 0, fixed seed.

**Where the gain came from**, in order of size:

| Change | Held-out effect |
|---|---|
| Escalation gate + `repeat_penalty` | **The bulk of it.** Escalation 30% → 80% |
| `qwen2.5:7b` → Ling 3.0 Tiny | Retrieval and search rate to ceiling |
| Prompt v1 → v4-enumerated | ~+5pp. Most apparent gains were overfitting |
| Reasoning-parser fix | Recovered 36% of destroyed answers, **0pp on tool metrics** |

Four rounds of prompt engineering moved held-out trajectory about 5 points. Fixing one
component's reliability moved it 10, and fixing the decision mechanism moved escalation
60. That ordering is the main finding of the project.

**The open problem is over-escalation**, 0% → 30%. Trajectory already charges for it —
every answerable held-out case lists `escalate_to_human` in `forbidden_tool_calls`, so
the +20pp is net of those false alarms — but three in ten ordinary questions now reach a
human unnecessarily, and the corpus itself calls that a failure rather than a safe
default. See "Fixing over-escalation made it worse" below.

## The prompt-only phase (qwen2.5:7b)

Agent: `qwen2.5:7b`, temperature 0, fixed seed, `num_ctx` 16384. Retrieval top-k 3.
Everything except the system prompt is held constant across every row.

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

## The improvement is roughly a third of what the golden set reports

The v1 held-out baseline was missing for most of this project, which meant the held-out
column had no comparison and "60.0% on unseen questions" could not be called an
improvement over anything. Running it changed the headline.

| | v1 golden | v1 held-out | v4-enum golden | v4-enum held-out |
|---|---:|---:|---:|---:|
| Trajectory | 50.8% | 55.0% | 63.9% | 60.0% |
| Escalation | 0.0% | 20.0% | 35.0% | 30.0% |
| Search rate | 72.4% | 90.0% | 75.9% | 80.0% |

| Improvement v1 → v4-enumerated | Golden | Held-out |
|---|---:|---:|
| Trajectory | **+13.1pp** | **+5.0pp** |
| Escalation | +35.0pp | **+10.0pp** |

Escalation reads as a 35-point gain on the set the prompts were derived from and a
10-point gain on questions they have never seen. Trajectory reads as +13.1 and is +5.0.

Note also that v1 scores *higher* on held-out than on golden (55.0% vs 50.8%), which
suggests the held-out cases are somewhat easier rather than harder. That cuts against
the v4 result too: 60.0% on an easier set is a weaker showing than it first appears.

The honest summary is that most of the measured improvement is concentrated on the
cases that shaped the prompts. Some of it generalises. Considerably less than the
headline suggests.

## The bug that invalidated every Ling number

> **Status: every figure in the two sections below is being remeasured.** They are left
> in place because the correction is the point, not because they are right.

Ling 3.0 is a reasoning model: it thinks, then answers. `llama-server` splits that into
two response fields — `reasoning_content` and `content` — and `agent/llm.py` read
`content`. **The split is unreliable for this model.** The chat template opens the
thinking block in the prompt, so any generation that never emits a closing `</think>`
is classified as thinking *in its entirety*, and `content` comes back empty.

The provider seam then returned the empty string, and the agent loop, seeing neither
text nor a tool call, concluded the agent had finished and stopped.

| Run | Model | Empty replies |
|---|---|---:|
| v1, v3, v4-enumerated | qwen2.5:7b | **0 / 70** |
| v4-enumerated golden | Ling | **25 / 70 (36%)** |
| v4-enumerated held-out | Ling | **7 / 20 (35%)** |
| v6 + gate golden | Ling | **17 / 70** |

The model was working throughout — those cases carry 47-512 completion tokens each.
On `doc-001` it produced a complete, correctly formatted document checklist and the run
recorded an empty reply, because the whole answer had been filed as thinking.

The bug has two shapes, and the second is worse than a lost answer:

1. **The answer is stranded.** Scored as a total failure on every judged metric.
2. **A tool call is stranded**, left as raw `<tool_call>…</tool_call>` markup in the
   discarded field. The loop sees no tool call and terminates — so the agent was being
   killed *mid-investigation*, before it ever reached the point of deciding to escalate.

**Fix:** request `reasoning_format: "none"`, take the raw generation, and split it in
`agent/llm.py` (`_split_thinking`, `_recover_tool_calls`). This removes llama-server's
classifier from the measurement path. Verified on the 8 worst cases: 8/8 went from an
empty reply to a full answer.

### What it does and does not invalidate

* **The v1 → v4 prompt comparison is unaffected.** All of it ran on qwen, which had zero
  empty replies across every run.
* **Every judged Ling figure is unsafe**, and the gate runs are unsafe because 17 of
  their 70 replies were empty.
* **The deterministic tool metrics were not affected.** Trajectory, escalation and
  retrieval read the tool trace, which the bug never touched. Re-running both sets with
  the fix moved trajectory by 0.0pp (68.8% golden, 65.0% held-out), escalation by 0.0pp,
  and flipped zero cases — while recovering all 32 empty replies. The bug destroyed
  answer text and nothing else.
* **The gate's 20-25% failure rate was NOT this bug.** That was proposed here and is
  false: with the chat-path fix in place and the gate reading its own path, a smoke test
  still failed 1 of 5 calls, after 68 seconds. Constrained decoding forces the JSON into
  `content`, so the gate was never exposed to the misfiling. Its failure rate is a
  separate, still-open problem — see "Open items".

### Why no metric caught it

Nothing in the suite looked at whether the agent said anything. Trajectory, retrieval and
escalation all kept returning plausible mid-60s numbers while a third of the answers were
being dropped, so the run read as *disappointing* rather than *broken* — and was acted on.
`empty_replies` is now a tracked operational metric and `test_no_empty_replies` fails the
suite at a count of one, on the grounds that an empty reply is never a model result.

## Changing the model beat four rounds of prompt engineering

> ⚠️ **The table in this section was produced under the parsing bug above and is being
> remeasured.** The retrieval and search-rate gains are load-bearing for the argument and
> are the ones most likely to survive; the trajectory deltas are the ones most likely to
> move, since a third of the Ling run terminated early.

`Ling-3.0-tiny` (inclusionAI, released 6 Aug 2026) is a mixture-of-experts model with
1.3B active parameters of 7.9B total. Ollama cannot load it — the `bailingmoe3`
architecture needs a patched llama.cpp build — so it runs behind `llama-server` through
a second backend in `agent/llm.py`. Same prompt, same tool descriptions, same cases,
same temperature and seed. **The model is the only variable.**

| Metric | qwen2.5:7b | Ling 3.0 Tiny | Δ |
|---|---:|---:|---:|
| Trajectory golden | 63.9% | **68.8%** | +4.9 |
| Trajectory held-out | 60.0% | **65.0%** | +5.0 |
| Search rate | 75.9% | **96.5%** | **+20.7** |
| Retrieval recall | 67.2% | **96.5%** | **+29.3** |
| Recall \| searched | 88.6% | **100%** | +11.4 |
| Over-escalation | 2.0% | **0.0%** | −2.0 |
| Cases calling no tools | 12 | **3** | −9 |
| Median latency | 7,430ms | **5,156ms** | −31% |
| Escalation golden | **35.0%** | 25.0% | −10.0 |
| Escalation held-out | 30.0% | 30.0% | 0.0 |

Better on eight of nine metrics, twice as fast, on a third of the active parameters.

The comparison worth drawing is against the prompt work:

| Source of improvement | Golden | Held-out |
|---|---:|---:|
| Four prompt iterations (v1 → v4-enumerated) | +13.1pp | +5.0pp |
| One model swap (same prompt) | +4.9pp | **+5.0pp** |

**On unseen questions, swapping the model matched four rounds of prompt engineering.**
The prompt work took days; the swap took an afternoon, most of it compiling llama.cpp.

### Escalation is not a model-capacity problem

Escalation stayed in the same 25-30% band, which rules out the capacity explanation the
abandoned 14b diagnostic was meant to test.

For qwen the failure mode is clear and holds up:

| | escalated | said it, did not call |
|---|---|---|
| qwen2.5:7b golden | 7/20 | **10/13** — recognises it, does not act |

That is an execution failure. A single prompt asking one model to answer the question,
choose among six tools, and remember a safety rule fails at whichever of those it is
weakest on — which is the argument for taking the decision out of the prompt entirely.

> **Retracted.** This section previously carried two more rows — Ling golden 7/15 and
> Ling held-out **1/7**, read as "Ling does not recognise escalation at all, it answers
> the questions" — and concluded that the two architectures fail for opposite reasons.
> That conclusion was an artefact of a bug. 6 of those 7 held-out replies were the empty
> string: llama-server had classified the model's entire output as reasoning and
> `agent/llm.py` read only the (empty) content field, so the analysis counted a parsing
> fault as a model property. On the single miss where text existed, the reply *did*
> raise going to an adviser — the opposite of the claim. See "The bug that invalidated
> every Ling number" below. Ling's escalation behaviour is being remeasured.

## The escalation gate: the mechanism works, the implementation does not

Four prompt formulations across two architectures moved escalation by essentially
nothing. The gate takes the decision out of free-form generation entirely: a separate
schema-constrained call answers "does this need a human?", and the tool is invoked in
code. See `agent/escalation_gate.py`. Enabled with `ESCALATION_GATE=1`, off by default.

| Metric | Ling | +gate | Δ | Ling held-out | +gate held-out | Δ |
|---|---:|---:|---:|---:|---:|---:|
| Trajectory | 68.8% | **80.3%** | **+11.5** | 65.0% | 65.0% | **0.0** |
| Escalation | 25.0% | **60.0%** | **+35.0** | 30.0% | **40.0%** | **+10.0** |
| Over-escalation | 0.0% | 8.0% | +8.0 | 0.0% | 10.0% | +10.0 |
| Median latency | 5,156ms | 19,009ms | 3.7x | — | — | — |
| **Gate failures** | — | **14/70 (20%)** | — | — | **5/20 (25%)** | — |

**+35pp escalation on golden, +10pp on held-out** — the same roughly one-third ratio
between derived and unseen cases that this project has now measured three times.

Trajectory does not move at all on held-out: the escalation gain is cancelled by 10%
over-escalation, leaving net tool choice flat on unseen questions.

Where the gate does fire it is reasonably precise — 8 of 12 correct on golden, 4 of 5 on
held-out. It is not spraying escalations. The damage is in the 20-25% of calls that
never return an answer, each defaulting silently to no-escalation.

### Not shipped as default, for three reasons — SUPERSEDED

1. **20-25% failure rate**, and it fails toward under-escalation on a safety metric.
2. **Over-escalation 0% to 10%** on held-out, from a clean baseline.
3. **3.7x latency**, 5.2s to 19s.

> **This verdict no longer holds.** Reason 1 was a reasoning loop, fixed below, and it
> was suppressing the gate's real performance. With it fixed the gate ships as default:
> held-out trajectory 65.0% → 75.0% and escalation 30.0% → 80.0%. Reasons 2 and 3
> remain true and are the cost of that.

### The remaining defect, named precisely

Ling emits reasoning tokens before the JSON. On harder questions those tokens exhaust
the budget and the response comes back empty. Every attempted fix traded one failure for
another:

| Attempt | Result |
|---|---|
| `max_tokens` 400 | 0/5 escalation recall — reasoning starved, degenerate valid output |
| `maxLength` on `reason` | Fixed the string running away, not the thinking before it |
| Thinking disabled | 5/5 recall, **4/5 false alarms** — labels document checklists HIGH_CONSEQUENCE |
| `max_tokens` 6000 | Failures 19% to 20%, one call at 111s |

The fix is not a bigger budget. It is either a model whose reasoning is bounded, or a
gate that streams and stops at the first complete JSON object rather than waiting for
the generation to end.

**The finding stands regardless:** the only intervention that moved escalation at all
did so while a fifth of its calls were crashing. That is evidence about the mechanism,
not about the prototype.

### The failures were a reasoning loop, and one parameter fixed them

The diagnosis above — "reasoning tokens exhaust the budget" — was right about the symptom
and wrong about the cause, which is why every fix derived from it failed. Capturing the
raw output of the failing calls settled it. They are not truncated JSON. They contain no
JSON at all: ~28,000 characters of the model arguing with itself.

On `esc-007` it repeated **"But wait: the question is…" 24 times**, cycling 23 unique
lines across 69, until the budget ran out. The JSON grammar constrains the *answer* and
not the thinking in front of it, so nothing terminated the loop.

That explains the whole table above. A bigger budget buys a longer loop. Starving the
budget truncates mid-loop. Disabling thinking removes the loop and the judgement with it.

A repetition penalty ends the loop while leaving the reasoning intact:

| Golden, 40 cases | baseline | `repeat_penalty=1.15` |
|---|---:|---:|
| Gate call failures | 22% (9/40) | **2% (1/40)** |
| Escalation recall | 35% | **70%** |
| False alarm rate | 15% | **10%** |

Recall doubled *and* false alarms fell. Not a threshold moved — a fault removed. The nine
looping calls were never wrong answers; they were absent answers scored as
"do not escalate", and six of the nine were escalation-required cases.

Alternatives, measured on three known-looping cases: `frequency_penalty=0.4` fixed 1 of 3;
a different seed fixed 2 of 3 but **flipped both to the wrong verdict**, which is evidence
the loop is a property of the reasoning path rather than unlucky sampling.

Held-out confirms the gain survives, at roughly half the size — the ratio this project has
now measured four times:

| Held-out, 20 cases | no gate | gate | gate + `repeat_penalty` |
|---|---:|---:|---:|
| Correct handoffs caught | 3/10 | 4/10 | **6/10** |
| Answerable wrongly escalated | 0/10 | 2/10 | 2/10 |
| Gate call failures | — | 25% | **5%** |

The net accounting flips with it: the gate previously gained one correct handoff and cost
two false ones, which is why it was not shipped. It now gains three and costs two.

### What did not change

`rule-named` — escalating whenever the model names a rule, instead of trusting its
boolean — was tested in the same sweeps and stays rejected. It reaches 90% recall on
golden and 80% on held-out at **65-80% false alarms**, escalating 8 of 10 ordinary
held-out questions. `either` scores identically to `rule-named`, which shows the boolean
carries no signal the enum does not already have.

The remaining misses are not a reliability problem. On four of them the rule text names
the case almost verbatim — Rule 4 says *"a refused, expired, or at-risk visa"* and
`visa-009` says *"My visa application was refused"* — and the model answered no,
reasoning that it was "a routine administrative process covered by company policy".
Rules 2 and 4 name "end an assignment early" and "an extension crossing six months";
`esc-008` and `esc-006` are those cases stated plainly, and both were missed.

That is a matching failure, not a specification failure, and it is the strongest argument
in this project for handling escalation with a deterministic classifier rather than a
model. The false alarms are the opposite problem and *are* specification failures: Rule 1
is broad enough ("personal financial position") to justify escalating almost any
first-person question, and it fires on "what is **my** housing stipend" despite the
prompt explicitly forbidding exactly that inference.

### The gate, as shipped

Full agent runs, gate on, `repeat_penalty` 1.15:

| | Golden (70) | | | Held-out (20) | | |
|---|---:|---:|---:|---:|---:|---:|
| | no gate | gate | **gate+rp** | no gate | gate | **gate+rp** |
| Trajectory | 68.8% | 78.7% | **88.5%** | 65.0% | 60.0% | **75.0%** |
| Escalation recall | 25.0% | 60.0% | **85.0%** | 30.0% | 40.0% | **80.0%** |
| Over-escalation | 0.0% | 8.0% | 12.0% | 0.0% | 20.0% | 30.0% |
| Gate call failures | — | 16/70 | **2/70** | — | 3/20 | **1/20** |
| Median latency | 10.5s | 24.0s | 20.0s | 12.1s | 25.7s | 24.2s |

The middle column is the version that was rejected. Note it *lowered* held-out trajectory
(65.0% → 60.0%): it gained one correct handoff and caused two wrong ones. Fixing the loop
turns that into five gained against three — the same mechanism, with its reliability
defect removed.

Held-out trajectory failures fall from 7 to 5, of which only 2 are escalation. Escalation
has stopped being the thing that caps the system.

## Fixing over-escalation made it worse

Over-escalation is the remaining defect: 12% golden, 30% held-out. Reading the nine false
escalations, **seven were rule 1 misfiring**, in two patterns:

* **A family member is mentioned** (4 cases). Partner language lessons, spousal career
  support, a dependent child's documents. These are benefits the company provides *to* a
  dependant — the employee's own entitlement, written down in the corpus. On `multi-002`
  the gate claimed the question "asks whether the husband and kids may WORK", which it
  does not.
* **The question says "my"** (4 cases). "How much furniture can I ship", "does tax
  equalization apply to me", "does my housing money stop". The v1 criteria already say
  *"do not answer yes merely because the question says I or my"*, and it does not hold,
  because rule 1 offers "personal financial position" as a justification broad enough to
  cover anything.

So a v2 of the criteria was written: rule 1 narrowed to name the specific facts policy
cannot supply, plus an explicit carve-out for dependant benefits. Both changes follow
directly from the traces. **It was abandoned 12 cases into the sweep.**

| Golden sweep | v1 criteria | v2 criteria |
|---|---:|---:|
| Criteria length | 2,061 chars | 2,606 chars (+26%) |
| Gate call failures | **1/40 (2%)** | 4/12 (33%) |

The reasoning loop came back. `repeat_penalty` stops the model circling a single thought;
it does not reduce how much there is to weigh, and two extra paragraphs of exclusions to
check every question against is more deliberation, not less.

**The finding is worth more than the fix would have been:** for a small reasoning model,
correcting a wrong answer by adding a clarifying rule can make the component *less
reliable*, and that cost appears in no accuracy metric — only in the failure rate, which
most eval suites do not measure. Exclusions need to be encoded structurally rather than
described: a deterministic pre-filter, or the post-retrieval design, where the gate is
*shown* the policy that covers the question instead of being told such policy exists.

v2 is kept in `agent/escalation_gate.py`, unused, behind `GATE_PROMPT_VERSION`.

## Reproducibility, verified twice

`v1_raw.json` and `v2_raw.json` originally carried no temperature, seed, or `num_ctx`
— they predate the commit that began recording configuration alongside results. That is
the exact defect config recording exists to prevent, sitting on the baseline every claim
is measured against.

Both were re-run under the recorded configuration:

| | Old | Re-run | Cases with a different tool path |
|---|---:|---:|---:|
| v1 | 50.8% | 50.8% | **0 / 70** |
| v2 | 31.1% | 31.1% | **0 / 70** |

Identical, eight days apart, across every metric. The baselines were deterministic all
along and simply could not prove it. The superseded files are kept as
`v1_unrecorded-config_raw.json` and `v2_unrecorded-config_raw.json` so the difference
between an unverifiable and a verified run remains inspectable.

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
