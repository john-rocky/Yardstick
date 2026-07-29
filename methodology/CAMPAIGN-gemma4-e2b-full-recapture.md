# Campaign — Gemma-4-E2B: make every published comparison fair (opened 2026-07-28)

**This is not a tidying exercise.** An earlier version of this document argued that the table
should be re-captured because its columns span several dates. That is the wrong reason, and it
produced the wrong scope. The right reason is narrower and more serious:

> **The report currently publishes comparisons that are not like-for-like, and the bias runs
> toward LiteRT-LM — the arm belonging to the team we are sending it to.**

Everything below is organised by that. A defect qualifies only if it makes a *comparison* unfair.
A cell that is merely old, and is not compared against a newer one, is not a defect — it is a
dated measurement, which is normal and disclosable.

The deliverable is a public X thread that Lu asked for, plus this collaboration doc. Both need the
same accuracy; the thread needs more than that, because **a tweet cannot carry a footnote.** Any
caveat that only exists in the doc's footnotes does not exist for a reader of the thread.

---

## How this came about

Written down because the scope moved three times, twice in the wrong direction, and a reader who
only sees the current scope cannot tell which parts are load-bearing.

**It started with one cell.** The published iPhone table read `2,975 / ~56 / 92 MB` for
LiteRT-LM's deep-context column. The 92 MB was queried — it was implausibly small — and turned out
to be a **post-teardown footprint sample**: a single reading taken after `benchmark()` returned and
released the engine, tabulated against the other arms' in-run peaks. The in-run figure is 732 MB
median / 734 MB peak. A caption built on it ("MLX pays 37–54× LiteRT's memory", and elsewhere
"three orders of magnitude") was wrong by the same factor, **and it ran in LiteRT's favour**.

**Fixing that one cell is what set everything else in motion:**

1. The 92 MB could not be corrected without re-measuring the deep-context column → the
   2026-07-27 capture (LiteRT-LM, MLX-PTQ, llama.cpp, Core AI; context forced to 2048).
2. That left column ④ holding 07-27 cells beside 07-18 ones → the prefill column was aligned too.
3. Which left MLX-OptiQ and Cactus stale next to re-measured neighbours → *"re-measure
   everything."*
4. Step 3 was challenged, and it did not survive: a dated cell that nothing is compared against
   is not a defect. The scope was cut to the gaps.
5. That cut was also wrong — it was made while mistaking the deliverable. The X thread is
   something **Lu asked for**; it is public, and a tweet cannot carry a footnote.
6. Re-framing on **fairness** rather than on dates produced the five defects in this document.
   Some of what step 3 wanted is in it; some of what step 4 cut is back. The organising question
   is now "is this comparison like-for-like", which neither of the earlier framings asked.

**What is already fixed in the doc, and is not this campaign's work.** `~/Downloads/docs.docx`
carries 27 corrections in red, machine-diffed against `docs-original-backup.docx` and listed in
document order in `~/Downloads/meeting/docx-changes-inorder-2026-07-28.html`. Those include the
④ column, the prefill column, the memory-inversion finding, footnotes ⓘ/ⓙ/ⓚ, Note 6's retired-value
list, and the two-panel chart replacing the one built on 92 MB. **This campaign is about what
those corrections could not fix**: comparisons that are unfair by construction.

**Findings this session produced and then retracted.** Listed because the pattern matters more
than the items — each was asserted before the raw was opened, and each was caught by the user
asking a question rather than by the session checking itself:

| claimed | actual |
|---|---|
| 92 MB is "added/delta" memory | post-teardown sample; the claim's own arithmetic contradicted it |
| Mac energy is untraceable | a hand-written field list in our own checker omitted the metric |
| LiteRT Mac energy improved 36% | an XNNPACK cache from a CPU-backend run had degraded the baseline |
| decode-at-depth is not flat | measured over a 15–33 token window; **the original wording was right** |
| prefill ranking reverses | n=1–3 in the first-ever regime |
| MLX's +42% prefill cannot be attributed | it can: a warm-up ramp inside an n=3 median |
| LiteRT-LM cannot report prompt tokens | it can; **our harness discards them when capped** |
| Core AI cannot be measured on Mac | it can; Apple's `llm-benchmark`, path in §5 below |

Two of those — the last two — are defects in this document's own earlier drafts. That is the
reason for the verification table that follows.

**Artifacts produced along the way**, usable independently of this campaign:

- `~/Downloads/meeting/verify-mlx-prefill.py` — prints every stored MLX deep-context run and
  re-derives the published cell from them, so the warm-up-ramp finding does not depend on anyone's
  prose
- `~/Downloads/meeting/AUDIT-REQUEST-docx-2026-07-28.md` — 18-item adversarial audit of the red
  corrections, for a session that did not write them
- `~/Downloads/meeting/docx-changes-inorder-2026-07-28.html` — the 27 corrections in document order

## Re-verify this document before acting on it

Every claim below is either something this session **measured**, or something it **read in code
or notes**, or something it **inferred**. The three are not equally reliable, and this session's
record in particular does not earn the benefit of the doubt: several of its earlier findings
dissolved on inspection, and every one of them was caught by the user asking a question rather
than by the session checking itself.

| claim | basis | how to re-verify |
|---|---|---|
| MLX prefill 2,307 is the median of a warm-up ramp (764 / 2,307 / 3,111) | **measured** from stored JSON | `python3 ~/Downloads/meeting/verify-mlx-prefill.py` |
| Energy: two headline cells started `fair`; battery window 5% vs 10% | **measured** from stored JSON | re-aggregate `task == "energy"`, iOS, E2B, and print `initialThermalState` + `batteryDeltaPercent` |
| Prefill is discarded because `capped == true`, not because LiteRT-LM cannot report it | **read in code**, `MediaPipeRuntime.swift:232` + `BenchmarkRunner.swift:315` | read both lines; then prove it by running the fix and seeing a non-zero `promptTokenCount` on a capped run |
| Prefill counters are complete before decode is capped | **inferred** from where prefill sits in the turn | **not verified.** This is the load-bearing assumption of the fix. Test it: same prompt, capped vs EOS run, compare `lastPrefillTokenCount` and `lastPrefillTokensPerSecond`. If they differ, the fix is wrong |
| Mac Core AI ran through Apple's `llm-benchmark` | **read in notes** + binary exists and runs | `GEMMA4_LU_BENCH_HANDOFF.md` §2/§5; the bundle question below is open |
| n = 4 for MLX and Core AI; 10–12 for LiteRT-LM and llama.cpp | **measured** (README of the 07-27 capture) | `analyze_comparability.py` over that campaign |
| MLX-OptiQ and Cactus were absent from this checkout | **measured** — they were, and were ported 07-28 | `git log` on `ModelCatalog.swift`, `RuntimeKind.swift` |
| v0.15.0 does not exist | **checked 2026-07-28** on the releases page | re-check; it may have shipped since |

Where a row says *inferred*, treat the corresponding work item as "test the assumption first".

The previous handoff in this series (`NEXT-SESSION-gemma4-recapture.md`) had three premises that
did not survive contact with the measurement and is marked SUPERSEDED. This document is the same
kind of artifact and deserves the same suspicion. **It is the spec, not the evidence.**

## The five defects

### 1. The prefill column compares two different instruments — and the cause is our harness, not LiteRT-LM

LiteRT-LM's prefill comes from `benchmark()`: a forced 1,024-token prefill **with no prompt**.
Every other arm's comes from the task, running a real templated prompt of ~1,098 tokens through
tokenization and template handling. Different work, different denominator (1,024 vs ~1,098), and
the side that skips work is LiteRT's. Footnote ⓗ discloses it; **X post 4/6 does not**, and prints
`MLX 8,505 · LiteRT-LM 7,305 · Core AI 82` as one ranking.

**The cause, found 2026-07-28 — `MediaPipeRuntime.swift:232`:**

```swift
let bench = capped ? nil : (try? conversation.getBenchmarkInfo())
```

LiteRT-LM *does* return `lastPrefillTokenCount` and `lastPrefillTokensPerSecond` through the app
path. **We throw them away the moment a run is capped** — and `long-context-1024-gen256` caps at
exactly 256 generated tokens, always. So `bench = nil`, `promptTokenCount = 0`, and
`BenchmarkRunner` skips the wall-clock prefill too because it gates on `promptTokens > 0`
(`BenchmarkRunner.swift:315`).

The comment gives the reason: capped mid-turn, the per-turn counters are not finalized. **That is
true of decode and not of prefill.** Prompt processing completes before the first token is
emitted; where decode is later stopped cannot change it. Discarding the prefill counters because
the *decode* was capped is collateral damage.

This was known and worked around rather than fixed. From the gapfill script in the `code/`
checkout: *"the task that fixed decode-at-depth took prefill away for that one arm. The fix needs
no code change: the older `long-context-1024` task ends on EOS, so it still yields LiteRT's
prefill counters."* Running a different task for one arm is exactly how the column stopped being
one instrument.

**Fix (verify before trusting this description):** keep the benchmark info when capped and use it
for prefill only; continue falling back to chunk-count + wall-clock for decode. Then every arm's
prefill comes from the same task and the column is rankable for the first time.

**Also note the reading of Marissa's spec.** Her `benchmark()` line is how Google measures
*LiteRT-LM* — it is the method behind the model card, and no other runtime has that entry point.
It is a card-reconciliation instrument, not a cross-runtime one. Two rows, never merged:

| row | instrument | compares against |
|---|---|---|
| card-comparable | `benchmark()`, 1,024 forced, no prompt | the HF model card, LiteRT-LM only |
| cross-arm prefill | the task's real ~1,098-token prompt | the other runtimes |

### 2. X post 4/6 ranks Mac arms in a way our own notes forbid

`~/code/coreai/GEMMA4_LU_BENCH_HANDOFF.md` §2, written when those numbers were taken:

> MLX vs Core AI is the only true runtime comparison (same ckpt/recipe/block size) … **LiteRT is
> a different product — do not rank it here.**

MLX and Core AI on Mac ran from the *same* checkpoint (`gemma-4-E2B-it-qat-q4_0-unquantized`),
so their gap is a runtime result. LiteRT-LM ran its own `.litertlm` wNa8o8 build with int8
activations — different weights, different recipe. Post 4/6 ranks all three anyway.

**Mac LiteRT-LM is measured, not dropped.** Removing it would be worse, not better: it is one of
the three arms the reader cares about. What has to change is the *claim*: it is a comparison of
each vendor's shipped build, and the checkpoint difference has to be visible in the post itself,
not only in a footnote. Re-measure it under the protocol — the Mac CLI can now express it (see
"already built") — and word the post so the reader knows what varies.

### 3. n is unequal, and the smaller n belongs to the arms that are not ours

LiteRT-LM n=10–12 and llama.cpp n=10, against **MLX n=4 and Core AI n=4** — in a table that
presents all four as equally established, and while citing an n ≥ 7 floor — our own operational convention (the protocol doc's rule, not part of Marissa's agreed spec). The
two under-sampled arms became runnable late in the 2026-07-27 session and ran out of thermal
budget. Fix by measuring, not by annotating.

### 4. The two MLX rows are 8–10 days apart

MLX-PTQ (2026-07-27) sits directly above MLX-OptiQ (2026-07-19 depth, 2026-07-17 chat) — the same runtime, two builds,
read against each other by every reader. Whichever session was kinder wins a comparison that is
supposed to be about the quantization recipe. OptiQ costs almost nothing to add: same runtime,
same driver, one model id, and the catalog entry now exists in this checkout.

### 5. The two headline energy cells were captured while already thermally throttled

X post 3/6 publishes `LiteRT 0.122 J/tok vs MLX 0.151` and `76% vs 64% burst retention`. Audited
2026-07-28 against the stored 07-19 captures, and the defect is not the one this section
originally guessed at:

| timestamp | runtime | J/tok | decode | ITL p95 | initial thermal | battery window |
|---|---|--:|--:|--:|---|--:|
| 07-19 03:00 | litert-lm | 0.122 | 40.3 | 27.6 | **fair** | 5% |
| 07-19 03:18 | mlx-swift | 0.151 | 29.6 | 39.6 | **fair** | 5% |
| 07-19 04:09 | core-ai | 0.352 | 19.0 | 60.7 | nominal | 5% |
| 07-19 06:29 | mlx-optiq | 0.207 | 23.3 | 46.6 | nominal | 5% |
| 07-19 08:21 | llama.cpp | 0.483 | 20.2 | 62.9 | **fair** | **10%** |
| 07-19 18:55 | cactus | 0.322 | 28.7 | 40.7 | nominal | **10%** |

Two things are wrong, and both are in the published comparison:

**The agreed protocol requires not starting throttled** — Marissa, 7/14: *"not started throttled —
fans or a few minutes between runs."* The two cells the thread leads with both start `fair`. That
field alone is the defect; do not lean on their decode rates as corroboration — 40.3/52.7 = 0.765
is literally the published 76% retention ratio (sustained vs burst measure different regimes by
design), and two of the three runs behind the 52.7 burst median started `fair` themselves.
Whether the 0.122/0.151 ratio survives at nominal is unknown — energy per token does not scale
with clock the way rate does, so it cannot be reasoned out; it has to be measured.

**(Corrected 2026-07-28 — the original "battery window" bullet was a misdiagnosis.)** All six
07-19 cells ran the same fixed ~600 s sustain (durations 601–667 s in raw). The 5% vs 10%
battery deltas are an **outcome of power draw** at the 1% battery-level resolution (llama.cpp
≈9.8 W vs LiteRT ≈4.9 W), not a per-arm window setting; nothing ran twice as long. The one real
per-arm asymmetry on 07-19 was Core AI's per-call `maxTokens=192` against 2048 for the others,
which the one-driver re-capture equalizes by construction.

Two further notes so this is not over-read: llama.cpp's first two energy attempts recorded
`null` J/tok with a 0% battery delta (failed captures, correctly not published), and the XNNPACK
cache question that opened this section is **not answerable from stored data** — no result records
whether one was present. That check has to happen at capture time, before the run.

Energy therefore moves from "probably fine, audit it" to **must be re-measured**: all arms, one
session, `nominal` start enforced, one battery window.

---

## What is NOT a defect (do not spend the device on these)

- **Cactus's date.** Footnote ⓖ already routes its cross-row ratios through a same-session LiteRT
  control — the correct technique, correctly applied. Its absolute cells stand at 2026-07-20.
  Re-measuring costs a 3.9 GB sideload and changes no claim.
- **GSM8K.** A property of the weights, n=100 on one Mac harness for every row. Session-independent
  by construction. It has never been verified by this session's chain, but that is an audit task,
  not a measurement one.
- **LiteRT-LM and llama.cpp on iPhone.** Already n=10–12 at 2026-07-27, on protocol. Re-running
  them buys nothing and costs thermal budget the under-sampled arms need.
- **Thinking.** Not measurable, and that *is* the result: v0.14.0 (2026-07-08) is the newest
  release and ships no `ThinkingConfig`; **no v0.15.0 exists** (checked 2026-07-28). The row reads
  "not measurable as of the released binaries", full stop.

---

## Already built (2026-07-28) — do not rebuild

Verified by build, not by intention:

| | |
|---|---|
| MLX-OptiQ E2B catalog entry | `ModelCatalog.swift`, beside the PTQ entry |
| Cactus runtime | `CactusRuntime.swift`, `Vendored/cactus-ios.xcframework` (gitignored), `RuntimeKind.cactus`, factory case, `project.yml` both targets — `xcodegen` clean |
| iPhone blocks | `bench_gemma4_e2b_protocol_iphone.sh run-block <A1\|A2\|B1\|B2\|C\|N\|E>` |
| Cactus staging | `stage-cactus` — refuses a bundle with no `config.txt` at its root |
| Escalating cooldown | `cool()`: +60 s per 30 min of block age, capped |
| `RUNS` default 3, not 4 | the positional rule discards run 4; measuring it only spent heat |
| Mac driver | `bench_gemma4_e2b_protocol_mac.sh` |
| **Mac CLI protocol flags** | `--context-tokens`, `--litert-native-benchmark`, `--model-id` — **built and smoke-tested**. Before this, the Mac CLI could not express the agreed protocol at all, so no Mac cell had a forced context and none was card-comparable |
| Harness stamp | `2026-07-28-agreed-protocol-r3` |

---

## The work

**The operational document is `methodology/NEXT-SESSION-fairness-recapture.md`** — step order,
commands, the block table, and the per-step gotchas live there and are not repeated here, so the
two cannot drift apart. This file answers *why* each step exists; that one says *what to run*.

In outline: fix `MediaPipeRuntime`'s capped-prefill discard and prove it with one launch (defect 1)
→ install and stage → iPhone blocks A1, A2, C, E with LiteRT-LM as the anchor in each (defects
3, 4, 5) → Mac in parallel with the phone's cooldowns (defect 2) → answer the Mac Core AI bundle
question before measuring it → analyse, import, then the doc.

## Standing traps, and the definition of done

Both live in `methodology/NEXT-SESSION-fairness-recapture.md` and are deliberately not repeated
here — a trap list that exists in two files is a trap list that will disagree with itself. That
file's version is a superset of what this one used to carry.

The one-line form of *done*: **not "every cell shares a date", but "every ranking the report
prints is between cells measured the same way."** Everything in this document is downstream of
that sentence.
