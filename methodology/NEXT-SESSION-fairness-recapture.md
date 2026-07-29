# Handoff — Gemma-4-E2B fairness re-capture (written 2026-07-28)

You are the execution session. Your job is to close five fairness defects in a report that goes
to Google's LiteRT team as a collaboration doc **and** out publicly as an X thread Lu asked for.

**This file is the operational one — work from it.** The reasoning behind the five defects, the
history of how the scope moved, and the evidence for each claim live in
`methodology/CAMPAIGN-gemma4-e2b-full-recapture.md`. Read that when you need to know *why*; work
from here for *what to do next*.

---

## Before anything: this handoff is not evidence

The previous handoff in this series (`NEXT-SESSION-gemma4-recapture.md`) had **three premises
that did not survive contact with the measurement**, and is now marked SUPERSEDED. The session
that wrote this file produced several findings that dissolved on inspection, and **every one of
them was caught by the user asking a question, not by the session checking itself.**

So: the campaign doc opens with a table splitting every claim into **measured / read in code /
inferred**, with a re-verification method for each. Start there. The rows marked *inferred* are
the ones that will cost you a day if they are wrong.

One in particular is load-bearing and **unverified**:

> Prefill counters are complete before decode is capped.

The whole prefill fix rests on it. Test it before building on it: run the same prompt capped and
uncapped, compare `lastPrefillTokenCount` and `lastPrefillTokensPerSecond`. If they differ, the
fix in defect 1 is wrong and the resolution reverts to "LiteRT-LM is absent from the prefill
ranking".

---

## Order of operations

### Step 1 — the code fix, and its test (no device needed)

`MediaPipeRuntime.swift:232`:

```swift
let bench = capped ? nil : (try? conversation.getBenchmarkInfo())
```

Split it: keep the benchmark info when capped, use it for **prefill only**, keep the existing
chunk-count + wall-clock fallback for decode. `BenchmarkRunner.swift:315` then stops skipping the
wall-clock prefill, because `promptTokens > 0` becomes true.

**Prove it on device with one launch before scheduling any block.** A capped
`long-context-1024-gen256` run must now report a non-zero `promptTokenCount` for LiteRT-LM, and
that count must match what an EOS-terminating `long-context-1024` run reports for the same prompt.
If it does not match, stop and re-read the assumption above.

### Step 2 — build, install, stage

```bash
D=scripts/bench_gemma4_e2b_protocol_iphone.sh
$D inventory                      # FIRST — what is actually on the device
$D install                        # WIPES the ML Drift kernel cache
$D stage                          # MLX + GGUF                              ** USB **
COREAI_BUNDLE=<dir> $D stage-coreai                                      # ** USB **
CACTUS_SRC=~/code/cactus/weights $D stage-cactus                         # ** USB, ~7.6 GB **
$D unplug-wait                    # unplugged for everything after this
$D warmup                         # gets past first-ever; output DISCARDED
```

`install` is what makes the next runs *first-ever* rather than cold — LiteRT prefill reads ~1,650
there against ~3,234 once the cache is on disk. `warmup` exists for that and its output is thrown
away. This is the single most common way a number in this campaign goes wrong.

### Step 3 — iPhone blocks, one per sitting

| block | task | arms | closes |
|---|---|---|---|
| A1 | depth | litert · mlx · optiq | defects 3, 4 |
| A2 | chat | litert · mlx · optiq | 3, 4 |
| C | chat | litert · coreai · cactus-shipped | 3 |
| E | energy | all arms, one battery window, `nominal` start | 5 |
| N | native | litert | the card-comparable row |

```bash
$D run-block A1     # ~90 min. ONE per sitting. Do not chain two.
```

LiteRT-LM is in every block **as the anchor** — that is what makes cells in different blocks
comparable at all. Device state drifts between sessions by more than the effects being measured
(same binary, same pin: 126–133 tok/s in June, 159–180 in July), so a cross-block ratio goes
through the anchor and the anchor's drift becomes a measured quantity. Footnote ⓖ in the doc
already does this once, for Cactus.

`arm_can_depth` skips Core AI at depth automatically. That is a finding, not a failure: its iOS
KV is capped at 1,024 upstream (`coreai-models CoreAIPipelinedEngine.swift:1416`,
`apple/coreai-models#124`). Do not retry it.

**Block E needs driver work before it runs.** The 07-19 energy cells are unusable because two of
them started `fair` and the battery window was 5% for four arms and 10% for two. The energy block
must gate on `initialThermalState == nominal` (skip and retry the cell otherwise) and use one
window for every arm. That gate does not exist yet — add it.

### Step 4 — Mac, in parallel with the phone's cooldowns

The Mac has no thermal budget to protect: mains power, fans, no staging, no unplug discipline.
It costs nothing in wall-clock and it is why the Mac table went stale — never expensive, just
never scoped.

```bash
M=scripts/bench_gemma4_e2b_protocol_mac.sh
$M build && $M all        # chat + depth + native, arms: litert mlx optiq llamacpp
$M energy
$M analyze
```

The Mac CLI gained `--context-tokens`, `--litert-native-benchmark` and `--model-id` on 2026-07-28.
**Before that it could not express the agreed protocol at all**, so every existing Mac cell ran at
a context derived from the prompt and none was card-comparable. This capture is the Mac table's
first protocol-compliant one.

### Step 5 — Mac Core AI: answer the bundle question before measuring

The harness exists and runs:
`~/code/coreai-models-020-bench/.build/out/Products/Release/llm-benchmark`
Method of record: `~/code/coreai/GEMMA4_LU_BENCH_HANDOFF.md` §2 and §5.

Two things were verified on 2026-07-28, and neither resolves how the published 82.4 / 75.9 was
produced:

- `device_b2/gemma4_e2b_qat_decode_int4lin_tbl_aotc_h18p` is AOT-compiled for **h18p**; this Mac
  is **h16c** → `incompatibleCompiledAssetArchitecture`
- `fleet_exports/gemma4_e2b_qat_decode_int4lin_tbl` loads but dies at p=1024 with
  `GPU buffer allocation failed: per-token input 'ple_table' (9.6 TB)` — the `tbl` gather
  exploding at S>1, which the handoff describes as macOS's known behaviour

**Find the route that produced the published row. Do not substitute a different bundle and call it
the same cell.** `$M coreai` drives the CLI once you know which bundle.

### Step 6 — analyse, import, then the doc

```bash
$D finalize                       # pull + --since filter + import + audit
python3 scripts/analyze_comparability.py results/raw/<campaign> --since=<start>
```

Then the doc. `~/Downloads/docs.docx` carries the current corrections **in red** — that markup is
the user's working aid for transferring edits into the Google Doc. **Do not strip it.** The
document-order change list is `~/Downloads/meeting/docx-changes-inorder-2026-07-28.html`,
regenerated by diffing `docs.docx` against `docs-original-backup.docx`.

---

## Traps, each of which has already cost a day

- **Wrong checkout.** Measure in `~/Downloads/ios-llm-benchmark`. `~/code/apple-silicon-llm-bench`
  diverged 2026-07-20 and holds the *older* published raw
  (`results/raw/2026-07-18-gemma4-bestquant/`). Searching one and concluding "not stored" has
  produced two wrong findings in this campaign alone.
- **First-ever ≠ cold.** The published MLX prefill 2,307 is the median of three launches — 764,
  2,307, 3,111 — with TTFT falling 1,450 → 504 → 372 ms as the kernel cache built. It is the
  middle of a warm-up ramp, published as a steady state.
- **Exclude the thermal tail by position, never by its flag.** Flag-based exclusion drops the run
  of whichever arm heats up first, flattering it by ~1.8%.
- **`--console` waits for the app to exit.** Use it instead of fixed sleeps. An earlier pass
  pulled results before the runs finished and concluded llama.cpp had failed when all six of its
  cells landed minutes later.
- **The pull copies every capture the device has ever taken.** Always filter with `--since`.
- **`scripts/verify_published_numbers.py` does not work.** It matches on bare value with no regard
  for runtime, model or task — it traced Gemma-4's prefill `2,307` to a Phi-4-mini TTFT. An `ok`
  from it is evidence of nothing. Fixing it is part of the job.
- **No result JSON records an HF revision.** That is why the MLX checkpoint question
  (`2c3e507` vs `238767…`) cannot be settled from stored data. Adding it to `BenchmarkResult` is
  the one change that would have prevented that ambiguity — do it while you are in there.
- **Do not edit a driver while it is running.** bash reads scripts incrementally. Measured
  2026-07-27: a scratchpad copy silently wrote three cells' logs to the wrong tree and the
  summariser reported an arm as missing.
- **Open the raw before deciding what a number means.** Every retracted finding in this
  campaign's history came from explaining first and checking second.

---

## Already built — do not rebuild (all verified by build, not by intention)

MLX-OptiQ catalog entry · Cactus runtime + vendored xcframework + `RuntimeKind.cactus` + factory
case + `project.yml` (xcodegen clean) · `run-block` with the anchor design · `stage-cactus` (refuses
a bundle with no `config.txt`) · escalating `cool()` · `RUNS` default 3 (the positional rule
discards run 4, so measuring it only spent heat) · the Mac driver · the Mac CLI protocol flags
(built and smoke-tested) · harness stamp `2026-07-28-agreed-protocol-r3`.

Not done: the device build/install, the `MediaPipeRuntime` prefill fix, the energy thermal gate,
and every measurement.

---

## Done means

Not "every cell shares a date". Per comparison:

1. Every ranking the report prints — table or tweet — is between cells measured the same way.
2. X post 4/6 states in the post what varies between the Mac arms.
3. No arm carries n < 7 in a column that is ranked.
4. The two MLX rows share a session.
5. Energy is re-measured at a `nominal` start with one battery window for every arm.

Anything you cannot close, say so explicitly and leave the cell dated. A disclosed gap is a
result; a quietly substituted number is not.
