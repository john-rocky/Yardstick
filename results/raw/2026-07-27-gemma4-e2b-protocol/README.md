# Gemma-4-E2B iPhone re-capture — 2026-07-27, agreed protocol

Device iPhone 17 Pro (iPhone18,1), iOS 27.0 **24A5380h**, unplugged, harness stamp
**`2026-07-27-agreed-protocol-r2`**, context forced to **2048** on every cell.
Driver: `scripts/bench_gemma4_e2b_protocol_iphone.sh`.
Protocol: `methodology/agreed-protocol-gemma4.md`.

## Two rows, two instruments — do not merge them

This is the distinction the published **92 MB** cell got wrong, so it is written down before  <!-- external: published iPhone table, LiteRT-LM row -->
the numbers are read.

| | instrument | what it is for | arms it can cover |
|---|---|---|---|
| **native row** | `LiteRTLM.benchmark()`, force-prefill 1024, no prompt | comparison **with the model card** | LiteRT-LM only — no other runtime has this entry point |
| **cross-arm column ④** | `long-context-1024-gen256` task through `BenchmarkRunner` | comparison **between runtimes** | every arm, LiteRT-LM included |

Consequences that have already caused one published error each:

- **The native row's memory is not the ④ column's memory.** They are different runs of
  different code paths. `native_*.json` memory belongs in a card-reconciliation sentence, not
  in the ④ cell.
- **The ④ column's prefill cannot be one instrument.** LiteRT-LM reports
  `promptTokenCount == 0` on the app path (n=16, 2026-07-26), so its prefill can only come
  from `benchmark()` while every other arm's comes from the task. If the ④ cell shows a
  prefill number for LiteRT, it is from the native row and must be marked as such.
- **`memoryPostTeardownFootprintMB` is not `memoryPeakDuringDecodeMB`.** The published 92 MB  <!-- external: published iPhone table, LiteRT-LM row -->
  was the former, tabulated against other arms' latter. Both are now recorded per run, in the
  same record, so they can never be confused again.

## `contextTokensConfigured: 2048` does not mean the same thing on every arm

The field records the value the runner **handed to** `prepareContext`, not a value every
runtime obeyed. Only two of them implement it:

| arm | honours `--context-tokens`? | what its KV actually does |
|---|---|---|
| LiteRT-LM | **yes** — as a ceiling | **grows into it, does not pre-allocate**: same ctx 2048, footprint 497 MB at 128 tokens vs 732 MB at ~1,354 <!-- derived: short-chat vs depth medians, both ctx 2048 --> |
| llama.cpp | **yes** (as of 2026-07-27; was hardcoded `n_ctx = 4096`) | allocates KV to `n_ctx` = 2048 |
| MLX | **no — no-op** | dynamic KV; grows to what the run actually uses |
| Core AI | **no — no-op**, and 2048 is unreachable | capped at 1024 upstream on iOS (see below) |

An earlier version of this table said LiteRT-LM "pre-allocates KV to `maxNumTokens`". It was
written before the measurement and the measurement contradicts it: two cells at the identical
forced context differ by 235 MB, tracking tokens used rather than tokens budgeted. Setting a
context ceiling and holding that much KV are different things, and only one of them is what a
flag does.

So an MLX row carrying `contextTokensConfigured: 2048` ran at *whatever it needed*, not at
2048. This is not a defect to fix — a dynamic-KV runtime has no budget to force, and forcing
one would misrepresent it — but it **must be disclosed** wherever the memory column is
compared, because "same context for every arm" is true for two arms and vacuous for two.

Measured effect of the llama.cpp fix, same task, same device, same prompt (1,098 tokens):
**274 MB at n_ctx 4096 (2026-07-26) → 238 MB at n_ctx 2048 (2026-07-27)**, i.e. the published
llama.cpp footprint was taken with twice the context budget LiteRT-LM was given. The direction
favours llama.cpp's ranking rather than threatening it — equalising makes it *smaller* — but
the column was not measuring one thing.

## Native row (LiteRT-LM, `benchmark()` 1024x256, ctx 2048)

Console logs `console_NATIVE_litert_*.txt`; lifted to JSON by
`scripts/import_native_benchmark.py` (they never pass through `ResultStore`, so without that
step they are invisible to both audit scripts).

**Thermal is null on these rows on purpose** — `runNativeBenchmark` starts no `ThermalSampler`.
`analyze_comparability.py` therefore excludes them from its nominal-gated speed table, which is
correct: we cannot attest a regime that was never recorded. *Follow-up for the app: give the
native path the same thermal/energy sampling `BenchmarkRunner` has.*

Cell 1 of the block runs low on speed (prefill −26%, decode −9% vs the rest) while its memory
is identical (661 vs 663 MB). A clock/thermal ramp, not a configuration difference — it is
kept, not deleted, and the median absorbs it.

## Results

Speed medians follow the positional rule (drop run 1 as cold, run 4+ as the degraded tail);
memory medians use every run, since footprint is not thermally sensitive and moved <1% across
all four runs of every launch.

### Native row — card-comparable, LiteRT-LM only (n=7, ctx 2048)

| | median | MAD | range |
|---|---|---|---|
| prefill | **3,878.7** tok/s | 0.1% | 2,853 – 3,884 |  <!-- derived: MAD over n=7 -->
| decode | **60.2** tok/s | 0.1% | 54.8 – 60.3 |  <!-- derived: MAD over n=7 -->
| footprint median | **663** MB | 0.0% | 661 – 663 |
| resident median | 693 MB | 0.0% | 693 – 695 |
| peak footprint | 718 MB | 0.1% | 716 – 720 |  <!-- derived: MAD over n=7 -->
| **post-teardown footprint** | **102** MB | 0.0% | 102 – 102 |

The low end of each speed range is cell 1 of the block; its memory is identical to the rest
(661 vs 663 MB), so it is a clock ramp, not a different configuration.

**This settles the 92 MB question.** The published cell was the post-teardown sample; ours  <!-- external: published cell + correction sheet estimate -->
reads 102 MB on the same basis, n=7, against an in-run median of 663 MB — a 6.5x gap between
two numbers produced by the same run. The correction sheet's estimate ("archived logs read
96–107 MB") lands exactly on it.  <!-- external: correction sheet, archived harness logs -->

### One launch had to be discarded — the session's thermal limit, and how it was recovered

A 4th depth launch was started to lift n from 6 to 8. It has to be **discarded**, and the
reason is worth keeping:

| run | initial | peak | decode |
|---|---|---|---|
| 1 | nominal | nominal | 56.84 |
| 2 | nominal | nominal | 56.94 |
| 3 | nominal | **fair** | **47.41** |
| 4 | **fair** | fair | 37.06 |

In rounds 1–3, run 3 always stayed `nominal` (54.4–56.3). Here it throttled — after ~2 hours
of continuous benchmarking the device no longer recovers inside a 180 s cooldown. That is
**session-level** heat, not the within-launch tail the positional rule corrects for, and the
positional rule cannot catch it: run 3 is inside the keep-window.

The round was stopped before the llama.cpp half ran, on purpose. llama.cpp runs cooler and
would likely have degraded *without* tripping `fair` — reintroducing precisely the arm
asymmetry the positional rule exists to remove. Half a degraded round is worse than none.

That launch is excluded. The session then recovered: the device was cooled for 900 s (not the
usual 300) and the cooldown between launches raised from 180 s to 300 s, after which LiteRT-LM
run 3 read 55.3–56.3 again — back in line with rounds 1–3. **The final depth column is n=12
(LiteRT-LM), n=10 (llama.cpp), n=4 (MLX)**, all from launches whose runs stayed nominal.

The lesson is a driver setting, not a discarded number: 180 s between launches is enough for
the first ~90 minutes and not enough after that. A long session needs its cooldown to grow, or
a mid-session cool-off.

### Cross-arm column ④ — p≈1024 (1,098 prompt tokens), 256 generated, ctx 2048

| arm | decode eng | decode wall | footprint med | resident | n (warm) |
|---|---|---|---|---|---|
| LiteRT-LM | 56.6 | 56.4 | **732** MB | 849 MB | 12 |
| MLX | 46.2 | 46.1 | **3,368** MB | **2,161 MB — MAD 46%, do not rank** | 4 |
| llama.cpp | 37.4 | 38.0 | **239** MB | 3,165 MB | 10 |
| Core AI | — | — | — | — | 0 — prompt rejected, see below |

Engine and wall-clock agree to ~1% on every arm — the `-r2` fix working. Before it, the same
LiteRT cell read 55.2 engine / 15.8 wall.

**The ranking inverts between the two memory columns** — the substance of correction sheet
item 3. By charged footprint: llama.cpp 239 < LiteRT-LM 732 < MLX 3,368. By residency the
order reverses, because llama.cpp's GGUF is mapped rather than wired. **LiteRT-LM is the only
arm under a gigabyte on both**, reproduced here at a matched context budget, which the
published version was not.

**But the resident column cannot carry a three-arm ranking.** MLX's resident median swings
93% between launches (1,152 vs 3,162 MB, identical cell 20 minutes apart) while LiteRT-LM
holds 848 MB across six launches to 0.2% and llama.cpp 3,166 MB to 2.4%. MLX's position in  <!-- derived: per-launch spreads -->
that order therefore depends on which launch you draw. Report MLX's resident as a range with
n; the footprint ordering is safe for all three.

### short-chat column ① — 128 generated, ctx 2048 — all four arms

| arm | decode eng | decode wall | footprint med | resident | n (warm) |
|---|---|---|---|---|---|
| LiteRT-LM | 60.0 | 59.6 | 497 MB | 678 MB | 10 |
| MLX | 48.4 | 48.1 | 2,998 MB | **1,946 MB — MAD 31%, do not rank** | 4 |
| Core AI | 47.0 | 46.6 | **756 MB — range 90%, do not rank** | 2,404 MB | 4 |
| llama.cpp | 38.8 | 39.7 | 191 MB | 3,126 MB | 10 |

This is the only column Core AI can appear in at all (see the 1024 cap below). Its footprint
swings 686→873 MB *within a single launch* while every other arm holds its footprint to ~1%,
so it is reported without a rankable median.

Cold (run 1) vs warm (runs 2–3), the side-by-side that was promised and never delivered:
LiteRT-LM 57.97 → 59.36 and llama.cpp 37.52 → 39.08 <!-- derived: run1 vs runs2-3 medians -->.
The gap is far smaller than the 0.76–0.85 warm/cold ratios measured on Qwen3 0.6B/1.7B/4B
<!-- external: earlier campaign, RESULTS.md -->, because `cold` here means "first run of this
process", not "cold caches": the ML Drift kernel cache survives in the app's `tmp/` across
launches *and across app updates*. **`coldRun: true` is not evidence of a cold cache**, and a
table that leans on the cold/warm distinction has to say which it means.

This is the warm side-by-side that was promised and never delivered. The gap is far smaller
than the 0.76–0.85 warm/cold ratios measured on Qwen3 0.6B/1.7B/4B — because `cold` here means
"first run of this process", not "cold caches": the ML Drift kernel cache survives in the app's
`tmp/` across launches *and across app updates*. **`coldRun: true` is not evidence of a cold
cache**, and a table that leans on the cold/warm distinction has to say which it means.

LiteRT-LM's short-chat → depth footprint increment is **+237 MB** (497 → 733). The published
cell's 92 MB was labelled as an increment; it is not one, and it is not the total either.  <!-- external: published iPhone table legend -->

## A forced context is a ceiling, not an occupancy

The handoff treated `--context-tokens 2048` as the thing that would make our memory figure
comparable with a "2048-context" number. It is not. Three cells, all at a forced ctx 2048:

| | footprint | prompt tokens |
|---|---|---|
| short-chat (n=22) | 497 MB | 21 |
| deep-context (n=24) | 732 MB | 1,098 |
| vendor `benchmark()` (n=7) | 663 MB | 1,024 (forced, no prompt) |

They differ by 235 MB and the differences track **tokens actually used**, not the configured
budget. **LiteRT-LM grows into its KV rather than pre-allocating `maxNumTokens`.**

So a memory cell must state the prompt length it was measured at. "At a 2048 context" does not
pin a footprint; two cells with that label can differ by a third.

`--context-tokens` is still required — it is in the agreed protocol, it is what makes LiteRT-LM
and llama.cpp comparable *to each other*, and without it llama.cpp silently ran at 4096. It just
does not do the job of fixing memory at a context.

**The model card's 1,450 MB is not a target.** It is a stale figure (confirmed 2026-07-27), so
the distance from it is not evidence about our measurement. `analyze_comparability.py` still
prints a CARD RECONCILIATION block against it; that block should be updated or removed rather
than read.

## Core AI cannot meet the agreed context on iOS — and the reason is upstream

The deep-context cell does not fail on memory here. All four runs fail before generation:

```
Context length exceeded: position 0 >= max context length 1024
```

Traced to `coreai-models` `CoreAIPipelinedEngine.swift:1416`, commit `bd8dcf7`:

```swift
let iosDynamicKVCapacityCap = 1024   // "corrupt from the first token" above this on iOS;
                                     // macOS is correct. Remove when the compiler fix lands.
                                     // Full bisect: apple/coreai-models#124
```

So the 1024 is neither our harness nor the exported graph — it is a deliberate cap inside
Apple's runtime package, guarding a device-compiler bug. **Core AI cannot run at the agreed
2048 context on iOS at all**, and the deep-context task's ~1,098-token prompt is already over
the cap, so the arm is rejected before it decodes a token.

This does **not** retract the published "Core AI hits the app memory ceiling and never
finishes" line. That claim is about deep *decode* and is backed by the 2026-07-18 jetsam
evidence (working set +0.25 GB @depth128, +1.06 @256, +3.47 @512 → SIGKILL, reproduced down to
`--sustain-seconds 30`). What today adds is a *second, earlier* wall on deep *prompts*, with a
named upstream cause. Both belong in the note; neither replaces the other.

It also sharpens what `--context-tokens 2048` means across the table:

| arm | at ctx 2048 |
|---|---|
| LiteRT-LM | honoured |
| llama.cpp | honoured (as of today) |
| MLX | ignored — dynamic KV, nothing to force |
| Core AI | **unreachable on iOS** — capped at 1024 upstream |

"Every arm at the same context" is true for two, vacuous for one, and impossible for one.

## MLX: footprint is stable, resident is not a stable quantity at all

| | published (2026-07-26) | launch 51 | launch 52 |
|---|---|---|---|
| footprint (median) | 3,410 MB | 3,370 MB | 3,365 MB |
| resident (median) | 3,170 MB | **1,152 MB** | **3,161 MB** |

Two launches of the identical cell, ~20 minutes apart, same session, same binary, same
checkpoint: **median resident swings 2.7×** while the footprint holds to 0.2%. The published  <!-- derived: 3161/1152 and footprint spread -->
3,170 MB is one draw from that spread, and so is 1,152.

This is written down because the first version of this section got it wrong. Seeing only
launch 51, it reported "resident does not reproduce" and reached for the checkpoint and loader
changes as an explanation. One launch is not a measurement of a quantity this noisy, and the
harness's own `MemorySampler` documentation had already said so — it warns that resident is
page-cache noise at 66–281% run-to-run, and the only mistake was assuming that applied to the
peak but not the median.

**Rank MLX on footprint. Its resident size may be reported only as a range with n attached.**
LiteRT-LM (848 MB) and llama.cpp (3,166 MB) hold their resident medians to ~1% across every
launch, so the instability is specific to MLX, not to the metric everywhere.

MLX is also the only arm with no run-4 thermal tail (LiteRT −13…−25%, llama.cpp −7…−11%,
MLX ≈0%): at 45 tok/s against a 3.4 GB working set it is bandwidth-bound, not compute-bound,
so it barely heats the device. Under a thermal-flag exclusion rule MLX would have kept every
run while LiteRT lost its tail — a larger asymmetry than the llama.cpp case that motivated
the positional rule.

## One 10-minute stall, and why it did not reach the decode column

`CHAT_litert_52` run 2 recorded `ttft_ms=599840` — a ~10-minute wait for the first token,
matching the known LiteRT-LM teardown hang (cancellation not propagating). Its decode was
normal: 60.75 engine / 60.28 wall, 128 tokens. One occurrence in 100+ runs this session.

It is worth writing down because of what did *not* happen. Under the `-r1` decode window
(call start → end of stream) that run would have been recorded at **0.2 tok/s** and dragged  <!-- derived: 128 tokens / 599.8 s, counterfactual under the -r1 window -->
its cell's median down by a third. The `-r2` change — close the window at the last observed
chunk — was made to keep the LiteRT stream *drain* out of the measurement, and it turned out
to exclude this unrelated stall as well.

**TTFT needs its own handling.** 599.8 s is a stall, not a latency, and the positional rule
will not catch it: run 2 is inside the keep-window. Any published TTFT must drop it explicitly
as an order-of-magnitude outlier, with the reason attached.

## Warm-up cells are NOT data

`console_WARMUP_*.txt` were captured under stamp `-r1` and immediately after an app install.
They exist to build the ML Drift / Metal kernel caches and to prove each arm loads. Two
findings came out of them and nothing else should:

- MLX failed to load on the pinned pre-7/6 checkpoint — see the revision note in the driver.
- Core AI aborts at **load** on this OS build (`odiec_module_t`, signal 6), before it can
  reach the depth wall that the published "cannot finish" claim rests on. That claim is
  backed by the 2026-07-18 jetsam evidence and is *not* re-verified here.
