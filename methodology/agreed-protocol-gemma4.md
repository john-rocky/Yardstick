# Agreed measurement protocol — Gemma-4-E2B (LiteRT team, 2026-07-13→15)

This is the protocol Marissa Ikonomidis (Google, LiteRT) specified and we confirmed in the
collab doc. It is not a house convention we can revise on our own: numbers published against
the LiteRT team, or compared with the HF model card, have to be captured this way or labelled
as something else.

Source: collab doc comment thread, ids 36→43. Quoted verbatim below so nobody has to re-open
the doc to check a premise.

## The spec

| axis | agreed value | source |
|---|---|---|
| **prefill** | **exactly 1024**, via LiteRT-LM's `benchmark()` entry point (force-prefill, **no prompt**) | Marissa 7/13 |
| **decode** | **256 tokens** | Marissa 7/14 |
| **context length** | **forced to 2048** for Gemma-4 (per-card for other models) | Marissa 7/14 |
| **sampler** | defaults, unchanged | Marissa 7/14 |
| **memory** | the card's definitions (`task_vm_info::phys_footprint` on iOS/macOS) | Marissa 7/14 |
| **thermal** | not started throttled — fans or a few minutes between runs | Marissa 7/14 |
| **GPU API** | Android OpenCL · Windows/Linux/Web WebGPU · **iOS Metal** · **macOS historically WebGPU**, moving to Metal | Marissa 7/14 |
| **regime** | **warm** for the card-comparable side-by-side; cold is reported separately as first-use | Daisuke 7/12, 7/14 |

Verbatim, Marissa 7/13:

> We would normally benchmark it at 1024 tokens for prefill. We would typically use LiteRT-LM's
> benchmark model to force 1024 prefill tokens without having to provide a prompt. I believe
> this 2B model shipped with fixed chunk sizes of 128 and 1024 ... if you benchmark with 1024
> or 128 exactly, my upcoming changes shouldn't impact any of your metrics.

Verbatim, Marissa 7/14:

> We typically decode with 256 tokens. Often for benchmarking we force the context length to
> either 1280 or 2048. For the Gemma 4 models, we have been using 2048 ... I don't believe we
> change any default sampler settings for benchmarking ... When we measure memory, we typically
> use these definitions [the HF card's] ... iOS uses Metal — macOS has historically used WebGPU
> but is starting to move to using Metal directly.

Our confirmation, 7/15 (comment id 43):

> Standardizing on: prefill exactly 1024 via benchmark(), decode 256, context forced to 2048
> for Gemma 4 (per-card for others), default sampler, memory per the card's definitions, and
> cool-start discipline between runs. On GPU APIs I'm matching your preferred set — iOS on
> Metal, macOS currently on WebGPU in my numbers ... Fresh warm side-by-side coming.

## Where the harness deviated, and what was done about it (2026-07-27)

The first three were known when this file was written. The last three were found while
closing them, and each one is larger than the deviation that led to it.

| # | deviation | status |
|---|---|---|
| 1 | **decode measured at 128 tokens**, not 256 — `ShortChatTask`, the published ① column | open by design: ① *is* the short-chat column. The 256-token figure is the ④ column, `long-context-1024-gen256`. Two columns, both labelled. |
| 2 | **context never forced to 2048** — `prepareContext` sized it to `prompt + maxTokens + 512` (≈660 short-chat, ≈1,849 at p=1024) | **fixed**: `--context-tokens`, recorded per run as `contextTokensConfigured`. |
| 3 | **the promised warm side-by-side was never delivered** — every published cell is cold | **addressed by the re-capture**: `--runs 4`, run 1 discarded, median of 2–4. |
| 4 | **LiteRT-LM's `benchmark()` cannot express the agreed context.** The Swift helper hardcodes `maxNumTokens: max(prefill, decode) + 32` — **1056** at 1024x256, not 2048. So the entry point the protocol names could never produce the configuration the protocol specifies. | **fixed** by a patch to the vendored package (`ios/BenchmarkApp/Vendored-patches/`). `Engine.initializeForBenchmark` is module-internal, so no wrapper could do it. **Worth raising upstream.** |
| 5 | **`prepareContext` was implemented by exactly one runtime.** LiteRT-LM sized its KV from the task; **llama.cpp was pinned at `n_ctx = 4096`**; MLX is dynamic-KV. The published memory column therefore compared three arms at three different context budgets, in one column, without saying so. | **partly fixed**: llama.cpp now honours it. MLX allocates what it uses and has no budget to force — that has to be *disclosed*, not fixed. |
| 6 | **The deep-context cells had no cross-arm instrument.** `benchmark()` forces prefill without a prompt and only LiteRT-LM has it, so it cannot produce the other four arms' cells at all. | **fixed**: `long-context-1024-gen256` runs every arm, LiteRT-LM included, on one instrument. The vendor `benchmark()` number is reported as its own row — card-comparable, not column-comparable. |

| 7 | **The Mac CLI could not express the protocol at all.** `yardstick run` accepted only `--task/--runtime/--model/--output/--warm/--runs`: no `--context-tokens`, no `--litert-native-benchmark`, and not even the `--model-id` spelling the iOS driver uses. Every Mac cell therefore ran at whatever `prepareContext` derived from the prompt — deviation #2 above, fixed on iOS and left open here for months — and no Mac row could ever be card-comparable. | **fixed** 2026-07-28 (`-r3`): all three flags parsed, bad values rejected rather than defaulted, and the native path emits the same `YARDSTICK_NATIVE_OK` line the iPhone driver does so one importer handles both. |
| 8 | **`--runs 4` measured a run it always discards.** The positional rule keeps runs 2–3, so a 4-run launch and a 3-run launch yield the same two usable runs — the 4th is captured, thrown away, and its heat carried into the next cell. | **fixed**: `RUNS` defaults to 3. This is not a cosmetic change: heat is the binding constraint on how many arms fit in a session, and this returns ~25% of it. |
| 9 | **Two arms were not runnable in this checkout at all.** The MLX-OptiQ E2B catalog entry and the entire Cactus runtime existed only in `~/code/apple-silicon-llm-bench`. A previous handoff explicitly instructed *not* to port `CactusRuntime` — correct for a memory-only question, wrong for a table. This is why the published table has two MLX rows nine days apart. | **fixed** 2026-07-28: OptiQ catalog entry, `CactusRuntime` + vendored xcframework, `RuntimeKind.cactus`, factory case, and `stage-cactus` all ported. |

Two consequences worth stating plainly:

- **A vendor-`benchmark()` prefill number and a task-prompt prefill number are different
  measurements** and may not share a column. The first is what the model card quotes; the
  second is what the other arms can produce. `scripts/analyze_comparability.py` prints them
  separately for this reason.
- **(Superseded 2026-07-28.)** This section previously said LiteRT-LM reports no prompt-token
  count on the app path and that "there is no configuration in which one instrument gives a
  prefill row for all five arms". That was a fact about **our harness**, not about LiteRT-LM:
  `MediaPipeRuntime` discarded the whole per-turn benchmark info whenever a run was capped,
  and the deep-context task always caps. The prefill turn is recorded at prefill completion
  (`TimePrefillTurnEnd` in the vendored runtime), so the counters are valid regardless of how
  decode ends. The fix (keep prefill counters when capped; wall-clock fallback for decode
  only) is proven on both platforms: capped runs report `promptTokenCount` 1106 on Mac and
  iPhone (identical tokenizer count; iPhone capped-vs-EOS prefill rates agree within 3.5%,
  probes of 2026-07-28). The old evidence — "`promptTokenCount == 0`, measured n=16 on
  2026-07-26" — was itself an artifact of the same bug: all 16 of those runs were capped, so
  the ≤-r2 harness discarded the counters in every one of them; it never observed an uncapped
  app-path run. **One instrument now gives the cross-arm prefill row for every arm**;
  the vendor-`benchmark()` row remains a separate, card-reconciliation-only measurement.

## Traps that have already cost a day each

**First-ever ≠ cold.** Fairness rule 2 splits them for a reason. Measured 2026-07-26/27, same
device, same task, same build: LiteRT prefill reads **~1,650 tok/s** on the runs right after an
app install (ML Drift kernel cache being built) and **~3,234 tok/s** (n=7) once the cache is on
disk. That is a 2× swing with no code change. The card reconciliation that prompted rule 2 saw
the same thing at 2.6×. **Never quote a prefill number from the first runs after an install.**

**An XNNPACK cache silently degrades later GPU runs.** One CPU-backend run leaves
`<model>.litertlm.xnnpack_cache_*` (788 MB) beside the model in the HF cache; every later GPU
run then reads 13% slower with double the tail latency and half the GPU power (155→135 tok/s,
ITL p95 6.6→12.3 ms, 22.2→12.4 W). Delete it before any GPU timing.

**The two checkouts behave differently at the token cap.** `~/Downloads/ios-llm-benchmark`
drains the stream and stamps `cappedAt`; `~/code/apple-silicon-llm-bench` breaks out, which
leaves the callback task queued and wedges the *next in-process run* for ~10 minutes
(`callback_thread_pool DEADLINE_EXCEEDED`). Measure from this checkout.

**A `--runs N` launch runs back-to-back, and the protocol does not.** Marissa's spec says
"fans or **a few minutes between runs**". Our launches put ~18 s between runs and 180 s only
between launches, so run 4 of every 4-run launch is measurably throttled — LiteRT-LM −12.9%,
−23.3%, −25.0% and llama.cpp −7.1%, −11.0% against run 2 of the same launch (2026-07-27,
p=1024/g=256). Warm (model stays loaded) and cooled (wait between runs) are **not** in
conflict; the app simply does not implement the pause. *Follow-up for the app: an inter-run
sleep that does not unload the model.*

Until it exists, exclude the tail **by position, never by its thermal flag.** Flag-based
exclusion is asymmetric: on the same round LiteRT-LM's run 4 crossed into `fair` and was
dropped while llama.cpp's run 4, degraded by 11%, stayed `nominal` and was kept — which
flatters the faster arm by ~1.8% purely because it heats up first. The positional rule (drop
run 1 as cold, drop run 4+ as the degraded tail, for every arm alike) never looks at the
number, so it cannot do that. `scripts/analyze_comparability.py` applies it and prints a
RUN POSITION table so the effect being corrected for is visible rather than asserted.

One consequence to plan around: a 4-run launch yields only **2** usable runs, so n≥7 needs
four launches per cell, not three.

**…and that drain poisoned the harness wall-clock column.** Draining is the right call, but
the stream stays open through it, so a decode window measured to *end of stream* silently
includes the drain: measured 2026-07-27, one LiteRT-LM cell read **55.2 tok/s on the engine
column and 15.8 on the wall-clock column** — the column that exists precisely to be the
neutral cross-arm one. It only bites the arm that drains, which is what makes it dangerous:
the other four arms looked fine. Fixed in `-r2` by closing the window at the **last observed
chunk** and dividing by the gaps it spans, which needs no cooperation from any runtime. If
you ever see the two columns disagree by more than a few percent on a *capped* run, suspect
the window, not the engine.

**LiteRT-LM is the only arm whose timing can come from the engine.** `MediaPipeRuntime`
back-derives `generateTime` from `bench.lastDecodeTokensPerSecond`, but only on a natural EOS
finish — a capped run falls back to wall-clock. MLX's `info.generateTime` is a `Date` diff
inside mlx-swift-lm (verified in the linked checkout, `Evaluate.swift:1874-1928`), i.e. wall
clock, same as llama.cpp and Core AI. So a 128-token short-chat column is wall-clock for every
arm; the seam only opens on EOS-terminated runs.

## What a compliant capture looks like

```bash
# prefill + decode, the agreed way (LiteRT only — no other arm has an equivalent entry point).
# --context-tokens is required here too: without it benchmark() runs at 1056, not 2048.
--yardstick-autorun --runtime litert-lm --model-id <id> \
    --litert-native-benchmark 1024x256 --context-tokens 2048

# the cross-arm column: same task, same forced context, every arm including LiteRT-LM
--yardstick-autorun --runtime <arm> --model-id <id> \
    --task long-context-1024-gen256 --context-tokens 2048 --runs 4
```

`--runs 4` with run 1 discarded and the median of runs 2–4 is the warm convention (fairness
rule 2). Both flags exist as of harness stamp `2026-07-27-agreed-protocol-r2`, and the Mac CLI
gained them in `-r3` (2026-07-28); a result whose
`harnessStamp` is older, or whose `contextTokensConfigured` is null, was captured before the
protocol was implementable and is not comparable with the card.

The driver that runs all of this: `scripts/bench_gemma4_e2b_protocol_iphone.sh`.
