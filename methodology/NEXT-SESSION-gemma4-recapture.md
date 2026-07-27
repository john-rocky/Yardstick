> **SUPERSEDED 2026-07-27 — kept for provenance, do not act on it.**
> The re-capture it describes was carried out; results and method are in
> `results/raw/2026-07-27-gemma4-e2b-protocol/README.md` and
> `methodology/agreed-protocol-gemma4.md`.
> Three of this document's premises did not survive the measurement:
> (1) forcing ctx to 2048 does **not** fix a footprint at that context — LiteRT-LM grows into
> its KV rather than pre-allocating it; (2) `long-context-1024-gen256` **had** to be ported,
> since it is the only cross-arm instrument for the deep-context column; (3) the model card's
> 1,450 MB is a stale figure and no gap should be read from it.
> It also missed four harness defects, listed in the protocol doc's deviation table.

# Handoff — Gemma-4-E2B re-capture under the agreed protocol (written 2026-07-27)

## Read this before touching anything

The previous session spent a full day producing findings that dissolved one after another. Every
one of them came from the same root: **acting on a premise that was never checked.** Four
"discoveries" were retracted (Mac energy untraceable → a bug in our own checker; LiteRT Mac
energy improved 36% → a contaminated cache; decode-at-depth not flat → a 15-token measuring
window; prefill ranking reverses → first-ever regime at n=1). A fifth nearly shipped: a
correction that would have replaced an agreed-with-Google protocol with an invented one.

So: **do not start from this document's conclusions. Start by re-verifying its premises.**

### Premise checklist — run this first, before any code or measurement

1. **Which checkout am I in?**
   `git -C ~/Downloads/ios-llm-benchmark log --oneline -1` and the same for
   `~/code/apple-silicon-llm-bench`. They diverged 2026-07-20 and neither contains the other.
   Measurement work belongs in **~/Downloads/ios-llm-benchmark** (it has
   `--litert-native-benchmark`, the agreed prefill path). The published table's raw data lives
   only in **code/**`results/raw/2026-07-18-gemma4-bestquant/`. Searching one and concluding
   "not stored" is how the last session got the Mac question wrong twice.

2. **What is the agreed protocol?**
   `methodology/agreed-protocol-gemma4.md` — quotes Marissa's spec verbatim so you never have
   to re-derive it. If a measurement you are about to take deviates from it, say so out loud
   before taking it.

3. **Is the device in the right state?**
   Unplugged (charging drives thermal `fair` within ~2 cells), `initialThermalState == nominal`,
   and **no XNNPACK cache beside the model** —
   `ls ~/.cache/huggingface/hub/models--litert-community--*/snapshots/*/*xnnpack_cache*`.
   One CPU-backend run leaves one, and it costs 13% of GPU decode silently.

4. **Is this run cold, first-ever, or warm?**
   Fairness rule 2. After any install, the first runs are **first-ever** (kernel caches being
   built) and read ~half the prefill of true cold. Discard them.

5. **Does the number I am about to publish exist in a file?**
   `python3 scripts/verify_published_numbers.py <doc> --results results --results ~/code/apple-silicon-llm-bench/results`
   Both roots. Derived values need `<!-- derived: ... -->`, vendor values `<!-- external: ... -->`.

## State of play

### Already sent to Lu (2026-07-27)

A short status note: the deep-context memory line is wrong, the LiteRT-LM cell in that column
came from a different instrument, the "three orders of magnitude" does not hold (it is a
single-digit multiple, and it ran in Google's favour), and the whole iPhone table is being
re-captured under the pinned protocol. **Check the doc/chat for the exact wording before
following up — do not re-send it.** No numbers were promised beyond "later this week".

### Settled, independent of any re-measurement

These six can go into the doc now (orange), from
`~/Downloads/meeting/doc-corrections-2026-07-26.md`:

| # | correction |
|---|---|
| 4 | Mac table LiteRT GSM8K 85.0 → **86.0** |
| 5 | Note 2 transfer table 85.0 → **86.0** |
| 6 | Note 1: the card's machine is a MacBook Pro M4 Max, ours a Mac Studio |
| 7, 8 | thinking row/paragraph → "as of the released binaries (v0.14.0)" |
| 9 | Note 2's closing question: the card *does* say "static activations"; the open question is whether it is a *requirement* |
| 14 | Cactus's 2.2×/2.6× ratios must go through the same-session control, per footnote ⓖ's own rule |

Verification behind each: quality cells match `results/quality/gsm8k_*.json` exactly; the card
was fetched 2026-07-26; `ThinkingConfig` landed in LiteRT-LM main 2026-07-09 (`ba9a470`) and is
absent from v0.14.0, present in v0.15.0-alpha0's source but that tag ships no binaries.

### Pending the re-capture

Items 1, 2, 3, 10, 11, 12, 15 of the correction sheet. Every number in them was measured with
the **wrong build** (code/ checkout, no agreed prefill path) at a **context of ≈1,849 instead of
2048**, so none of it is publishable. What is confirmed regardless:

- The published **92 MB** deep-context cell is a post-teardown single footprint sample from the
  native-benchmark harness (`BenchmarkApp.swift`, the `footprintMB()` call *after*
  `LiteRTLM.benchmark()` returns and releases the engine). Archived logs of that harness read
  96–107 MB. It is not comparable with the in-run peaks in the rest of that column.
- The Mac energy row (0.090 / 0.106 / 0.154 / 0.170) **is correct and reproducible** — five arms
  re-measured 2026-07-27, main two within 2%. Only the *attribution* needs fixing: LiteRT's Mac
  cells run on **WebGPU**, which is the path Marissa named as macOS's standard, so word it as
  "as shipped", not as an unfair comparison. This applies to the Mac **speed** row too
  (7,305 / 153.2), where the current note mentions only the checkpoint difference.

## The work

1. **Port three things from code/ into Downloads/** (they exist only in the old checkout):
   - median memory sampling — `memoryMedianMB` / `memoryMedianResidentMB` /
     `memoryFinalResidentMB`. Without resident, llama.cpp's 274 MB footprint hides ~3 GB of
     mapped GGUF and the memory ranking cannot be stated honestly.
   - `decodeTokensPerSecondWallClock` / `promptTokensPerSecondWallClock` — LiteRT-LM is the only
     arm that can report engine-derived timing (EOS finishes only), so keep both columns.
   - `harnessStamp` on every result — the 92 MB cell rode five revisions because nothing on it
     said which harness produced it.
   Do **not** port `long-context-1024-gen256` (the agreed decode length is covered by the native
   benchmark) or `CactusRuntime` (needs a 3.9 GB sideload; out-of-thread arm).

2. **Implement `--context-tokens`** — the one piece of the agreed spec no checkout has ever had.
   `BenchmarkRunner.swift` currently sizes the context to `prompt + maxTokens + 512`; the spec
   says force 2048 for Gemma-4. This is the whole explanation for our memory figures sitting far
   under the card's 1,450 MB, so it must land before any memory claim.

3. **Capture**, unplugged, nominal, no XNNPACK cache, discarding the post-install runs:
   - LiteRT-LM prefill+decode: `--litert-native-benchmark 1024x256`
   - every other arm: the app path with `--context-tokens 2048`, `--runs 4`, median of runs 2–4
     (warm — the card-comparable regime), plus a cold set kept separately as first-use
   - n≥7 per cell; `scripts/analyze_comparability.py --since=<start>` decides what may be ranked

4. **Then** finish the correction sheet, regenerate `x_deepcontext_e2b_iphone.png`, and send Lu
   the corrected numbers together with the X thread and the v0.15.0 thinking cell.

## Standing traps

- **First-ever ≠ cold**: prefill reads ~1,650 right after an install and ~3,234 once caches are
  warm on disk. 2× with no code change.
- **XNNPACK cache**: 13% GPU decode loss, doubled tail latency, half the GPU power.
- **Cap behaviour differs between checkouts**: Downloads drains the stream (safe); code/ breaks
  and wedges the next in-process run for ~10 minutes.
- **`--console` waits for the app to exit** — use it instead of fixed sleeps. The earlier passes
  pulled results before the runs finished and concluded llama.cpp had failed when all six of its
  cells had landed minutes later.
- **The pull copies every capture the device has ever taken** — always filter with `--since`.
