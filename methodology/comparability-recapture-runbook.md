# Comparability re-capture — Gemma-4-E2B, iPhone 17 Pro

**Trigger:** the 2026-07-26 audit of the delivered report found that several published
comparisons put numbers of different kinds side by side. The measurements themselves
reproduce exactly; what failed was the claim layer. This runbook re-captures the axes on one
stated basis so the claims can be rebuilt.

Nothing here needs new analysis code on the device. The app was changed on 2026-07-26 to
record **both sides of every seam** in the same run:

| field | meaning |
|---|---|
| `decodeTokensPerSecond` | engine-reported where the engine exposes counters (LiteRT-LM, MLX), else harness wall-clock |
| `decodeTokensPerSecondWallClock` | **new** — harness wall-clock, identical basis for every arm |
| `promptTokensPerSecondWallClock` | **new** — same, for prefill |
| `memoryPeakDuringDecodeMB` | `phys_footprint` — the jetsam basis; excludes clean file-backed pages |
| `memoryPeakResidentMB` | **new** — `resident_size`; includes mapped-and-resident pages |
| `memoryMedianMB` / `memoryMedianResidentMB` / `memoryFinalResidentMB` | **new** — sampled medians and the settled end value. **Rank on these**, not the peaks |

`medianResidentMB − medianMB` is the mapped share, which is the number that decides whether a
footprint column can be ranked across arms at all.

## Run it

```bash
scripts/bench_definitive_iphone.sh        # ~2.5-3 h, UNPLUGGED, from a full charge
```

This is the one to run. The two earlier scripts
(`bench_comparability_recapture_iphone.sh`, `bench_comparability_gapfill_iphone.sh`) are the
2026-07-26 partial passes and are superseded — keep them only to re-read what was done.

Four protocol changes came out of those passes and are baked into the definitive script:

| change | why |
|---|---|
| **round-robin over arms**, not arm-by-arm | the phone warms over a session; a block design charges that drift to whichever arm runs last. MLX was captured at `fair` while LiteRT-LM had run at `nominal`. |
| **n=7**, not n=3 | measured spread over three identical runs: footprint 1.1-1.3%, but prefill 6-113% and decode 17-38%. One cold-cache outlier (MLX prefill 292 vs a 2,058 median) moves a 3-run median; it cannot move a 7-run one. |
| **`long-context-1024-gen256`** | the original task's tail asks for one sentence, so every arm hit EOS after 15-33 tokens — "decode tok/s at p=1024" was a rate over a couple of dozen tokens. Same prompt, but the model now fills the 256-token budget. The old id is untouched so the 7/18 cells keep their meaning. |
| **`--since` on the analyzer** | the device keeps every capture ever taken and a pull copies all of them. Without the filter, cells from earlier app builds and earlier protocols pool into one median. |

Also fixed in the app on 2026-07-26: `memoryMedianResidentMB` / `memoryMedianMB` /
`memoryFinalResidentMB`. Rank memory on the **medians** — peak resident swung 66% (LiteRT-LM)
and 281% (MLX) across identical runs, because mapped pages fault in and out and the sampler
catches whatever is resident at that instant.

Builds, installs, pre-warms each arm once (discarded), then runs 3 arms × 2 tasks × 7 reps
round-robin, pulls to `/tmp/gemma4-definitive`, and prints the verdicts via
`scripts/analyze_comparability.py --since=<run start>`. Bench **unplugged** and leave the
phone alone (charging drove MLX to `fair` within two cells on 2026-07-26).

## What each cell answers

| capture | finding it closes |
|---|---|
| **LiteRT-LM × long-context-1024-gen256** | **F1** — the first p=1024 footprint for LiteRT on *our* instrument. The published 92 MB was the *added* memory, printed next to other arms' full footprints. Also **F3**: LiteRT finally runs the same task, same templated prompt (~1,082 tok) as every other arm, instead of its own `benchmark()` forced to 1024. |
| **every arm × short-chat** | **F2** — engine-reported and wall-clock decode recorded side by side. The published +14% decode crown was measured across that seam. |
| **every arm × long-context-1024-gen256** | **F1** at depth — footprint *and* resident, so the mapped share is visible per arm. llama.cpp's 300 MB at p=1024 is lower than LiteRT's; whether that is a fair comparison depends entirely on this column. |
| **LiteRT short-chat vs long-context-1024-gen256** | card reconciliation. Google's card reports **1,450 MB** for iPhone 17 Pro / GPU at a 2048 context with the same `phys_footprint` metric; we published 487 MB. `prepareContext` sizes LiteRT's KV to `prompt + output + 512`, so short-chat gets a tiny KV and long-context-1024 gets ~1,800 — if our footprint tracks the KV budget, that is the explanation. |

## Decision rules — what may be claimed afterwards

Apply these literally; they are what the audit was missing.

1. **Decode.** Publish `decodeTokensPerSecondWallClock` for every arm, or publish both columns
   labelled. Do not rank on `decodeTokensPerSecond` — it is engine-reported for two arms and
   wall-clock for the others.
2. **Memory.** Rank on one column and name it. If the mapped share (`resident − footprint`)
   differs materially between arms, the "mmap'd weights — not comparable" caveat applies to
   **every** mapping arm, LiteRT-LM included (its card documents mapping 1.12 GB of
   embeddings). The current table applies it to llama.cpp and Core AI only.
3. **Card gap.** If neither column lands near 1,450 MB at a 1024-token context, we do not yet
   understand the difference, and no memory crown may be published on our number. Say so in
   the report rather than picking the flattering column.
4. **Ratios.** Both operands must share instrument, task, depth and basis. Write the pair down
   before computing the ratio.
5. **Cross-session.** Absolute cells stand with their dates; cross-row ratios go through the
   same-session control. (The report already states this rule and then broke it — the Cactus
   "2.2× memory / 2.6× energy" line is a raw cross-session ratio.)

## If something fails

- **LiteRT rejects or OOMs on the gen256 task** — capture the failure and report it; that is
  itself the answer. It did not fail on the **old** `long-context-1024` task on 2026-07-26
  (818 MB, n=3, spread 1.3%), but that run only decoded 33 tokens; gen256 holds the KV budget
  the same while actually decoding 256, so expect the footprint at or slightly above 818 MB.
- **A model was evicted** — the arm's first cell returns empty. Re-run that arm alone rather
  than lengthening every sleep.
- **Core AI** is not in the matrix: it needs the patched-engine sideload
  (`methodology/core-ai-arm-provenance.md`) and it cannot reach 1024 on iPhone anyway (note 4).
  Its short-chat row is already wall-clock, so F2 does not move it.

## After the run

1. The script already prints the verdicts. To re-read them:
   `python3 scripts/analyze_comparability.py /tmp/gemma4-definitive --since=<run start>`
   (the start timestamp is echoed at the end of the run).
2. Import: `python3 scripts/import_ios_export.py /tmp/gemma4-definitive` into
   `results/raw/2026-07-26-definitive/`. **The importer has no `--since`** — the pull contains
   every capture the device has ever taken, so filter by the run-start timestamp (echoed at the
   end of the run) before importing, or the directory will mix app builds and protocols.
3. Apply the Doc corrections that were held for this measurement — the sheet is
   `~/Downloads/meeting/doc-corrections-2026-07-26.md` (items 1, 2, 3-first-half, 10, 11, 12).
4. Regenerate `x_deepcontext_e2b_iphone.png` on the chosen basis, and the hero chart if the
   decode column moves.
5. Post the correction comment (`~/Downloads/meeting/doc-corrections-2026-07-26.md`) with the
   final numbers rather than the provisional ones.


---

## 2026-07-27 — what the definitive run settled, and the trap it uncovered

46 captures, n=7-9 per cell, MAD 0-1%, **none started above nominal**. Imported to
`results/raw/2026-07-27-definitive/`.

### F2 (decode instrument) — real in the code, immaterial in the numbers

| arm | engine-reported | harness wall-clock | gap |
|---|--:|--:|--:|
| LiteRT-LM | 56.0 | 56.0 | 0% |
| MLX | 49.7 | 49.5 | 0.4% |
| llama.cpp | 39.1 | 40.3 | 3% |

The seam exists — LiteRT and MLX read their own counters, llama.cpp is wall-clock — but it is
worth ≤3%, and LiteRT's two columns are identical because short-chat caps and the capped path
already falls back to wall-clock. **The decode ranking does not rest on the seam.** Publish the
wall-clock column anyway; it costs nothing and removes the question.

### F1 (memory) — the ranking inverts between the two columns, and that is the finding

| arm | footprint | resident | mapped share |
|---|--:|--:|--:|
| llama.cpp | **274 MB** | 3,292 MB | **+3,018** |
| LiteRT-LM | 786 MB | **768 MB** | −18 |
| MLX | 3,410 MB | 3,170 MB | −240 |

llama.cpp's 274 MB is its 2.6 GB of GGUF weights being mapped and therefore not charged to
`phys_footprint`. Counted as resident it is the **largest** arm. LiteRT-LM is smallest on the
column that counts everything resident (768 MB).

**Rule that follows:** state the column. "Smallest memory" is true of LiteRT-LM on resident and
false on footprint, and both columns are legitimate — they answer different questions (jetsam
risk vs total residency).

### The trap: an XNNPACK cache silently degrades later GPU runs by 13%

Running LiteRT-LM once on the **CPU** backend leaves
`<model>.litertlm.xnnpack_cache_<...>` (788 MB) beside the model in the HF cache. Every later
**GPU** run then reads slower:

| | decode | ITL p95 | GPU power |
|---|--:|--:|--:|
| cache present | 134.5 tok/s | 12.26 ms | 12.35 W |
| cache moved aside | **153.2 tok/s** | **6.78 ms** | **21.81 W** |
| 2026-07-19 reference | 155.0 | 6.59 | 22.17 |

p50 barely moves while p95/p99 double: the GPU is stalling, not running slower. This cost us a
day — the 7/26 Mac energy re-capture read 0.098 J/tok against a published 0.154 and looked like
a 36% improvement, when it was a contaminated run. **Check for the cache before any GPU timing,
and delete it if a CPU run happened since the last capture.**

### Gaps left for `bench_gapfill_definitive_iphone.sh`

- **Cactus produced nothing** — eight launches, zero files, undiagnosed. The gap-fill script's
  first cell runs it with the console attached so the reason is visible.
- **LiteRT-LM has no prefill in gen256** — the 256-token cap means `capped == true`, and
  `MediaPipeRuntime` only reads LiteRT's counters on a natural EOS finish, so promptTokenCount
  comes back 0. The older `long-context-1024` task ends on EOS and still yields prefill. Run
  both and quote each for what it can measure.
