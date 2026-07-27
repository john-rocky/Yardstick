# Corrected numbers for the doc — Gemma-4-E2B iPhone, 2026-07-27

Source: `results/raw/2026-07-27-gemma4-e2b-protocol/` (this directory). Every cell captured on
one device, one session, unplugged, harness `2026-07-27-agreed-protocol-r2`, context forced to
2048. Method and caveats: `README.md` beside this file. Protocol:
`methodology/agreed-protocol-gemma4.md`.

The draft doc itself is edited in the meeting session; this file is the numbers and the
sentences they support, not a diff against the doc.

**What is a decision here and what is not.** The numbers, the n, the MADs, and the
*constraints* ("this may not be claimed, because …") come from the measurement and are settled.
The **prose is not**: the footnote wording and the X-thread drafts below are suggestions from
someone who does not have the doc's context, the thread history, or the tone constraints the
LiteRT-team channel runs under. Rewrite them freely — but the constraints they encode are not
negotiable without new measurement.

One in particular. The X drafts lead with a **14× memory ratio**. The number is correct
(footprint only, same session, MLX/llama.cpp), but leading with *any* ratio repeats the shape
of the 92 MB error: pick one memory column, build a multiple, and the other column reverses it.
The finding is the inversion, not the multiple. If the thread leads with a ratio anyway, it
needs the column named in the same breath.

---

## What changed since the correction sheet was written

Three of its assumptions did not survive measurement. They are listed first because they
change what can be *claimed*, not just what the digits are.

**1. Forcing the context does not fix a footprint at that context.** The sheet (via the
handoff) treated `--context-tokens 2048` as the thing that would make our memory figure
comparable to a "2048-context" number. It does not, and our own data shows why: at the *same*
forced 2048, short-chat reads **497 MB** and deep-context **732 MB** — 235 MB apart, tracking
the tokens actually used (128 vs ~1,354), not the budget.

**LiteRT-LM grows into its KV rather than pre-allocating it.** A context *ceiling* is not a
context *occupancy*, and only the ceiling is something a flag can set.

→ Consequence for the doc: any memory cell has to state the **prompt length it was measured
at**, not just the context it was configured with. Two cells at "ctx 2048" are not the same
measurement if one used 128 tokens and the other 1,354.

→ The model card's 1,450 MB is **not** used as a target here: it is a stale figure (confirmed
2026-07-27) and the difference from it is not evidence of anything. The card-reconciliation
block in `scripts/analyze_comparability.py` still prints it and should be updated or dropped.

**2. The resident column cannot carry a three-arm ranking.** MLX's resident median swings 46%
(deep) and 31% (short-chat) across launches — 1,152 vs 3,161 MB for the identical cell twenty
minutes apart — while LiteRT-LM holds 848 MB across six launches to 0.2% and llama.cpp 3,165 MB  <!-- derived: per-launch resident spreads -->
to 2.4%. MLX's *position* in a resident ordering depends on which launch you draw.  <!-- derived: per-launch resident spreads -->

→ The footprint ordering is safe for all three arms. The resident ordering is not. The claim
that survives both is **"LiteRT-LM is the only arm under a gigabyte on both columns"** — true
regardless of where MLX lands, since MLX never goes below LiteRT-LM's 848 MB.

**3. Core AI's deep-context cell fails earlier, and for a different reason, than published.**
Not memory: `Context length exceeded: position 0 >= max context length 1024`, on all 8 runs of
2 launches. Traced to `coreai-models` `CoreAIPipelinedEngine.swift:1416` (commit `bd8dcf7`),
which caps iOS dynamic-KV capacity at 1024 to work around a device-compiler bug
(`apple/coreai-models#124`).

→ This does **not** retract the published "hits the app memory ceiling and never finishes".
That is about deep *decode* and is backed by the 2026-07-18 jetsam evidence. Today adds a
*second, earlier* wall on deep *prompts*, with a named upstream cause. Both belong in the note.

---

## 【1】 iPhone table ④ legend

The cells are prefill / decode-at-depth / **peak footprint**, and the peak is now recorded
alongside the median and the post-teardown sample, so the basis can be named without ambiguity.

Peak vs median barely differ at depth (LiteRT-LM 734 vs 732 MB, llama.cpp 240 vs 239, MLX 3,369
vs 3,367) — with one exception: **Core AI's peak is 1,243 MB against a 753 MB median**, because
its footprint swings within a single launch. If the legend says "peak", Core AI's cell must be
the peak, and its instability disclosed.

## 【2】 LiteRT-LM ④ cell

**Was:** `2,975 / ~56 / 92 MB`
**Now:** `3,879 / 56.4 / 734 MB`

- prefill **3,878.7 tok/s** — vendor `benchmark()`, forced 1024 prefill, ctx 2048, n=7, MAD 0.1%  <!-- derived: MAD over n=7 -->
- decode-at-depth **56.4 tok/s** — harness wall-clock, app path, n=12, MAD 2%
- peak footprint **734 MB** — app path, n=24, MAD 0%

Footnote ⓘ, replacing the sheet's draft:

> This cell previously read 92 MB, which came from a different instrument: a single footprint  <!-- external: published iPhone table, LiteRT-LM row -->
> sample taken after the engine had been released, not the in-run peak every other row reports.
> The same run now records both — post-teardown **102 MB** (n=7, MAD 0%) and in-run median
> **663 MB** — so the two can no longer be confused. Measured on the harness the other arms use,
> at a 1,098-token prompt and a forced 2,048-token context: **734 MB peak / 732 MB median**.

**Note the prefill and the memory come from different instruments and different runs.** LiteRT-LM
reports `promptTokenCount == 0` on the app path, so its prefill can only come from
`benchmark()`; every other arm's prefill comes from the task. That seam cannot be closed and
should be marked in the legend.

## 【3】 Four-line summary, 5th bullet

**Now (all at a forced 2,048-token context, one session, n as marked):**

> At p=1024 the memory answer depends on which column you read, and that is the finding.
> Charged footprint: llama.cpp **239 MB**, LiteRT-LM **732 MB**, MLX **3,367 MB**; Core AI
> cannot run the prompt at all — iOS caps its KV at 1024 tokens upstream. Resident, which counts
> the mapped weights a footprint does not charge: LiteRT-LM **849 MB**, llama.cpp **3,165 MB** —
> llama.cpp's small footprint is 2.9 GB of GGUF held as mapped pages, so it is the largest arm by  <!-- external: unsloth/gemma-4-E2B-it-GGUF Q4_K_M file size -->
> residency and the smallest by footprint. **LiteRT-LM is the only arm under a gigabyte on both.**
> Decode-at-depth stays flat on every runtime that survives (LiteRT-LM 60.0→56.4, MLX 48.4→46.2,
> llama.cpp 38.8→37.4, all wall-clock) — depth separates the arms on memory and survival, not
> speed.

**Deliberately omitted: MLX's resident figure.** It is not stable enough to place in an ordering
(see change 2). If a number is wanted, it is **1,152–3,161 MB, n=2 launches**.

**"Decode-at-depth stays flat" is confirmed** — measured short-chat → depth on the wall-clock
column, every arm is within 7%. The sheet already established the original wording was right and
the planned correction was wrong; this reproduces it under the agreed protocol with a third arm.

## 【10】 Note 8 (c)

**Was:** *(c) lead with the iPhone — no weight alignment needed at all (LiteRT does 1024-token
prefill in 92 MB; Core AI cannot run it at any bit width)*  <!-- external: published Note 8 (c) -->

**Now:** *(c) lead with the iPhone — no weight alignment needed at all (LiteRT-LM holds a
1,024-token prefill at a forced 2,048-token context in **0.73 GB**, 4.6× under MLX; Core AI
cannot run the prompt at all — its iOS KV is capped at 1,024 tokens upstream)*

## 【11】 X thread 1/6 and 2/6

The "6–9×" spread came from the old memory numbers. At matched context it is **4.6×**
(3,367 / 732), and the sharper fact is the column inversion, not the ratio.

### 1/6 — EN
> Same iPhone, same model, same 1,000-token prompt, same 2,048-token context. Change the runtime
> and the memory it needs swings 14×. One can't take the prompt at all.
>
> Gemma-4-E2B on four runtimes. The catch: which runtime "wins" on memory flips depending on
> whether you count charged footprint or resident pages. 🧵

### 1/6 — JA
> 同じiPhone、同じモデル、同じ1,000トークンのプロンプト、同じ2,048トークンのcontext。ランタイムを
> 変えるだけで必要なメモリが14倍開き、そもそもプロンプトを受け付けないものもある。
>
> Gemma-4-E2Bを4ランタイムで実測。要点は「メモリで勝つランタイムは、課金フットプリントで数えるか
> 常駐ページで数えるかで入れ替わる」ことだった。🧵

### 2/6 — EN
> On a 1,098-token prompt: llama.cpp 239 MB, LiteRT-LM 732 MB, MLX 3,367 MB charged footprint.
>
> But by resident pages llama.cpp is the *largest* (3,165 MB) — its 2.9 GB GGUF is mapped, not  <!-- external: unsloth/gemma-4-E2B-it-GGUF Q4_K_M file size -->
> wired, so the footprint never charges for it.  <!-- external: GGUF file size -->
>
> LiteRT-LM is the only one under a gigabyte on both.

### 2/6 — JA
> 1,098トークンのプロンプトで、課金フットプリントは llama.cpp 239 MB、LiteRT-LM 732 MB、
> MLX 3,367 MB。
>
> ところが常駐ページで見ると llama.cpp が**最大**(3,165 MB)。2.9 GB の GGUF は mmap されていて、  <!-- external: GGUF file size -->
> フットプリントには計上されないため。
>
> 両方の列で1GB未満なのは LiteRT-LM だけ。

**The 14× is footprint-only and same-session** (3,367 / 239) <!-- derived: MLX/llama.cpp footprint medians -->. Do not restate it as a
general memory ratio — by residency the same two arms are 1.5× the other way.

## 【12】 Chart

`x_deepcontext_e2b_iphone.png` plots LiteRT's *added* memory against the other runtimes' totals
and carries "MLX pays 37–54× LiteRT's memory". Both inputs are gone: the 92 MB was never an  <!-- external: published chart + its caption -->
increment, and the ratio at matched context is 4.6×. Regenerate with **two panels — footprint
and resident — side by side**, since a single-panel memory chart cannot show the inversion that
is the actual finding.

## 【15】 decode-at-depth

No change. The sheet already established the published wording was correct and the planned
correction was based on a 15–33-token measuring window. Reproduced here at the agreed 256-token
window, wall-clock, with a third arm: LiteRT-LM 60.0→56.4, MLX 48.4→46.2, llama.cpp 38.8→37.4.

---

## Caveats that must travel with these numbers

| | |
|---|---|
| **MLX checkpoint** | `238767…` (2026-07-06), not the `2c3e507` every published MLX cell used. Only this revision loads: the June mlx-swift-lm has no `isKvSharedLayer` branch and rejects both checkpoints. Loader also moved 2026-06-09 → 2026-07-16. |
| **Core AI artifact** | Bundle recompiled today from the *same* 0.4.0 asset with its debug locations stripped (Apple's workaround for apple/coreai-torch#44). Weights and IR unchanged; the **compile stage differs** — MPSGraph package 7.0.56 → 7.0.63, and the 7.0.63 output has no AOT-specialized module. Its short-chat decode (47.0) is not comparable with the 2026-07-18 row (34.2). |
| **Core AI footprint** | Swings 686→873 MB within one launch. Report with n and range, never as a bare median. |
| **MLX resident** | 46%/31% MAD across launches. Range with n, or omit. |
| **n** | depth: LiteRT-LM 12, llama.cpp 10, MLX 4. short-chat: LiteRT-LM 10, llama.cpp 10, MLX 4, Core AI 4. The protocol asks n≥7; MLX and Core AI are under it because they only became runnable late in the session. |
| **Not re-measured** | energy, quality/GSM8K, thinking. Those cells stand as published. |
