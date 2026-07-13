# litert-community & official-converter LLMs on Apple Silicon — Core AI vs MLX vs LiteRT-LM

**A cross-framework decode/prefill benchmark for the LiteRT-LM team.** Extends
`falcon3-comparison.md` (which covered only Falcon3) to two model sets:

1. **litert-community (un-benched)** — official Google `litert-community/*` `.litertlm` models
   that had no Core AI / MLX comparison yet: **DeepSeek-R1-Distill-Qwen-1.5B, Phi-4-mini,
   Gemma3-1B-IT, TinySwallow-1.5B, VibeThinker-1.5B**.
2. **our official-converter conversions** — dense models we converted with upstream `litert-torch`
   (BOCTAV4 int4): **OLMo-2-1B, Qwen3-1.7B, Llama-3.2-3B, SmolLM3-3B, Ministral-3-3B**.

**Hardware:** Mac (M4 Max GPU) + iPhone 17 Pro (on-device). **Protocol (Mac):** steady-state decode at
512 generated tokens, greedy; Core AI via `llm-benchmark -p 512 -g 1024 -n 5`, MLX via `mlx_lm` 512-tok
stream, LiteRT-LM via `litert-mac-verify --max-tokens 512 --backend gpu`. **Protocol (iPhone):** short-chat,
128 tokens, greedy, median of 3 cold launches (Yardstick app). Date: 2026-06-23. Author: john-rocky.

**Conditions (per platform / runtime):**

| Platform | Runtime | Prompt / prefill | Decode (greedy, temp 0) | Repeats |
|---|---|---|--:|---|
| iPhone 17 Pro | Core AI / MLX / LiteRT-LM | `"Explain what on-device AI means in simple terms."` (~20–30 tok incl. chat template) | **128 tok** | median of 3 cold |
| Mac M4 Max | Core AI | 512-token synthetic prompt | **512 tok** | n=5 |
| Mac M4 Max | MLX | short prompt | **512 tok** | — |
| Mac M4 Max | LiteRT-LM | short prompt | **512 tok** | — |

Decode is **iso** across runtimes (same generated-token count + greedy per platform; same int4 weight size unless
noted) → the decode tok/s comparison is apples-to-apples. **The Mac *prefill* prompt is not yet iso** (Core AI uses
a 512-token prompt, MLX/LiteRT short) → it affects **prefill tok/s only** (directional-only, see below), not decode,
and is being standardized to one shared prompt. The iPhone table is fully iso.

> **Cold vs warm (2026-07-13 update, [fairness-rules §2](../methodology/fairness-rules.md)).** The iPhone
> numbers in this report are **cold** (fresh process, first generation) — NOT comparable with vendor model
> cards, which quote in-process steady state. The Qwen3 0.6B/1.7B/4B and Gemma-4-E2B iPhone cells were
> re-captured warm-alongside-cold on 2026-07-13 (`results/raw/2026-07-13-iphone-warm/`; warm = median of
> runs 2-4 of one `--runs 4` process) — see the warm sub-table below. The remaining models' Core AI bundles
> cannot currently be re-assembled (`methodology/coreai-build-regression-2026-07.md`), so their rows stay
> cold-labelled. Two cross-cutting findings from the re-capture: (1) cross-session ratios on this device are
> invalid (`results/raw/2026-07-13-mlx-variance/README.md`) — this table's cells were captured within one
> June-era device state (Jun 23-27), so its *internal* comparisons stand; (2) for ≥1.7B models today's cold re-captures match
> the June values within noise (MLX 1.7B 62.4 vs 62.8; 4B 28.1 vs 27.3), so the June table is not stale for
> those sizes — only the 0.6B dispatch-bound regime moved.

> ⚠️ **Quantization is not uniform — this is the central caveat.** MLX and Core AI are benchmarked at
> **4-bit** (their standard mobile weight size). For LiteRT-LM we benchmark **the file litert-community
> actually publishes**, which differs per model: **DeepSeek-R1 and Phi-4-mini ship only as INT8 (q8)**;
> Gemma3-1B ships INT4. Decode is memory-bandwidth-bound, so an INT8 model reads ~2× the bytes per token
> and is inherently ~half the tok/s of an INT4 one — **the int8 LiteRT rows are not directly comparable to
> the int4 MLX/Core AI rows.** Our own conversions are all INT4 (BOCTAV4), so those rows *are* iso-bit.

---

## Headline — Mac M4 Max GPU, decode tok/s (4-bit unless noted; higher is better)

| Model | LiteRT quant | **Core AI** | **MLX** | **LiteRT-LM** |
|---|---|--:|--:|--:|
| **— litert-community —** | | | | |
| DeepSeek-R1-Distill-Qwen-1.5B | int8* | **319.5** | **323.6** | **115.9*** |
| Phi-4-mini-instruct | int8* | ✗ partial-RoPE¶ | **167.4** | **67.7*** |
| Gemma3-1B-IT | int4 | **327.2**† | **345.5** | **185.7**‡ |
| TinySwallow-1.5B | int8* | **324.1** | **326.7** | **119.6*** |
| VibeThinker-1.5B | int8* | **322.7** | **176.3**§ | **119.6*** |
| **— our official-converter (all int4 BOCTAV4) —** | | | | |
| OLMo-2-1B | int4 | **384.4** | **413.3** | **140.2** |
| Qwen3-1.7B | int4 | **239.1** | **322.7** | **115.8** |
| Llama-3.2-3B | int4 | **198.3** | **208.0** | **94.0** |
| SmolLM3-3B | int4 | **192.9** | **196.0** | **91.2** |
| Ministral-3-3B | int4 | **186.0** | **189.0** | **92.8** |

§ VibeThinker MLX decodes ~1.8× slower than the other 1.5B Qwen-family models (176 vs ~320) despite the same
param count — likely a wider intermediate/vocab in its config; flagged for a re-check.

‡ Gemma3-1B LiteRT = **self-converted** (google/gemma-3-1b-it → int4 BOCTAV4 via official `litert_torch`; the
litert-community mirror is HF-gated). At 1B + gemma-tuned LiteRT kernels it decodes **185.7** — *faster* than the
larger generic int4 builds (OLMo-2-1B 140, Qwen3-1.7B 116, Llama-3.2-3B 94), a direct illustration of the
"Gemma gets first-class LiteRT kernel treatment, the generic third-party path doesn't" thesis from
`litert-speed-findings.md`. The MLX gap is correspondingly smaller here (345 / 186 = 1.86×).

`*` int8 = the only quant litert-community publishes for that model; not iso-bit vs the int4 MLX/Core AI cells.
`†` Gemma3-1B Core AI via a local text-only-config fix to `coreai-models/export/pipeline.py` (the gemma3 wrapper
otherwise assumes the multimodal config). The **2026-06-25 export pass** added macOS classes for **olmo2 / smollm3 /
ministral3** (now benched — 384.4 / 192.9 / 186.0) → **Mac Core AI is now 9/10**. `¶` **partial-RoPE** = Phi-4-mini's
`partial_rotary_factor` 0.75 rotates 48 of 64 head-dims; the export passes HF parity but both the Mac `llm-benchmark`
harness (`RoPE freqs [48] ≠ half_embed [64]`) and the iOS pipeline (`Array index out of range`) assume full rotary —
the lone remaining Mac block, and a hard wall on iPhone **GPU + ANE**. `△` Gemma3-1B ANE = dual local/global RoPE +
sliding/full attention not expressible on the single-mask / single-RoPE iOS pipeline (GPU fine). `◇` Ministral-3-3B ANE
= needs the multimodal-FP8 `Mistral3ForConditionalGeneration` loader (GPU ships via the macOS-shim path).

## Prefill tok/s (Mac M4 Max GPU)
| Model | Core AI | MLX† | LiteRT-LM |
|---|--:|--:|--:|
| DeepSeek-R1-1.5B | **4210** | 168† | 486 |
| Qwen3-1.7B | **3640** | 1004† | 396 |
| OLMo-2-1B | **4975** | 1491† | 672 |

† MLX/LiteRT prefill use short prompts (not the 512-tok synthetic prompt Core AI uses), so compare prefill
*within* a runtime / directionally only. Core AI's prefill lead is real and large regardless.

---

## Findings (preliminary — Mac)

1. **On Apple GPU, Core AI ≈ MLX, both clearly ahead of LiteRT-LM on decode** — the same conclusion as the
   Falcon3 study, now reproduced across more architectures. For the iso-bit int4 rows: Qwen3-1.7B = Core AI
   239 / MLX 323 / LiteRT 116; OLMo-2-1B = Core AI 384 / MLX 413 / LiteRT 140. LiteRT-LM lands at **~35–50 % of MLX**
   decode at the same int4 weight size.
2. **The int8-only community models look even slower — but that's a publishing choice, not a runtime defect.**
   DeepSeek-R1 and Phi-4-mini are only on litert-community as **q8/int8** — at int8 they read 2× the bytes, so
   decode is ~half what an int4 build would. That's almost certainly a deliberate quality call (int4 PTQ hits a
   reasoning model like R1 hardest), so we **disclose the quant rather than prescribe int4**: part of their gap
   vs MLX/Core AI is the published quant, not the runtime.
3. **Core AI's prefill is in a different class** (3.6–4.2 k tok/s vs LiteRT's ~0.2–0.7 k) — matches Falcon3.
4. **Root cause is unchanged from `litert-speed-findings.md`:** LiteRT-LM runs int4×int8 **INTEGER** matmul over
   the **WebGPU(Dawn)→Metal** delegate; MLX/Core AI dequantize int4→fp16 and run **native-Metal fp16 GEMM**,
   which Apple GPUs do extremely fast. The decode gap is kernel/delegate efficiency, not bit-width.
5. **Core AI arch coverage was the practical limiter — the 2026-06-25 export pass closed most of it, validating the
   "just needs a per-arch class" thesis.** qwen2/qwen3/mistral wrappers cleanly cover DeepSeek-R1 (319.5), TinySwallow
   (324.1), VibeThinker (322.7), Qwen3-1.7B (239.1); Gemma3-1B needed only a text-only-config fix to the existing gemma3
   wrapper (now 327.2 — it had assumed the multimodal `text_config` / `language_model.` layout and `AttributeError`'d on
   the text-only 1B). The pass then **wrote the missing macOS classes for olmo2 / smollm3 / ministral3** — all bench
   cleanly (**384.4 / 192.9 / 186.0**), exactly as predicted ("unwritten class, not unsupported"). **Mac Core AI is now
   9/10.** The lone block is **Phi-4-mini partial rotary** (`partial_rotary_factor` 0.75): its class exports and passes
   HF parity, but `llm-benchmark`'s RoPE assumes full rotary and aborts (`RoPE freqs [48] ≠ half_embed [64]`) — a
   harness/exporter limitation on partial-rotary, not an arch Core AI "can't do" (the same wall recurs on iPhone GPU+ANE).
6. **The 1.5B Qwen-family models all decode at ~320 tok/s on both Core AI and MLX** (DeepSeek 319.5/323.6,
   TinySwallow 324.1/326.7, VibeThinker 322.7/—) — same architecture, same Apple-GPU ceiling; Core AI and MLX
   are statistically tied, LiteRT-LM's published int8 builds sit at ~half that (bandwidth, not runtime).

## iPhone 17 Pro (on-device) — Core AI vs MLX vs LiteRT-LM (short-chat, 128 tok, median of 3 cold)

| Model | **Core AI ANE** | **Core AI GPU** | **MLX** | **LiteRT** |
|---|--:|--:|--:|--:|
| DeepSeek-R1-1.5B | **83.3** | **75.9** | **73.0** | **30.7** |
| TinySwallow-1.5B | **74.8** | **75.0** | **71.6** | **30.6** |
| VibeThinker-1.5B | **71.5** | **75.7** | — | **30.4** |
| Qwen3-1.7B | **64.8** | **67.6** | **62.8** | **23.2** |
| Qwen3-4B (iso-int4) | **29.7** | **28.3** | **27.3** | **21.2** |
| Gemma3-1B | ✗ dual-RoPE△ | **103.6** | **97.6** | gated |
| Phi-4-mini | ✗ partial-RoPE¶ | ✗ partial-RoPE¶ | **29.6** | OOM |
| OLMo-2-1B | **95.6** | **86.1** | — | **24.6** |
| Llama-3.2-3B | **24.2** | **19.3** | **34.0** | **18.4** |
| SmolLM3-3B | **23.0** | **20.5** | **36.8** | **22.8** |
| Ministral-3-3B | ✗ multimodal-fp8◇ | **17.6** | ✗ | **18.0** |

![On-device decode — Core AI ≈ MLX ≫ LiteRT-LM across architectures (iPhone 17 Pro)](charts/iphone_decode_summary.png)

### Warm re-capture (2026-07-13) — Qwen3 ladder + Gemma-4-E2B, cold AND warm

Per fairness-rules §2 (warm = median of in-process runs 2-4 of one `--runs 4` launch; cold =
run 1; thermal-gated, Release, one session window; raw device JSON:
`results/raw/2026-07-13-iphone-warm/`):

| Model | Runtime | iPhone cold | iPhone warm | Mac cold | Mac warm (r2-5) |
|---|---|--:|--:|--:|--:|
| Qwen3-0.6B | MLX | 167.2 | **158.8** | — | 455† |
| Qwen3-0.6B | LiteRT-LM (mixed int4) | 121.0 | **120.4** | 247.2 | **267.1** |
| Qwen3-1.7B | MLX | 62.4 | **60.4** | — | — |
| Qwen3-1.7B | LiteRT-LM (int4-mixed, ours) | 47.6 | **48.2** | 166.8 | **172.6** |
| Qwen3-4B | Core AI ANE | withheld† | withheld† | — | — |
| Qwen3-4B | MLX | 28.1 | **27.9** | — | — |
| Qwen3-4B | Core AI GPU | withheld† | withheld† | — | — |
| Qwen3-4B | LiteRT-LM (mixed int4) | 24.6 | **23.8** | 106.6 | **109.2** |
| Gemma-4-E2B | LiteRT-LM | 55.1-55.9 | **59.7**‡ | 151.0 | **152.4**§ |
| Gemma-4-E2B | MLX | 47.5 (June cold) | blocked¶ | — | — |

† **Core AI 4B rows withheld**: every run of both 2026-07-13 sessions (including the
thermal-gate re-runs) reported `initialThermalState=fair`, and the driver logged
THERMAL_FAIL for both cells — protocol requires nominal, so these numbers are not published
(raw data kept in the campaign dir's EXCLUDED.txt for audit; values were internally flat and
the cold matched June's 29.7/28.3, but a clean-nominal re-capture is required first).
Mac MLX from the June steady-state `mlx_lm` protocol (already warm-equivalent).
‡ n=2 warm runs (morning session; the evening re-run was interrupted at run 1 = the 55.9 cold).
§ 1,029-token-prompt protocol vs the official card — `results/raw/2026-07-13-e2b-mac-webgpu/`.
¶ `mlx-community/gemma-4-e2b-it-4bit` updated upstream 2026-07-06; no longer loads with the
pinned loader (`v_proj.weight not found`), so no valid MLX E2B re-capture exists.

**What warm changes — and what it doesn't.** Cold→warm is flat (±5%) for MLX and LiteRT on
every Qwen3 size, +9% for LiteRT on E2B (engine ramp), and Core AI GPU ramps *up* in-process
(22.9→26.5 within its session). **No cross-runtime ranking flips at ≥1.7B** — the June cold
table's ordering stands there. The one correction is at **0.6B**: the previously reported
"LiteRT ≈ MLX tie (118.6 vs 119.6)" compared LiteRT-Release against an MLX **Debug** build;
Release-warm same-session gives **MLX 158.8 vs LiteRT 120.4 → LiteRT ≈ 0.76× MLX**. The
LiteRT/MLX warm ratio improves with size (0.76 → 0.80 → 0.85 at 0.6B/1.7B/4B) — consistent
with the dispatch-vs-bandwidth analysis above. Core AI 0.6B/1.7B warm rows are blocked by a
toolchain regression (`methodology/coreai-build-regression-2026-07.md`); Core AI's warm
behaviour is carried by the 4B pair above.

**On-device the result is size- and arch-dependent — but Core AI's ANE remains the trump card MLX/LiteRT structurally can't
use.** For the qwen-arch ≤1.7B Core AI iOS covers, Core AI leads: DeepSeek-R1 **ANE 83.3** / GPU 75.9 vs MLX 73.0 vs
LiteRT 30.7; TinySwallow 74.8 / 75.0 vs 71.6 vs 30.6; Qwen3-1.7B 64.8 / 67.6 vs 62.8 vs 23.2 — **Core AI ≈-or-beats MLX
(ANE tops MLX on DeepSeek-R1, 83 vs 73), both ~2.5× LiteRT-LM.** The **2026-06-25 export pass added six more models on
iPhone**, and the outcome **splits by size**: at **1B** Core AI stays ahead — **Gemma3-1B GPU 103.6 > MLX 97.6**, and
**OLMo-2-1B ANE 95.6 / GPU 86.1 ≈ 4× LiteRT 24.6** (no MLX repo). At **3B, MLX pulls ahead of Core AI**: Llama-3.2-3B
MLX 34.0 vs Core AI ANE 24.2 / GPU 19.3 (≈ LiteRT 18.4); SmolLM3-3B MLX 36.8 vs Core AI ANE 23.0 / GPU 20.5 (≈ LiteRT
22.8); Ministral-3-3B Core AI GPU 17.6 ≈ LiteRT 18.0 (MLX can't load it). **Two constants hold at every size:** (1)
**within Core AI the ANE always beats its own GPU** (OLMo-2 95.6 > 86.1, Llama 24.2 > 19.3, SmolLM3 23.0 > 20.5) — and the
**ANE is reachable only via Core AI / CoreML** (MLX and LiteRT-LM are GPU-only on Apple), so it is also the most
power-efficient path, Core AI's real on-device edge that's invisible on Mac; (2) **both Core AI paths stay ≥ LiteRT-LM.**
So the honest on-device summary is **Core AI ≥ MLX ≫ LiteRT for ≤1.7B-qwen and 1B; MLX > Core AI ANE > GPU ≈ LiteRT at 3B**
— MLX's mature Metal decode kernels win the bandwidth-bound 3B regime, while Core AI owns the ANE and the smaller models.

**Three iPhone Core AI cells are confirmed architectural walls, not pending work** (legitimate Core AI iOS-pipeline coverage
findings): **Phi-4-mini partial rotary** (`partial_rotary_factor` 0.75) crashes **both GPU and ANE** with `Array index out of
range` — the same root cause as the Mac `llm-benchmark` `RoPE freqs [48] ≠ half_embed [64]` failure; the standard pipeline
assumes full rotary. **Gemma3-1B ANE** can't express alternating sliding/full attention + dual local/global RoPE on the
single-mask/single-RoPE iOS pipeline (GPU is fine — 103.6). **Ministral-3-3B ANE** needs the multimodal-FP8
`Mistral3ForConditionalGeneration` loader (GPU ships via the macOS-shim path — 17.6). (Earlier-session note: the VibeThinker
ANE bundle was re-assembled and measured at **71.5 tok/s**; `ministral3`/OLMo-2/VibeThinker have no mlx-community iPhone repo.)

**iPhone 3B int4 loads fine on LiteRT-LM — the earlier failures were a *harness* misconfiguration (missing memory
entitlements), not LiteRT.** The BenchmarkApp was built without `com.apple.developer.kernel.increased-memory-limit`
and `…extended-virtual-addressing`. Without them a large weight-section `mmap` fails with `Cannot allocate memory`
(ENOMEM) even on this **12 GB** device — we saw **Ministral-3-3B 0–1/3 and Llama-3.2-3B 1/3**. **Adding both
entitlements fixed it: Ministral-3-3B 3/3 (~18 tok/s), Llama-3.2-3B 5/5 (~18.4).** Both are required —
`increased-memory-limit` raises the footprint ceiling (the many-small-sections case, Ministral) and
`extended-virtual-addressing` supplies address space for one large contiguous section (Llama). We chased two wrong
hypotheses on the way — an `externalize_embedder` bug, then a "LiteRT mmap-loader limitation" — **both refuted.**
MLX only *appeared* to "load 3B where LiteRT couldn't" because its lower load-peak stayed under the default limit.
**LiteRT-LM is not at fault for the iPhone 3B failures; it was our app config.** (Lesson: the iPhone harness needs
both entitlements for any ≳2 GB model. We then re-checked Phi-4-mini LiteRT: its **int8 build (3.6 GB) OOMs even
*with* the entitlements** (`std::bad_alloc` → SIGABRT) — but that's the **quant, not Phi or LiteRT**. MLX runs Phi
at int4 (~2 GB, 29.6 tok/s) and LiteRT runs other 3B int4 builds (~2.2 GB) fine, so a LiteRT int4 Phi (~2 GB) would
load too — litert-community just only ships Phi as int8. So the iPhone "OOM" cells split: **3B int4 = harness
artifacts (now load); Phi int8 = too big at *that quant* (int4 would fit); genuinely-unrunnable on-device = none.**)

**Core AI iOS:** now measured for **16/20** cells — the qwen-arch ≤1.7B set (DeepSeek-R1, TinySwallow, VibeThinker,
Qwen3-1.7B) **plus the 2026-06-25 export pass** (Gemma3-1B GPU; OLMo-2-1B, Llama-3.2-3B, SmolLM3-3B both ANE+GPU;
Ministral-3-3B GPU). Core AI ≥ MLX ≫ LiteRT at ≤1.7B-qwen and 1B; **MLX leads at 3B**; **ANE > Core AI's own GPU throughout**.

### Coverage — 60 measured cells; every blank is a documented architectural block
- **Mac 29/30:** all 10 models have MLX + LiteRT; **Core AI 9/10** (qwen2/qwen3/mistral/gemma3 + the new
  olmo2/smollm3/ministral3 classes — 384.4/192.9/186.0). The only Core AI block is **Phi-4-mini partial-RoPE** (exports
  + passes parity, but `llm-benchmark`'s full-rotary RoPE aborts).
- **iPhone 31/40:** Core AI **16/20** (7 ANE + 9 GPU), MLX 7/10, LiteRT 8/10. Blocks: **Core AI ANE ×3** — Gemma3-1B
  (dual-RoPE/sliding), Phi-4-mini (partial-RoPE), Ministral-3-3B (multimodal-fp8 loader) — **plus Phi GPU** (partial-RoPE,
  the one GPU block); **MLX ×3** — no mlx-community repo for OLMo-2/VibeThinker, MLX-Swift can't load `ministral3`;
  **LiteRT ×2** — Gemma3-1B gated, Phi int8 (3.6 GB) OOM. (Llama/Ministral 3B LiteRT now load with the memory entitlements.)

## Supplementary — LiteRT int4 (the byte-vs-delegate split) + Core AI static-GPU status

**LiteRT int8 vs int4 — measured, not estimated.** The official litert-community DeepSeek-R1-1.5B ships **q8** (int8).
We pushed a **byte-matched int4** of the same model (own BOCTAV4 blockwise-32 + OCTAV, GSM8K 73%) as a supplementary
row; the official q8 row stays. iPhone 17 Pro, short-chat, median of 3 cold:

| DeepSeek-R1-1.5B · LiteRT-LM · iPhone | decode tok/s | peak MB | GSM8K (bf16 = 81.0) |
|---|--:|--:|--:|
| q8 (official litert-community, ≈quality-parity) | 30.7 | ~1,700 | ≈parity |
| int4 (own BOCTAV4, **reference only — not parity**) | **45.0** | 1,250 | **73.0 (−8 pt)** |

⚠ **The int4 row is a speed _reference value_, not a quality-parity artifact.** It scores GSM8K 73.0% vs bf16 81.0%
(−8 pt — a 1.5B reasoning model's 4-bit sensitivity; the 7B sibling is −1 pt, at parity). So read the byte effect as
"int4 _could_ reach 45.0, but at an 8-pt accuracy cost," not as a free speedup. The official **q8 is the parity row.**

With that caveat the split is still informative. The int8→int4 byte reduction buys **1.47×** (30.7 → 45.0) — real, but
short of the ~2× a pure-bandwidth model would give, and it **crosses a quality boundary**. The cleaner, **iso-int4**
comparison stays inside one quant: **LiteRT int4 (45.0) is only ~0.6× of Core AI / MLX int4 (~73–83)** — that residual
**~1.67×** is the **delegate/kernel gap** (WebGPU(Dawn)→Metal vs native-Metal fp16 GEMM), measured at matched 4-bit. So
LiteRT-LM's on-device gap is **both** int8 bytes (~1.5×, with an accuracy cost) **and** the delegate (~1.7×, iso-int4,
quality-neutral) — "it's just int8" does not explain it, and the native-Metal kernel path is the larger, cleaner lever.

**Engine efficiency — % of the memory-bandwidth ceiling.** Decode is memory-bandwidth-bound (a dense model reads all
its weights once per token), so **achieved read-BW = decode tok/s × weight GB** is the bytes-normalized engine metric.
DeepSeek-R1-1.5B, iPhone, exact weight bytes:

| DeepSeek-R1-1.5B | decode | weight GB | achieved GB/s | % of ceiling |
|---|--:|--:|--:|--:|
| Core AI ANE | 83.3 | 0.97 | **80.8** | 100% |
| Core AI GPU | 75.9 | 0.95 | 72.1 | 89% |
| MLX | 73.0 | 0.95 | 69.3 | 86% |
| LiteRT q8 (int8) | 30.7 | 1.70 | 52.2 | 65% |
| LiteRT int4 (BOCTAV4) | 45.0 | 1.00 | 45.0 | 56% |

![Engine efficiency — achieved decode bandwidth as % of the memory-bandwidth ceiling (DeepSeek-R1-1.5B, iPhone 17 Pro)](charts/iphone_bw_ceiling.png)

Ceiling = the best-achieved cell = **≥ 80.8 GB/s effective** (this corrects a too-low 60 GB/s estimate; the A19 Pro
peak is higher still). Two things fall out:
1. **Native runtimes saturate the bus (86–100%); LiteRT-LM saturates only 56–65%.** So the WebGPU(Dawn)→Metal delegate
   runs at **~0.6× the memory-bandwidth efficiency** of the native (ANE / Metal) kernels — the delegate gap as a
   number, not an adjective. (Reproduces on Qwen3-4B iso-int4: native 92–100%, LiteRT 71%.)
2. **LiteRT's int4 path is _less_ BW-efficient than its int8 path** (45.0 vs 52.2 GB/s; 56% vs 65%). That is exactly
   why int4 bought only 1.47×, not the full 2× the byte-halving allows: if the int4 kernel saturated the bus like the
   int8 one (52 GB/s ÷ 1.0 GB ≈ 52 tok/s), int4 would hit ~52, not 45. So **optimizing the int4 delegate kernel is a
   concrete, separable LiteRT lever** — on top of shipping int4 at all.

**Core AI ANE-vs-GPU — answered by a warm re-run, not a static-GPU bundle.** The clean way to isolate the ANE/GPU
*engine* would be a static-shape GPU export (matched shape + quant vs the ANE). All 10 `*_static_gpu` bundles
compiled (0 ANE regions) but are **un-runnable**: the GPU/MPSGraph compile emits only **one** specialized graph while
the static structure declares all 15 `extend_*` shape buckets, so the engine resolves a declared bucket and hits a
missing artifact → `EngineFactory` POSIX Code 2 `No such file`. This is an **artifact absence**, not a metadata or
loader issue — verified: re-compiling with `--expect-frequent-reshapes` is byte-identical, and the low-level
`PreparedModel` runner (which runs other GPU bundles) only rescues *dynamic* single-`main` graphs, a different
structure. A genuinely fixed-shape GPU export (the `qwen2_bucketed` host-cache port) is real but low-ROI and
Mac-unverifiable, so it is dropped.

Instead the engine-vs-cold question is answered directly by **re-running the existing dynamic-GPU bundle warm** (2nd
run, same shape — no new bundle needed). DeepSeek-R1-1.5B, iPhone, one session, run 1 cold → runs 2–4 warm:

| DeepSeek-R1-1.5B | decode cold | decode warm | TTFT cold → warm |
|---|--:|--:|--:|
| dynamic-GPU | 76.1 | 74.5 | 195 → 39 ms |
| static-ANE (control) | 82.7 | 82.3 | 79 → 26 ms |

**Cold re-specialization lands in TTFT (GPU 195 → 39 ms), not the decode rate** (GPU cold 76 ≈ warm 74.5; the ANE is
flat, as a static-AOT control should be). So **warm-GPU does not approach the ANE — the ~10% ANE>GPU decode delta is
steady-state, not a cold artifact.** Combined with the sign-flip (DeepSeek ANE>GPU, but Qwen3-1.7B / VibeThinker
GPU>ANE), that small steady-state delta is best read as the **static-vs-dynamic export shape** (model-specific), **not**
a stable engine ranking. An earlier fixed-shape GPU artifact settles the direction outright (coreai-side, qwen3-0.6b,
iPhone, iOS 26): **static-GPU 115 > ANE 69.6 > dynamic-GPU 57** — given the ANE's fixed shape, the GPU actually *beats*
the ANE. (The current h18p/27β static-GPU rebuild is un-runnable, but this earlier artifact already settles the
direction.) The defensible headline is unchanged: throughput is **ANE ≈ GPU**; ANE's consistent, real win is **energy +
exclusivity** (MLX/LiteRT can't target the ANE at all), not raw decode tok/s.

## Context-length sweep + prefill — Qwen3-4B (iso-int4)

How each runtime holds up as the prompt grows (decode-at-depth) and how fast it ingests that prompt (prefill).
Qwen3-4B, **all four runtimes at iso-int4** (no quant confound), iPhone 17 Pro, a short reply after a prompt of
{19, 666, 2681, 3977} tokens, 2 cold/cell.

**Decode-at-depth — the ANE's short-context lead inverts by ~2K.**

| context (prompt tok) | Core AI ANE | Core AI GPU | MLX | LiteRT-LM |
|---|--:|--:|--:|--:|
| 19 | **27.4** | 17.0 | 20.3 | 15.3 |
| 666 | **26.3** | 16.1 | 18.0 | 15.5 |
| 2,681 | 11.6 | 13.0 | **14.6** | 13.7 |
| 3,977 | 10.5 | **11.9** | ✗ ceiling | ✗ ceiling |

![Decode-at-depth — the ANE's short-context lead vanishes by ~2K (Qwen3-4B iso-int4, iPhone 17 Pro)](charts/iphone_context_sweep_decode.png)

ANE is fastest at short context (27.4) but its **static-bucket KV attention degrades steeply** — by 2.7K it is
**last** (11.6), behind MLX 14.6 / LiteRT 13.7 / GPU 13.0; the GPU/MLX/LiteRT paths degrade gently. So the
"ANE is the fastest decoder" caveat is now sharper: it holds **only for short prompts**; at depth the ranking inverts.
(MLX and LiteRT fail at the 3,977-token point — it brushes the 4,096 context ceiling — so that column is Core-AI-only.)

**Prefill — LiteRT-LM's 25-second wall.**

| Qwen3-4B @ 2.7K ctx | Core AI ANE | MLX | Core AI GPU | LiteRT-LM |
|---|--:|--:|--:|--:|
| prefill tok/s (prompt ÷ TTFT) | **671** | 514 | 359 | **~108** |
| TTFT (time to first token) | 4.0 s | 5.2 s | 7.6 s | **24.9 s** |

![Prefill — LiteRT-LM's 25-second wall (Qwen3-4B iso-int4, iPhone 17 Pro)](charts/iphone_context_sweep_prefill.png)

ANE prefills fastest (671 tok/s); **LiteRT-LM is ~5–6× slower (~108 tok/s)** → **25 seconds to the first token** at
2.7K context, vs 4–8 s for the others. Note the split personality: LiteRT's **decode**-at-depth is fine (flat ~14,
the gentlest curve), but its **prefill** is the on-device weakness — the same WebGPU(Dawn)→Metal delegate cost the
BW-ceiling section quantified, now visible as latency a user feels. (LiteRT-LM exposes no prompt-token count, so its
prefill rate is TTFT-derived; 2 cold/cell, some cells single-run after cold-load timeouts.)

## Energy — iPhone 17 Pro, sustained decode (battery-delta, J/token)

The decode tables above are short-chat snapshots. This axis asks the on-device **efficiency** question:
*which runtime delivers the most tokens per joule under a sustained load?* Instrument: `UIDevice.batteryLevel`
(1% steps) sampled across a **600 s `energy` task** that re-prompts to keep the runtime busy; the ~5% drop →
joules via the device pack (iPhone 17 Pro = 16.5 Wh). Whole-system (display + radios + OS included), **±~12% per
run** at 1% resolution — compare *within device* only. Method: [`energy-ios.md`](../methodology/energy-ios.md). **Three
models** — **Llama-3.2-3B**, **DeepSeek-R1-1.5B**, and **Qwen3-4B** (the latter two are Google's own litert-community
distributions) — all four
runtimes (4-bit unless noted; unplugged, screen on, mid-band battery):

**Llama-3.2-3B** (all int4) — adds a peak-memory column (`memoryPeakDuringDecodeMB`):

| Runtime | **J/token** ↓ | avg W | sustained tok/s | (short-chat) | **peak MB** | tokens/Wh |
|---|--:|--:|--:|--:|--:|--:|
| **Core AI GPU** | **0.224** | 4.67 | 21.0 | 19.3 | **732** | 16 058 |
| **Core AI ANE** | **0.225** | 4.62 | 20.6 | 24.2 | 2,577 | 16 019 |
| **MLX** | **0.245** | 4.70 | 19.3 | 34.0 | 3,221 | 14 681 |
| **LiteRT-LM** | **0.324** | 4.31 | 13.6 | 18.4 | 3,263 | 11 120 |

**DeepSeek-R1-1.5B** (Core AI / MLX int4; LiteRT int8* = its litert-community quant):

| Runtime | **J/token** ↓ | sustained tok/s | (short-chat) | **peak MB** | tokens/Wh |
|---|--:|--:|--:|--:|--:|
| **Core AI ANE** | **0.097** | 49.0 | 83.3 | 1,368 | 37,236 |
| **MLX** | **0.105** | 44.8 | 73.0 | 1,434 | 34,315 |
| **Core AI GPU** | **0.132** | 36.2 | 75.9 | **374** | 27,307 |
| **LiteRT-LM** | **0.204** | 24.2 | 30.7 | 1,000 | 17,678 |

**Qwen3-4B** (all **iso-int4** — LiteRT = litert-community's official mixed-int4, so no quant confound):

| Runtime | **J/token** ↓ | sustained tok/s | (short-chat) | **peak MB** | tok/1% |
|---|--:|--:|--:|--:|--:|
| **Core AI ANE** | **0.242** | 17.4 | 29.7 | 3,181 | 2,458 |
| **MLX** | **0.242** | 18.2 | 27.3 | 3,996 | 2,458 |
| **Core AI GPU** | 0.290 | 16.5 | 28.3 | **873** | 2,048 |
| **LiteRT-LM** | **0.725** | 18.0\* | 21.2 | 2,217 | 819 |

\* LiteRT's decode *rate* is competitive (~18) but its **effective** throughput is ~4.7 tok/s — it generated only
**4,096** tokens in the window vs 12,288 for ANE/MLX (re-prefill / overhead-dominated, 865 s gen-time) → **3× worst
tok/1%**, the largest gap of all three models, and at iso-int4 there is no quant excuse.

**Memory (peak during decode) — Core AI GPU is in a class of its own:** 732 MB (Llama) / 374 MB (DeepSeek) / 873 MB
(Qwen3-4B) via mmap-pipelined weights (resident set ~280 MB), vs ~3.2–4.0 GB for MLX & LiteRT and the
weights-resident Core AI ANE (2.6 / 1.4 / 3.2 GB) — a **~4× memory-headroom advantage** the pipelined-mmap path alone
delivers (decisive for fitting larger models on-device).

**Across all three models, Core AI is the most energy-efficient and LiteRT-LM the least** (Llama: Core AI ANE≈GPU
~0.224 vs MLX 0.245 vs LiteRT 0.324; DeepSeek: Core AI ANE 0.097 vs MLX 0.105 vs GPU 0.132 vs LiteRT 0.204;
Qwen3-4B: Core AI ANE 0.242 ≈ MLX 0.242 vs GPU 0.290 vs LiteRT 0.725). All
runtimes drained the same 5% (1 quantum), so **J/token is governed by *sustained* throughput** (joules constant,
tokens differ):
1. **Sustained load reorders the short-chat ranking — MLX is hit hardest.** MLX's 34 tok/s short-chat lead collapses
   to **19.3** under 10 min of thermal load (peak `serious`), *below* Core AI; LiteRT 18.4→13.6; Core AI is the most
   thermally stable (ANE 24→20.6, GPU 19.3→21.0). The energy task is the realistic long-generation regime.
2. **ANE ≥ GPU on energy; the GPU's edge is memory (not energy).** Core AI ANE wins or ties the GPU on J/token at
   every size — DeepSeek-1.5B (ANE 0.097 vs GPU 0.132), Qwen3-4B (0.242 vs 0.290), and a tie on Llama-3B (0.225 ≈
   0.224, where the GPU happened to sustain unusually well). It is **not** cleanly size-dependent. What the GPU wins
   consistently is **memory**: its mmap-pipelined path holds **374–873 MB** vs the ANE's weights-resident **1.4–3.2
   GB** — a ~4× headroom advantage. So **ANE = the efficiency pick, GPU = the memory pick**; whole-system power is
   display/OS-dominated (~4.6 W) so neither has a chip-power edge in the battery number.
3. **The "low-power-but-slow → worst J/token" role belongs to LiteRT here** (lowest draw 4.31 W, but slowest 13.6
   tok/s → worst 0.324) — the same shape the ANE showed *on the Mac* (below).

**Versus the prior energy rows in this repo** (`results/raw/*-energy-*.jsonl`):
- **The Mac pattern flips on the phone.** On **M4 Max**, CoreML/ANE gemma-4-E2B had the *lowest* watts (12.7 W) yet
  the *worst* J/token (**0.478**) — its 33 tok/s kept the package powered longest; MLX (0.240) and llama.cpp (0.247)
  won on speed. **On iPhone that inverts:** Core AI is *best*, because the GPU runtimes pay a steep on-device
  throttling tax (MLX 35→19) and whole-system draw equalizes power. This is exactly the question
  `energy-ios.md` posed — and the answer is **yes, the on-device throughput tax changes the energy winner.**
- **Efficiency is architecture-specific, not runtime-absolute.** For **gemma-4-E2B on iPhone, LiteRT-LM was the
  *best* (0.146 J/token)** — its first-class gemma kernels decode 30.8 tok/s sustained — whereas for Llama-3.2-3B
  (generic path) LiteRT-LM is *worst* (0.324). So "which runtime is greenest" depends on whether the model has
  first-class kernels, mirroring the decode-speed thesis.
- **Model size dominates the absolute number.** qwen3-0.6B on MLX hit **0.066 J/token** (99 tok/s, stayed `fair`) —
  ~4× more efficient than any 3B cell; a smaller model is the biggest energy lever, ahead of runtime choice.

**"How many tokens does 1% of iPhone battery buy?"** Re-cast as the headline metric (tokens ÷ battery-Δ%), the
**battery-efficiency crown flips with the model** — tracking first-class-kernel support, not the runtime:

| Model (LiteRT quant) | **LiteRT-LM** | **Core AI** (ANE / GPU) | **MLX** | greenest |
|---|--:|--:|--:|---|
| Gemma-4-E2B (int4, gemma kernels) | **4,074** | 2,867 (ANE) | 2,560 | **LiteRT-LM** |
| DeepSeek-R1-1.5B (int8*, generic) | 2,917 | **6,144** / 4,506 | 5,662 | **Core AI (ANE)** |
| Llama-3.2-3B (int4, generic) | 1,835 | 2,643 / **2,650** | 2,422 | **Core AI (GPU)** |
| Qwen3-4B (int4, generic) | 819 | **2,458** / 2,048 | 2,458 | **Core AI / MLX** |

![Tokens per 1% of iPhone battery — LiteRT-LM greenest only on Gemma](charts/iphone_tok_per_pct_flip.png)

tokens per 1% battery — the **battery-efficiency *ranking* flips with the model.** For **Gemma — Google's first-class
model — LiteRT-LM is the greenest runtime on Apple's phone (4,074 tok/1%)**. For a **generic model it falls to LAST
and Core AI wins** — confirmed across **three independent non-gemma models**: Google's own **DeepSeek-R1-1.5B** (Core AI
ANE **6,144** = **2.1×** LiteRT's 2,917), **Llama-3.2-3B** (Core AI 2,650 vs LiteRT 1,835), and **Qwen3-4B** —
all four runtimes **iso-int4** — where LiteRT is **3× worst** (Core AI/MLX 2,458 vs LiteRT **819**). The flip is robust to
the obvious confounds — it holds whether LiteRT ships **int4** (Llama, Qwen3-4B) or **int8** (DeepSeek), and whether the rows
are Debug/iOS 26.4.2 (Gemma) or Release/iOS 27.0 (DeepSeek, Llama) — so the cause is the **first-class-kernel gap**,
not a quant or build artifact (LiteRT decodes Gemma at 30.8 tok/s sustained vs DeepSeek 24.2, Llama 13.6). **Even on
a model Google itself distributes via litert-community (DeepSeek-R1), LiteRT-LM is the least battery-efficient of the
four.** (Absolute tok/1% rises for the smaller DeepSeek-1.5B — model size is the biggest lever — so compare the
*ranking* within a row, not magnitudes across rows.)

> ⚠️ Single-run, 1% battery resolution: treat ±0.02–0.03 J/token gaps (ANE vs GPU) as ties; the MLX and LiteRT
> gaps are larger than the error bar. All rows verified `batteryState=unplugged` with a non-nil drop.

## Coverage — LiteRT *traces*, Core AI *reimplements* (out-of-the-box ease vs custom-code ceiling)

Every model here converted to LiteRT with **zero custom code**; Core AI instead needs a hand-written class per
architecture — so before the 2026-06-25 export pass it covered only 6/10, **now 9/10 on Mac** once three more classes
were written. This is a direct consequence of the two converters' designs, not a quality gap. **LiteRT-LM (`litert_torch`) is trace-based and
architecture-agnostic:** `torch.export` captures the HF model's *actual* forward and lowers it to generic ops
(matmul, RMSNorm, real-valued RoPE, softmax-SDPA). It never needs to "know" the architecture — OLMo-2's
QK-norm, SmolLM3's NoPE, Ministral's layout are just *different arrangements of the same generic ops*, so any
traceable model with supported ops works. **Core AI (`coreai-models`) is a per-architecture reimplementation
registry:** hand-written Apple `nn.Module` classes (`macos/{qwen2,qwen3,mistral,gemma3_text,…}.py`) built from
Apple primitives (SDPA/RoPE/RMSNorm) into which HF weights are *loaded* (`_mutate_state_dict` even re-combines
QKV). An architecture with no hand-written class can't export — which is why olmo2/smollm3/ministral3 originally had
none (and can't borrow the `mistral` wrapper: they'd *run* but compute the wrong thing — skipping QK-norm, applying
RoPE to NoPE layers). **The export pass wrote those classes and they work** (Mac 384.4/192.9/186.0; iPhone too),
confirming the gap was "unwritten class," not "unsupported." Two residual walls are deeper than a missing class:
**Phi-4-mini partial rotary** (the class exists and passes HF parity, but the standard full-rotary RoPE path aborts
on both Mac `llm-benchmark` and the iOS pipeline), and the **Gemma3-1B / Ministral-3-3B ANE** cells (the iOS ANE
pipeline can't express Gemma3's dual-RoPE + sliding/full attention, and Ministral needs the multimodal-FP8 loader —
both run fine on the GPU path). Gemma3-1B was the early quick fix because gemma3 *was* already in the registry —
only its config-layout assumption needed relaxing.

Core AI reimplements (rather than traces) to emit a **stateful (KV-cache-as-state), Apple-tuned, memory-mapped
graph** — what makes it fast (320 tok/s, 4 k prefill). **The nuance that matters (and corrects an easy
over-claim): out-of-the-box, LiteRT-LM is *easier* for dense models — zero custom code — but it does NOT have
broader coverage.** Allow custom code and **Core AI's ceiling is far higher**: with a hand-written export class
*and* a custom Metal kernel you reach **and fast-run** MoE, diffusion, SAM / depth-anything / YOLO / efficient-SAM,
audio (whisper / wav2vec2 / clap), T5 on-device — 20+ families in the `coreai-models` zoo. LiteRT-LM can't match
that today: the newest archs (MoE / MLA / SSM) **drop to CPU even after they convert**, and its **GPU kernel layer
is closed** (ML Drift is prebuilt; no external kernel-registration path). So LiteRT-LM's real edge is
**zero-friction dense conversion + portability / cross-hardware reach**, *not* raw breadth — and the high-leverage
asks are to **raise the ceiling: (a) converter fixes for the new archs, (b) open the GPU kernel layer so
contributors can extend it the way they extend Core AI** — alongside closing the dense-path Apple-GPU speed gap
(below). (Of the 10 here — all dense — LiteRT converts 10/10 with **zero per-model code**; Core AI needs a hand-written
class per arch but, once written, also runs **9/10** on Mac. The *ease-for-dense* point is the **marginal cost of a new
architecture** — zero for LiteRT vs a class for Core AI — not a permanent coverage ceiling, as the 2026-06-25 pass showed.)

## Recommendations to the LiteRT-LM team (impact-ordered)
1. **Make weight-only int4 + FLOAT compute (int4→fp16 dequant + fp16 GEMM) actually execute** — today it
   converts but hangs in the WebGPU delegate (`deliverables/litert-float-gpu-hang/`). This is the MLX/Core AI
   compute model and the single highest-impact route to Apple-GPU decode parity.
2. **Native Metal delegate** for Apple (vs WebGPU/Dawn→Metal) to drop the abstraction layer.
3. **Raise the ceiling so contributors can extend LiteRT-LM the way they extend Core AI** — (a) fix the converter
   for the newest archs (MoE / MLA / SSM — today they drop to CPU even after converting) and (b) **open the GPU
   kernel layer** (ML Drift is prebuilt with no external kernel-registration path). This is what currently caps
   LiteRT-LM's on-device model set below Core AI's, despite the easier dense converter.
4. **Lean into LiteRT's real edge** — cross-hardware reach (Qualcomm/NPU on Android) and one portable bundle.
   On Apple GPU, MLX/Core AI are the native floor; parity (not beating) is the realistic target.

## Methodology / repro
**Open harness — reproduce it yourself: https://github.com/john-rocky/apple-silicon-llm-bench** (neutral, one
headless harness for every runtime; every cell traces to raw JSONL). Per-runtime:
- **Core AI:** `uv run coreai.llm.export <hf-id> --platform macOS --compression 4bit --compute-precision
  float16 --experimental`; bench `llm-benchmark --model <bundle> -p 512 -g 1024 -n 5`. Wrapper chosen by HF
  `model_type` (qwen2/qwen3/gemma3/mistral via `MODEL_TYPE_REMAPPING`).
- **MLX:** `mlx-community/*-4bit` (or `mlx_lm.convert -q --q-bits 4 --q-group-size 64`); 512-tok greedy stream.
- **LiteRT-LM:** litert-community `.litertlm` (published quant) or our BOCTAV4 conversions;
  `litert-mac-verify --max-tokens 512 --backend gpu`.
- **iPhone 17 Pro (all runtimes):** the Yardstick app (`ios/BenchmarkApp`, bundle
  `com.iosllmbenchmark.benchmarkapp`) driven headless via `devicectl … --yardstick-autorun --runtime
  {mlx-swift,litert-lm} --model-id <id> --task short-chat`, median of 3 cold launches. Model entries added to
  `ios/BenchmarkApp/Sources/Models/ModelCatalog.swift` (`mlx-community/*` = on-device MLX download;
  `litert-community/*` = on-device LiteRT download or side-load; `litert-local/*` = side-loaded `.litertlm`).
- **Raw decode cells (this repo):** `results/raw/2026-06-24-litert-community-crossframework/decode-cells.jsonl`
  — one JSON row per model×framework×device: `{model, group, framework, device, quant, decode_tps, prefill_tps, …}`.
