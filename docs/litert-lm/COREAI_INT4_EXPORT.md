# Core AI — matched-INT4 export (decoupling compute unit from quant)

> **2026-08-27 recheck (upstream asked "resolved on latest beta?"):** the SIGSEGV is gone —
> on macOS 26A5416b / coreai-build 3600.82.1 (MetalToolchain v27.1.5237.12) with coreai-torch
> 0.4.2 + coreai-core 1.0.0b2, three compiles of the regenerated H1-b artifact (June-era
> checkout and apple/main f43b6da both) exit 0 in ~5 min. But the ANE pre-compile now fails
> (`ANECCompileOffline() failed: aneCompileStatus=1`, empty ErrorList) and coreai-build emits a
> **GPU-only `.aimodelc` (0 ANE regions) with exit 0** — a silent fallback. Palettized control
> same day: still 31/31 ANE regions. **The negative result below stands** (ANE + linear-INT4
> remains unexpressible); only the failure mode changed (segfault → silent GPU fallback).
> Reported on apple/coreai-models#55 (issuecomment-5434952602); compile logs pinned in
> `~/code/coreai/coreai-models/exports/qwen3_0_6b_ios_int4linear/`.
>
> **Status: RESOLVED (2026-06-18) — negative result.** A matched **INT4-*linear***
> export that runs on the ANE is **not achievable** on this toolchain: the ANE
> pre-compiler **SIGSEGVs** on a static-shape program carrying linear blockwise-INT4
> weights, while the byte-for-byte-identical structure with palettized weights
> compiles cleanly to the ANE. The entanglement is a **platform constraint**, with one
> nuance: the *mixed* 4/8 split is a preset choice (a pure **uniform 4-bit** ANE export
> is reachable — via palettization, not linear INT4). See **Results** below; the README
> entanglement caveat stays.

## The problem (why the Core AI rows aren't a clean engine A/B)

In Core AI the **compute unit is fixed by the export shape**, and the two shapes
ship **different quantisation**, so the engine axis and the quant axis are entangled:

| Export | Shape | Quant | On disk | Lands on |
| --- | --- | --- | ---: | --- |
| dynamic (`--platform macOS`) | dynamic ctx | **INT4 (dynamic)** | ~327 MB | GPU (`coreai-pipelined`) |
| static (`--platform iOS`) | fixed ctx 4096 | **mixed 4/8-bit palettized** | ~434 MB | ANE (`static-shape`) |

Consequences:

1. **Core AI GPU vs ANE** confounds engine with quant — GPU is both a different
   engine *and* lighter (pure INT4 vs mixed 4/8). The 71→180 (GPU) vs ~50 (ANE)
   gap can't be attributed to the engine alone.
2. **Core AI vs LiteRT-LM** at "4-bit" hides a real byte-budget gap (327 vs 498 MB);
   a raw tok/s headline partly credits Core AI for carrying fewer bytes.

The package README discloses this (per-row quant + on-disk size + an explicit
"shipped-config, not engine A/B" caveat) but does **not** equalise it. Equalising
needs a matched-INT4 export — the experiment below.

## Goal

Produce a Core AI Qwen3-0.6B bundle that runs on the **ANE at pure INT4** (matched
to the GPU export's quant), so that:

- **Core AI GPU(INT4) vs ANE(INT4)** becomes a clean engine A/B (quant held equal), and
- a Core AI **INT4** row is byte-comparable to the LiteRT-LM INT4 row.

…**or** a documented negative result that the ANE path *requires* palettised
mixed-4/8 (with the compiler error as evidence), which establishes the entanglement
as a platform constraint rather than a preset choice — itself a publishable finding.

## Hypotheses to test

- **H1 — preset, not platform.** The mixed-4/8 on the static export comes from the
  export *recipe* (palettisation pass for ANE-friendliness), not a hard ANE
  requirement; a quant override yields a pure-INT4 static export.
- **H2 — platform constraint.** The ANE legaliser requires palettised weights; a
  pure-INT4 static export either fails to compile (`MPS→ANEC failed`, cf.
  [`methodology/coreai-engine-speed.md`](../../methodology/coreai-engine-speed.md))
  or silently falls back to GPU — in which case "ANE + INT4" is not expressible.

## Experiment plan (Mac-side; no device needed until the bench step)

Toolchain: `~/code/coreai/coreai-models` (Apple's `coreai.llm.export` + `xcrun
coreai-build compile`). Pin every artifact (see the lowering instability in
[`coreai-export-lowering.md`](../../methodology/coreai-export-lowering.md) — an
`.aimodel` is a build artifact, not a pure function of the recipe; stamp OS + date).

1. **Inspect the recipe.** Find where the static/iOS export selects mixed-4/8
   palettisation vs the dynamic export's INT4 (registry preset is
   `("qwen3-0.6b", …, "4bit", …)` for both — so the split happens in the
   platform/lowering path, not the bit-width field). Identify any quant/palettise
   override flag.
2. **Attempt a pure-INT4 static export** (H1). Export `--platform iOS` with INT4
   forced (no palettisation). Record the produced quant string + on-disk size.
3. **AOT-compile for the device** and **verify the engine actually used** — do not
   assume. `xcrun coreai-build compile … --preferred-compute neural-engine
   --architecture h18p`; confirm it compiles for ANEC (not a silent GPU fallback)
   and that `EngineFactory` selects `static-shape`.
4. **If ANE rejects INT4 (H2):** capture the exact compiler error, confirm the
   GPU-fallback behaviour, and write it up as the negative result.
5. **Assemble the loadable bundle** (compiled `.aimodelc` + `tokenizer/` +
   `metadata.json` with `assets.main`) ready for side-load, exactly as
   `scripts/bench_coreai_iphone.sh::assemble` does.

## Acceptance criteria

- A bundle whose recorded `quantization` is **INT4** *and* whose verified engine is
  **`static-shape` (ANE)** — or a reproduced compiler-rejection/fallback log proving
  it can't be. Either outcome closes this doc.
- Artifact provenance stamped (OS version, date, wheel versions, `strings main.mlirb`
  op evidence) like the other Core AI artifacts.

## Results

**Outcome: negative (H2, sharpened).** "ANE + INT4" is expressible only as
**palettized** 4-bit (LUT); the GPU export's **linear** INT4 (`blockwise_shift_scale`)
**crashes the ANE pre-compiler**. A Core AI GPU(INT4) vs ANE(INT4) A/B at the
*identical quant scheme* is therefore not possible. The *mixed* 4/8 split, however, is
a preset choice — a leaner **uniform 4-bit** ANE export is reachable and is provided.

### What was tried (Mac-side, h18p), and what each produced

Each artifact is pinned with its recorded quant string, on-disk `.aimodel` size, and
the **compute unit the AOT compiler actually targeted** — verified, not assumed, by
inspecting the compiled `.aimodelc`: `…_ANE_region_*` segments under
`main-h18p-delegates/MPSGraph/…/binary_0.llir.bundle/graph` ⇒ ANE; a bare `MPSGraph`
delegate with no such segments ⇒ GPU.

| Export | Quant (MLIR op) | `.aimodel` | Compiled for `neural-engine` h18p |
|---|---|---:|---|
| baseline GPU — `--platform macOS` (`4bit`) | linear INT4 — `blockwise_shift_scale` | 321 MB | **GPU** MPSGraph, 0 ANE regions \* |
| baseline ANE — `--platform iOS` (registry `mixed_4bit_8bit`) | palettized 4/8 — `lut_to_dense` | 434 MB | **ANE** — 31 regions, 0 non-ANE |
| **H1-a** — `--platform iOS --compression 4bit_weight_palettized_group32` | palettized uniform-4bit — `lut_to_dense` | 389 MB | **ANE** — 31 regions, 0 non-ANE |
| **H1-b (goal)** — iOS static + MLIR `quantize_weights` INT4 | linear INT4 — `blockwise_shift_scale` | 416 MB | **CRASH** — ANE pre-compiler SIGSEGV, no output |

\* The GPU `.aimodel` was *also* compiled with `--preferred-compute neural-engine`: it
still produced a bare GPU MPSGraph delegate with **0 ANE regions** (byte-different from
the `gpu` build, but no ANE segments). `--preferred-compute` is a no-op for a
dynamic-shape program — the engine is fixed by program **structure** (single dynamic
`main` → pipelined/GPU; static chunked `extend_<ctx>_<q>` → static-shape/ANE), exactly
as `methodology/coreai-ios.md` states. **The existing GPU INT4 export cannot be moved
onto the ANE by recompilation.**

### The decisive (controlled) comparison

H1-b isolates the variable. The linear-INT4 static `.aimodel` has the **identical 34
static-shape functions** as the H1-a palettized export (`extend_{256,512,1024,2048,4096}_{8,16,64}`,
`prompt_opt_*`, `gather_embeddings_*`, `load_embeddings`; `coreai-build inspect --io`).
The **only** difference is the weight encoding — `lut_to_dense` (k-means palettization,
dtype `UInt4`) vs `blockwise_shift_scale` (linear symmetric INT4, dtype `Int4`):

- palettized (LUT) static program → compiles to ANE (31 regions, 0 non-ANE);
- linear-INT4 static program → `coreai-build` **SIGSEGVs** in ANE pre-compilation,
  emits no diagnostic, writes no `.aimodelc`.

So the ANE legaliser accepts **palettized** weights but not **linear blockwise-INT4**
weights, everything else held equal.

### Crash evidence (reproducible, 2/2 runs)

`xcrun coreai-build compile … --preferred-compute neural-engine --architecture h18p` on
the linear-INT4 static `.aimodel` terminates with **`SIGSEGV` (exit 139)** after ~5 min
at 100% CPU, no output. Reports:
`~/Library/Logs/DiagnosticReports/coreai-build-2026-06-18-{102619,103230}.ips`.

```
EXC_BAD_ACCESS (SIGSEGV) — KERN_INVALID_ADDRESS   thread: MPSGraphExecutable_queue
  0  libobjc.A.dylib                    objc_release
  1  MetalPerformanceShadersGraph_host  GPU::anePreCompileBinary(MPSGraphExecutable*, …)
  2  MetalPerformanceShadersGraph_host  BaseModuleRef::compileAndLoadANE()
  3  MetalPerformanceShadersGraph_host  -[MPSGraphExecutable specializedModuleWithDevice:shapedEntryPoints:compilationDescriptor:…]
```

The fault is in MPSGraph's ANE binary pre-compilation (`compileAndLoadANE →
anePreCompileBinary`), i.e. legalising the dequant-then-matmul (`blockwise_shift_scale`)
form for the ANE. The GPU INT4 baseline lowers to the *same* `blockwise_shift_scale`
form (the macOS-27β "slow" lowering; see `coreai-export-lowering.md`) and runs fine on
the GPU MPSGraph path — it is only the **ANE** pre-compiler that crashes on it. Filed
upstream: [apple/coreai-models#55](https://github.com/apple/coreai-models/issues/55)
(with the two `.ips` crash reports offered).

### Verdict on the hypotheses

- **H1 (preset, not platform) — partly TRUE.** The *mixed* 4/8 is a recipe choice:
  swapping the registry `qwen3_0_6b_mixed_4bit_8bit.yaml` for the uniform
  `4bit_weight_palettized_group32` preset gives a **pure 4-bit** ANE export (389 vs
  434 MB), still 100% ANE. The byte-budget confound can be reduced.
- **H2 (platform constraint) — TRUE for the *scheme*.** "ANE + INT4" exists only as
  **palettized** 4-bit, never as the GPU export's **linear** INT4 (that crashes the ANE
  pre-compiler). The closest matched A/B is GPU(linear-4bit) vs ANE(palettized-4bit):
  bit-width-matched, scheme-different by platform necessity.

### Structural caveat for "byte-comparable"

Even the linear-INT4 static `.aimodel` (416 MB) is **not** byte-comparable to the
dynamic GPU export (321 MB): the iOS/static structure gathers tokens on a CPU front-end
and keeps the embedding table **fp16** (`load_embeddings`/`gather_embeddings`), whereas
the dynamic GPU forward quantises the tied embedding/head to INT4. That fp16 embedding
is the bulk of the ~95 MB gap. A static ANE export is never byte-identical to the
dynamic GPU export even at matched transformer quant.

### Recommendation + deliverable

The clean GPU-vs-ANE A/B at *identical* quant scheme is unobtainable. The best available
improvement is to swap the shipped ANE row from **mixed 4/8** to **uniform 4-bit
palettized** (389 MB `.aimodel`): it drops the mixed-4/8 byte confound and still runs
entirely on the ANE. A side-load bundle is assembled and ready:

```
~/code/coreai/coreai-models/exports/qwen3_0_6b_ane_pure4bit/
  qwen3_0_6b_ios_pure4bit.h18p.aimodelc   (compiled; 31 ANE regions, 0 non-ANE)
  tokenizer/  +  metadata.json (assets.main → the compiled .aimodelc)
```

Promote with `COREAI_CONFIG=Release scripts/bench_coreai_iphone.sh` (3 iso-cold
launches). Keep the README's per-row quant + size disclosure; the linear-vs-LUT scheme
difference between the GPU and ANE rows is a documented platform constraint, not a
preset that can be equalised.

### Provenance

- **macOS** 27.0 (26A5353q), **2026-06-18**, Mac16,9 (M4 Max).
- **coreai-build** Metal toolchain v27.1.5194.15, build 3600.67.5.8.1; device arch **h18p**.
- Wheels: **coreai-core 1.0.0b1, coreai-torch 0.4.0, coreai-opt 0.2.0**; torch 2.11.0
  (active), transformers 5.5.0.
- Op/engine evidence via `coreai-build inspect <model> --ops/--io/--compute/--storage`,
  `strings main.mlirb`, and `…_ANE_region_*` segment presence in the `.aimodelc`.
- Lowering instability applies (`coreai-export-lowering.md`): the GPU INT4 baseline here
  is the macOS-27β "slow" form (141 `constexpr_blockwise_shift_scale` + `ParametrizedLinear`).
  All artifacts pinned under `exports/qwen3_0_6b_*` (git-ignored).

### Reproduction

```bash
cd ~/code/coreai/coreai-models
# GPU dynamic INT4 (linear)
uv run coreai.llm.export qwen3-0.6b --platform macOS --output-name qwen3_0_6b_dynamic_int4
# ANE static, pure uniform 4-bit palettized (H1-a) — overrides the registry mixed-4/8 YAML
uv run coreai.llm.export qwen3-0.6b --platform iOS \
  --compression 4bit_weight_palettized_group32 --output-name qwen3_0_6b_ios_pure4bit
# CLI coupling that blocks the obvious linear-INT4-on-iOS route:
uv run coreai.llm.export qwen3-0.6b --platform iOS --compression 4bit
#   -> RuntimeError: macOS quantization preset provided, but platform is iOS.
# H1-b (goal): build the iOS static program with NO torch compression, then apply the
#   MLIR primitive coreai_opt.coreai_utils.quantize_weights(program, dtype=INT4,
#   qscheme=SYMMETRIC, granularity=PER_BLOCK, block_size=32) — the same call the
#   diffusion pipeline's apply_mlir_quantization uses — before save_asset. Then:
xcrun coreai-build compile exports/qwen3_0_6b_ios_int4linear/qwen3_0_6b_ios_int4linear.aimodel \
  --platform iOS --preferred-compute neural-engine --architecture h18p --output /tmp/out
#   -> SIGSEGV in MPSGraph anePreCompileBinary (exit 139); no .aimodelc produced.
```
