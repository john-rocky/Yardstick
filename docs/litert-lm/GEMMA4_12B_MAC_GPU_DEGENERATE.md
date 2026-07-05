# Bug report — `gemma-4-12B-it` LiteRT-LM emits degenerate output on the Mac GPU (ML Drift) backend

**Repo:** `litert-community/gemma-4-12B-it-litert-lm` (file `gemma-4-12B-it.litertlm`, int4).
**Runtime:** LiteRT-LM, GPU backend (ML Drift / WebGPU) on Apple Silicon (macOS).
**Machine:** Mac Studio, Apple M4 Max (40-core GPU), 128 GB, macOS 27β. Date: 2026-07-06.
**Severity:** high — the model is unusable on this backend (produces no valid text), though decode *throughput* is normal.

## Symptom

Greedy decoding produces a **degenerate token stream** — long runs of the same token
(`0000000…`, with occasional `1`/`2`), independent of prompt. Both driver paths reproduce it:

- Low-level `Engine` API, explicit `--backend gpu`, greedy (topK40/topP0.95/temp0): `OUTPUT: [2221222122212221]`, EOS after ~16 tokens.
- `LiteRTChat` API, `--greedy` (topK1/temp0), 640 tokens: `OUTPUT: [00000000…0001000…]`.

Decode throughput is healthy (**~48.5 tok/s**), so the graph runs and streams — only the
**logits/sampling result is wrong**. The model is **GPU-only** (`--backend cpu` →
`INVALID_ARGUMENT: Model requires one of [gpu] but Main backend is CPU`), so a same-file
CPU/GPU A/B isn't possible; we triangulated instead.

## Triangulation — the bug is specific to the 12B build, not the platform

Same machine, same LiteRT-LM Mac GPU backend, same driver, same prompt ("Explain memory
bandwidth in one sentence."), greedy:

| Model (.litertlm, Mac GPU) | Output | Decode tok/s |
|---|---|---|
| `litert-community/gemma-4-**E2B**-it` (same family, smaller) | ✅ **coherent** — "Memory bandwidth is the rate at which data can be transferred between the CPU/GPU" | 118.6 |
| `litert-community/**Qwen3-4B**` (different family) | ✅ **coherent** — "`<think>` Okay, the user wants a one-sentence explanation…" | 106.0 |
| `litert-community/gemma-4-**12B**-it` | ❌ **degenerate** (`0000…`/`2221…`) | 48.5 |

⇒ The Mac GPU (ML Drift) path is **correct for gemma-4-E2B and for Qwen3-4B**, and breaks
**only for the gemma-4-12B build**. So this is **not** a general Mac-GPU issue and **not** a
general gemma-4 issue — it is specific to the **gemma-4-12B `.litertlm`** (its conversion,
or a kernel that only mis-fires at the 12B's shapes/dims on ML Drift).

## Likely area (for triage)

The 12B differs from E2B in ways that stress GPU kernels at larger dims: 48 layers, hidden
3840, **16 attention heads with dual head_dim (sliding 256 / full 512)**, `attention_k_eq_v`
(full-attention layers have 1 KV head, value = raw pre-norm `k_proj`), attn scale = 1.0 with
per-head Q/K-RMSNorm, dual RoPE (θ=1e4 full-rotate 256 / θ=1e6 partial-0.25 over 512), final
logit softcap 30. A degenerate-from-first-token, throughput-normal failure points at a
**numeric/kernel issue on the GPU decode path** (e.g. an overflow/precision problem in the
full-attention 512-head_dim path or the softcap, or a bad weight/scale in the 12B conversion)
rather than orchestration. Note Apple's own Core AI stack hit a *different* but analogous
12B-only GPU issue (16-head × 512 Q overflowing an MPSGraph scratch heap) — the 12B's
16×512 attention geometry is a repeated trouble spot across GPU runtimes.

## Repro

```bash
# GPU-only model; degenerate:
litert-mac-verify gemma-4-12B-it.litertlm "Explain memory bandwidth in one sentence." \
  --max-tokens 32 --backend gpu          # OUTPUT: [2221222122212221]

# same backend/machine, coherent (controls):
litert-mac-verify gemma-4-E2B-it.litertlm  "<same prompt>" --max-tokens 32 --greedy
litert-mac-verify qwen3_4b_mixed_int4.litertlm "<same prompt>" --max-tokens 32 --greedy
```

Raw console logs: `results/raw/2026-07-05-gemma4-12b-mac/` (12B) and
`scratchpad/litert_triangulate.log` (E2B + Qwen3 controls).

## Ask

Please check whether the published `gemma-4-12B-it.litertlm` (a) reproduces on your ML Drift /
Android GPU path or is Mac-GPU-only, and (b) whether re-converting fixes it. The model-card
benchmarks imply it was validated on device — so a Mac-GPU-specific numeric regression in a
12B-only kernel is the leading hypothesis.
