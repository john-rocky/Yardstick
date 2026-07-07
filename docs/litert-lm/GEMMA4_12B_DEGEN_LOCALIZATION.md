# Localizing the gemma-4-12B LiteRT Mac-GPU degenerate-output bug (#2724) — progress notes

Working notes for the root-cause of [LiteRT-LM#2724](https://github.com/google-ai-edge/LiteRT-LM/issues/2724)
(gemma-4-12B-it `.litertlm` emits degenerate output on the ML Drift GPU; E2B & Qwen3-4B are fine).
**Not yet posted to the issue** — post the bisect result once the first-diverging op is pinned.
Method mirrors the established Apple-GPU parity work (`litert-apple-gpu-float-crash-and-parity.md`,
which pinned #2716 to `ml_drift::EmbeddingLookupOperationParser::Parse`). Constraint: **ml_drift is
prebuilt / closed** (0 `ml_drift` paths in public `google-ai-edge/LiteRT`) → localize by behaviour
(CPU/fp32 reference vs GPU, per-layer intermediate divergence), not by reading delegate source.

## Confirmed (from the actual decode graph)

Extracted the decode TFLite (`litertlm_peek_main --dump_files_dir` → `Section5_..._prefill_decode.tflite`,
5.6 GB) and parsed its op graph (`ai_edge_litert.schema_py_generated`):

- **Attention QK^T / A·V are all fp16 BATCH_MATMULs** — 30 in the full-attention path (head_dim 512),
  160 in the sliding path (head_dim 256). e.g. sliding QK^T = `Q[1,8,2048,256] × K[1,8,32771,256]`.
- **CPU (XNNPACK) cannot even run the decode graph**: `batch_matmul.cc: lhs type == FLOAT32||INT8||INT16
  was not true` at node 0 — the CPU batch-matmul kernel rejects an **fp16 activation lhs**. So the graph
  is designed to run its attention matmuls in fp16, which only the GPU (ML Drift) path does. A plain
  CPU reference is therefore **not** available without an fp32 rewrite.
- **Softcap is a single TANH at the final logits only** (output `[1,1,262144]`); there is **no per-layer
  attention softcap** in this graph. ⇒ softcap can only corrupt the final logits, not drive the
  from-first-token repetitive degeneration — **hypothesis "softcap" downgraded.**
- Graph: 48 decoder layers, dual head_dim (full 16×512 / sliding 8×256), 7171 fp16 + 10767 fp32 tensors.

## Correction to an earlier over-claim

An earlier note guessed "**fp16 overflow in the 512-head_dim full-attention**." That does **not** hold up:
**E2B's full-attention is also head_dim 512** (E2B = 8 heads × 512; 12B = 16 heads × 512) and E2B runs
correctly. Gemma also uses **QK-norm + attn scale 1.0 specifically to bound score magnitude**, so 512-wide
sums need not overflow fp16. Head_dim alone does **not** distinguish the working E2B from the broken 12B.

**Genuinely 12B-specific factors** (candidates for the real cause): full-attention **head count 16 (vs
E2B's 8)**, **48 layers** (deeper accumulation), and the **`gemma4_unified` multimodal architecture**
(encoder-free, audio encoder present) vs E2B's PLE E-series arch.

## Status: structure mapped, mechanism NOT yet pinned

We have confirmed *what the graph is* (fp16 attention, final-only softcap) but not *where the 12B path
first goes wrong*. That needs the bisect below.

## Next step — the decisive experiment (bisect-by-intermediate)

1. **fp32 reference.** Reconstruct the decode forward in fp32 (adapt `extract_gemma4_mixedbit.py` +
   `gemma4_mixedbit_oracle.py` from the E2B/PLE case to the 12B `gemma4_unified` arch — the section
   indices and arch differ). If the fp32 reference is **coherent + matches HF**, the weights/graph are
   correct ⇒ the bug is the **ml_drift GPU fp16 runtime, not the conversion** (rules out an entire class).
2. **Per-layer divergence.** Edit the decode TFLite to expose a per-layer hidden state as an extra output,
   run it on the **GPU (ML Drift via litert-mac-verify/Engine)**, and binary-search the **first layer whose
   GPU hidden diverges** from the fp32 reference. Then descend to that layer's per-op (QK^T vs A·V vs norm vs
   RoPE) and classify the failure: **inf/nan = fp16 range**, **finite-but-wrong = a miscompiled kernel for the
   12B's dims (16-head / 48-layer)**, isolating it for the runtime team.

## Repro assets (local, session scratch)
- Decode graph: `Section5_..._prefill_decode.tflite` (from the published `.litertlm` via `litertlm_peek_main`).
- Op-graph parser + fp16-batch-matmul classifier: ad-hoc `ai_edge_litert.schema_py_generated` scripts.
- Established harnesses to adapt: `litertlm-convert/scripts/{extract_gemma4_mixedbit,gemma4_mixedbit_oracle,
  parity_cpu_generic}.py`.
