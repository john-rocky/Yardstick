# int4 decode GEMV on Apple GPUs — a K-sliced baseline and one-variable variants

Companion experiment to [`docs/metal-profile-m4max-0170.md`](../../docs/metal-profile-m4max-0170.md).
Both programs are self-contained Swift + embedded MSL, GPU-timed (command-buffer GPU start/end),
run on the whole per-token weight set of a model (every layer's 7 matrices, distinct buffers), so
nothing is cache-resident across layers.

- `gemv_variants.swift` — **the decode experiment.** Kernel `L` is an int4 GEMV in the K-sliced,
  threadgroup-reduction style common to WGSL delegates (4 K × 4 channels of nibbles per 8-byte
  word, per-32-K half4 scale and zero, 16 channel-groups × 16 K-slices per threadgroup,
  fp16-arithmetic nibble unpack, threadgroup-memory reduction); its whole-token time matches the
  GEMV family measured inside LiteRT-LM 0.17.0 by ablation within 4–12%, which is what makes it a
  usable baseline. Function constants switch one thing at a time: integer unpack, no zero-point,
  16-byte loads, 4/8/32/64 K-slices. `C` is a row-major `[N][K]` one-simdgroup-per-row GEMV with
  `simd_sum` reduction (integer or fp16-arithmetic unpack). `--ds` runs
  DeepSeek-R1-Distill-Qwen-1.5B shapes instead of Qwen3-4B.
  ```
  swiftc -O gemv_variants.swift -o gemv_variants && ./gemv_variants 20 && ./gemv_variants 20 --ds
  ```
- `gemv_bench.swift` — a two-pass (dequantize to an fp16 buffer, then fp16 GEMV) vs fused
  comparison, written before the dispatch dump showed that steady decode does not run a two-pass
  path. Kept because it documents why a DRAM-round-trip model does not describe decode.

Output of the runs that the doc quotes: `results/raw/2026-09-08-m4max-metal-profile-0170/gemv_variants.txt`
and `gemv_bench.txt`.
