# Superseded: Gemma-4-E2B short-chat/quality captures (2026-05-28 era)

Superseded by `results/raw/2026-07-18-gemma4-bestquant/` (imported into the flat convention
alongside this note) for two reasons, per the fairness rules:

1. **Quant confounding** — these cells ran each arm on a different checkpoint quality class
   (MLX and llama.cpp on PTQ, LiteRT on QAT) with the build unstated per row.
2. **Cross-session pooling ban** — the 2026-07-18 re-captures are a different session/iOS
   build; the same binary has moved 25%+ across sessions before, so old and new runs must
   never share a median.

The `energy-tg128` captures are a different task and are NOT superseded; they remain in
`results/raw/`. These files stay for the audit trail.
