# results/quality — GSM8K reports behind the README table's quality column

Written by `scripts/parity_gsm8k.py` (one identical protocol per arm — see the script
docstring and `methodology/fairness-rules.md`). File = one arm × build × mode:

- `gsm8k_litertlm-gemma4-e2b-wna8o8-measured.json` — LiteRT wNa8o8, 85.0 (table row)
- `gsm8k_mlx-gemma4-e2b-qat-optiq4[.json / -thinking]` — MLX OptiQ 91.0 / thinking 90.0
- `gsm8k_mlx-gemma4-e2b-ptq4.json` — MLX PTQ 84.0
- `gsm8k_coreai-gemma4-e2b-q40-engine020[.json / thinking]` — Core AI own-int4 88.0 / thinking 92.0
- `gsm8k_llamacpp-gemma4-e2b-q4km.json` — llama.cpp Q4_K_M 76.0
- `gsm8k_cactus-gemma4-e2b-cq4-uncalibrated.json` — Cactus demoted CQ4 87.0 (table row)
- `gsm8k_cactus-gemma4-e2b-cq4.json` — Cactus shipped default 3.0 (footnote row)
- `gsm8k_cactus-cq254/cq326-probe.json` — n=25 lineage probes (4%/16%)
- `gsm8k_coreai-gemma4-e2b-official-wna8o8.json` — wNa8o8-on-fp16 48.0 (note-2 finding)
- `fakequant_gsm8k_*.json` — the int8-static-activation falsification ladder (note 2)

Provenance: measured across the 2026-07 campaign in sibling repos (hf-to-litertlm /
litertlm-convert) with this same harness lineage; copied here 2026-07-20 when the harness
became canonical in-repo. New runs land here directly.
