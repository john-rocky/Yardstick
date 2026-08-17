# Regression report — litert-lm v0.16.0-mac (2026-08-17)

```
/opt/homebrew/opt/python@3.14/bin/python3.14 /Users/majimadaisuke/Downloads/ios-llm-benchmark/scripts/regression_diff.py device --baseline campaign:flat --candidate campaign:2026-08-17-mac-litert-v0160 --anchors matrices/anchors.cells --no-rebuild --engine-under-test litert-lm --json-out /Users/majimadaisuke/Downloads/ios-llm-benchmark/results/regression-reports/2026-08-17-litert-lm-v0.16.0-mac/verdicts.json
```

```

== device / decode_tps ==
IMPROVED         Mac16,9 litert-lm litert-community/Qwen3-0.6B short-chat warm  270.4 (n=4, spread 1%) -> 309.9 (n=3, spread 1%)  +14.6%  [anchor-normalized +14.1% via mlx-swift mlx-community/Qwen3-0.6B-4bit short-chat; raw cross-session delta is informational]
INFO-ONLY        Mac16,9 litert-lm litert-community/Qwen3-0.6B short-chat cold  271.1 (n=1, spread 0%) -> 311.3 (n=1, spread 0%)  +14.8%  [cross-session — do not pool; anchor unavailable (no usable anchor (need n>=2, in-spread, runtime != engine under test)); use a same-session A/B for a verdict]
OK               Mac16,9 litert-lm litert-community/gemma-4-E2B-it-litert-lm short-chat warm  155.9 (n=4, spread 1%) -> 156.0 (n=3, spread 0%)  +0.0%  [quant label: INT4 (QAT) vs wNa8o8 (int2/int4/int8 + int8 activations, QAT)]  [anchor-normalized -0.3% via mlx-swift mlx-community/Qwen3-0.6B-4bit short-chat; raw cross-session delta is informational]
INFO-ONLY        Mac16,9 litert-lm litert-community/gemma-4-E2B-it-litert-lm short-chat cold  133.0 (n=1, spread 0%) -> 128.9 (n=1, spread 0%)  -3.1%  [quant label: INT4 (QAT) vs wNa8o8 (int2/int4/int8 + int8 activations, QAT)]  [cross-session — do not pool; anchor unavailable (no usable anchor (need n>=2, in-spread, runtime != engine under test)); use a same-session A/B for a verdict]
INFO-ONLY        Mac16,9 mlx-swift mlx-community/Qwen3-0.6B-4bit short-chat warm  553.8 (n=4, spread 3%) -> 555.9 (n=2, spread 1%)  +0.4%  [cross-session — do not pool; anchor unavailable (arm not under test (litert-lm) — raw cross-session is the drift signal); use a same-session A/B for a verdict]
INFO-ONLY        Mac16,9 mlx-swift mlx-community/Qwen3-0.6B-4bit short-chat cold  561.6 (n=1, spread 0%) -> 556.0 (n=1, spread 0%)  -1.0%  [cross-session — do not pool; anchor unavailable (arm not under test (litert-lm) — raw cross-session is the drift signal); use a same-session A/B for a verdict]

== device / prefill_tps ==
UNRELIABLE       Mac16,9 mlx-swift mlx-community/Qwen3-0.6B-4bit short-chat warm  1740.9 (n=4, spread 42%) -> 2037.9 (n=2, spread 2%)  +17.1%  [spread > 5% — throw out]
INFO-ONLY        Mac16,9 mlx-swift mlx-community/Qwen3-0.6B-4bit short-chat cold  31.3 (n=1, spread 0%) -> 611.1 (n=1, spread 0%)  +1850.8%  [cross-session — do not pool; anchor unavailable (arm not under test (litert-lm) — raw cross-session is the drift signal); use a same-session A/B for a verdict]

== device / ttft_ms ==
UNRELIABLE       Mac16,9 litert-lm litert-community/Qwen3-0.6B short-chat warm  46.5 (n=4, spread 9%) -> 59.0 (n=3, spread 0%)  +26.9%  [spread > 5% — throw out]
INFO-ONLY        Mac16,9 litert-lm litert-community/Qwen3-0.6B short-chat cold  64.0 (n=1, spread 0%) -> 194.0 (n=1, spread 0%)  +203.1%  [cross-session — do not pool; anchor unavailable (no usable anchor (need n>=2, in-spread, runtime != engine under test)); use a same-session A/B for a verdict]
UNRELIABLE       Mac16,9 litert-lm litert-community/gemma-4-E2B-it-litert-lm short-chat warm  37.0 (n=4, spread 16%) -> 40.0 (n=3, spread 10%)  +8.1%  [quant label: INT4 (QAT) vs wNa8o8 (int2/int4/int8 + int8 activations, QAT)]  [spread > 5% — throw out]
INFO-ONLY        Mac16,9 litert-lm litert-community/gemma-4-E2B-it-litert-lm short-chat cold  337.0 (n=1, spread 0%) -> 331.0 (n=1, spread 0%)  -1.8%  [quant label: INT4 (QAT) vs wNa8o8 (int2/int4/int8 + int8 activations, QAT)]  [cross-session — do not pool; anchor unavailable (no usable anchor (need n>=2, in-spread, runtime != engine under test)); use a same-session A/B for a verdict]
UNRELIABLE       Mac16,9 mlx-swift mlx-community/Qwen3-0.6B-4bit short-chat warm  11.5 (n=4, spread 61%) -> 10.0 (n=2, spread 0%)  -13.0%  [spread > 5% — throw out]
INFO-ONLY        Mac16,9 mlx-swift mlx-community/Qwen3-0.6B-4bit short-chat cold  613.0 (n=1, spread 0%) -> 35.0 (n=1, spread 0%)  -94.3%  [cross-session — do not pool; anchor unavailable (arm not under test (litert-lm) — raw cross-session is the drift signal); use a same-session A/B for a verdict]

wrote /Users/majimadaisuke/Downloads/ios-llm-benchmark/results/regression-reports/2026-08-17-litert-lm-v0.16.0-mac/verdicts.json (14 verdicts)

```
