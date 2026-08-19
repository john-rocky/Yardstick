# Regression report — litert-lm v0.16.0-ios (2026-08-19)

```
/opt/homebrew/opt/python@3.14/bin/python3.14 /Users/majimadaisuke/Downloads/ios-llm-benchmark/scripts/regression_diff.py device --baseline campaign:2026-07-14-iphone-full-warm --candidate campaign:2026-08-19-iphone-litert-v0160 --anchors matrices/anchors.cells --no-rebuild --engine-under-test litert-lm --json-out /Users/majimadaisuke/Downloads/ios-llm-benchmark/results/regression-reports/2026-08-19-litert-lm-v0.16.0-ios/verdicts.json
```

```

== device / decode_tps ==
OK               iPhone18,1 litert-lm litert-community/Qwen3-0.6B short-chat warm  119.5 (n=3, spread 0%) -> 121.5 (n=3, spread 1%)  +1.7%  [anchor-normalized -0.8% via mlx-swift mlx-community/Qwen3-0.6B-4bit short-chat; raw cross-session delta is informational]
INFO-ONLY        iPhone18,1 litert-lm litert-community/Qwen3-0.6B short-chat cold  119.8 (n=1, spread 0%) -> 122.5 (n=1, spread 0%)  +2.3%  [n<3 on a side — no verdict (no-cherry-pick)]
INFO-ONLY        iPhone18,1 litert-lm litert-community/Qwen3-4B short-chat warm  20.9 (n=1, spread 0%) -> 18.5 (n=2, spread 2%)  -11.8%  [n<3 on a side — no verdict (no-cherry-pick)]
INFO-ONLY        iPhone18,1 mlx-swift mlx-community/Qwen3-0.6B-4bit short-chat warm  167.3 (n=3, spread 1%) -> 171.6 (n=2, spread 3%)  +2.6%  [n<3 on a side — no verdict (no-cherry-pick)]
INFO-ONLY        iPhone18,1 mlx-swift mlx-community/Qwen3-0.6B-4bit short-chat cold  167.3 (n=1, spread 0%) -> 173.9 (n=2, spread 1%)  +3.9%  [n<3 on a side — no verdict (no-cherry-pick)]

== device / prefill_tps ==
INFO-ONLY        iPhone18,1 mlx-swift mlx-community/Qwen3-0.6B-4bit short-chat warm  404.5 (n=3, spread 39%) -> 507.3 (n=2, spread 9%)  +25.4%  [n<3 on a side — no verdict (no-cherry-pick)]
INFO-ONLY        iPhone18,1 mlx-swift mlx-community/Qwen3-0.6B-4bit short-chat cold  417.5 (n=1, spread 0%) -> 457.1 (n=2, spread 1%)  +9.5%  [n<3 on a side — no verdict (no-cherry-pick)]

== device / ttft_ms ==
UNRELIABLE       iPhone18,1 litert-lm litert-community/Qwen3-0.6B short-chat warm  187.0 (n=3, spread 5%) -> 150.0 (n=3, spread 4%)  -19.8%  [spread > 5% — throw out]
INFO-ONLY        iPhone18,1 litert-lm litert-community/Qwen3-0.6B short-chat cold  551.0 (n=1, spread 0%) -> 404.0 (n=1, spread 0%)  -26.7%  [n<3 on a side — no verdict (no-cherry-pick)]
INFO-ONLY        iPhone18,1 litert-lm litert-community/Qwen3-4B short-chat warm  1523.0 (n=1, spread 0%) -> 869.0 (n=2, spread 9%)  -42.9%  [n<3 on a side — no verdict (no-cherry-pick)]
INFO-ONLY        iPhone18,1 mlx-swift mlx-community/Qwen3-0.6B-4bit short-chat warm  48.0 (n=3, spread 31%) -> 39.0 (n=2, spread 10%)  -18.8%  [n<3 on a side — no verdict (no-cherry-pick)]
INFO-ONLY        iPhone18,1 mlx-swift mlx-community/Qwen3-0.6B-4bit short-chat cold  52.0 (n=1, spread 0%) -> 45.5 (n=2, spread 2%)  -12.5%  [n<3 on a side — no verdict (no-cherry-pick)]

wrote /Users/majimadaisuke/Downloads/ios-llm-benchmark/results/regression-reports/2026-08-19-litert-lm-v0.16.0-ios/verdicts.json (12 verdicts)

```
