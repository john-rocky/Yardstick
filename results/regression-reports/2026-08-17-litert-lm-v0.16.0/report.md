# Regression report — litert-lm v0.16.0 (2026-08-17)

```
/opt/homebrew/opt/python@3.14/bin/python3.14 /Users/majimadaisuke/Downloads/ios-llm-benchmark/scripts/regression_diff.py device --baseline engine:v0.15.0 --candidate engine:v0.16.0 --anchors matrices/anchors.cells --json-out /Users/majimadaisuke/Downloads/ios-llm-benchmark/results/regression-reports/2026-08-17-litert-lm-v0.16.0/verdicts.json
```

```
wrote /Users/majimadaisuke/Downloads/ios-llm-benchmark/results/summary/quality.csv (31 rows)
wrote /Users/majimadaisuke/Downloads/ios-llm-benchmark/results/summary/device-runs.csv (1056 rows)
wrote /Users/majimadaisuke/Downloads/ios-llm-benchmark/results/summary/history.csv (0 rows)
wrote /Users/majimadaisuke/Downloads/ios-llm-benchmark/results/summary/README.md

== device / decode_tps ==
UNRELIABLE       Pixel 8a litert-lm-cpu litert-community/Qwen3-0.6B short-chat cold  11.4 (n=3, spread 20%) -> 14.4 (n=5, spread 8%)  +26.4%  [spread > 5% — throw out]
OK               Pixel 8a litert-lm-gpu litert-community/Qwen3-0.6B short-chat cold  15.2 (n=3, spread 2%) -> 15.2 (n=3, spread 1%)  +0.3%

== device / prefill_tps ==
UNRELIABLE       Pixel 8a litert-lm-cpu litert-community/Qwen3-0.6B short-chat cold  8.3 (n=3, spread 30%) -> 9.8 (n=5, spread 53%)  +18.3%  [spread > 5% — throw out]
OK               Pixel 8a litert-lm-gpu litert-community/Qwen3-0.6B short-chat cold  41.6 (n=3, spread 2%) -> 41.5 (n=3, spread 4%)  -0.3%

== device / ttft_ms ==
UNRELIABLE       Pixel 8a litert-lm-cpu litert-community/Qwen3-0.6B short-chat cold  2510.0 (n=3, spread 23%) -> 2110.0 (n=5, spread 91%)  -15.9%  [spread > 5% — throw out]
OK               Pixel 8a litert-lm-gpu litert-community/Qwen3-0.6B short-chat cold  550.0 (n=3, spread 2%) -> 550.0 (n=3, spread 4%)  +0.0%

wrote /Users/majimadaisuke/Downloads/ios-llm-benchmark/results/regression-reports/2026-08-17-litert-lm-v0.16.0/verdicts.json (6 verdicts)

```
