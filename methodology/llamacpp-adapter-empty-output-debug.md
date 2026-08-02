# DEBUG SESSION BRIEF — llama.cpp adapter empty-output bug (written 2026-07-19)

> **RESOLVED 2026-07-19 (same day).** Root cause: the adapter's applyChatTemplate fell
> back to the BARE prompt on every task — llama_chat_apply_template pattern-matches
> template *source* and unsloth's 18.8 KB gemma-4 Jinja contains no literal turn marker;
> gemma-4 renamed the markers to `<|turn>role`…`<turn|>` (ids 105/106) and `<turn|>` IS
> the EOS, so untemplated complete-looking prompts sampled EOS as the first token
> (short-chat's one-liner invited continuation and escaped). Fixed in
> LlamaCppRuntime.swift (vocab-probe family detection + hand-rolled gemma-4 markup,
> per-call KV clear, per-call sampler honoring task temp). All acceptance criteria met;
> full chain + validation in `results/raw/2026-07-18-gemma4-bestquant/SUMMARY.txt`
> ("llama.cpp adapter empty-output bug — ROOT-CAUSED + FIXED"). Kept for methodology.

Self-contained. Launch the session in `~/code/apple-silicon-llm-bench`. The bug blocks two
cells of the published Gemma-4-E2B table (README + the Lu report), both annotated ⓕ —
fixing it and filling those cells is the deliverable.

## The symptom — exactly what is known

The bench app's **llama.cpp arm generates ZERO tokens on every task except `short-chat`**,
on device (iPhone 17 Pro, iOS 27.0, Release, vendored llama.xcframework **b8999**, model
`unsloth/gemma-4-E2B-it-GGUF` Q4_K_M via in-app HF download):

| task | result | evidence (in `results/raw/2026-07-18-gemma4-bestquant/`) |
|---|---|---|
| `short-chat` | **WORKS** — 37.6 tok/s, 128 tokens, coherent text | `llama.cpp_*_short-chat_2026-07-18T00-{36,42,45}*.json` |
| `quality` | empty `outputSample`, ~0 tokens | `llama.cpp_*_quality_2026-07-18T05-21-24Z.json` |
| `long-context-1024` | prefill RUNS (904–1,040 tok/s), decode **0 tokens**, `ttft_ms 0`, `stopReason "stop"`, empty sample | `llama.cpp_*_long-context-1024_*.json` (n=3, all identical) |
| `energy` | **no record at all** + the app SPINS FOREVER | see "Danger" below |

Counter-evidence that this is OUR adapter, not llama.cpp or the model:
- The SAME gguf runs GSM8K 76/100 on the Mac via llama-server b10064 (same chat template,
  `reports/parity/gsm8k_llamacpp-gemma4-e2b-q4km.json` in `~/code/hf-to-litertlm`).
- The same app run, same model, works on `short-chat`.
- Every other runtime (litert/mlx/core-ai) answers `quality` fine in the same binary.

## Why energy spins forever (understand before touching it)

`EnergyTask` re-prompts until `sustainSeconds` of ACTIVE DECODE accumulate
(`BenchmarkRunner.swift` ~line 135). Zero tokens decoded → zero seconds accumulate → the
loop never exits and never writes a record. This is also why the 2026-07-19 energy pass
left a BenchmarkApp process running for ~40 min until it was SIGKILLed.

**Danger: do NOT launch `--task energy` on the llama arm until the bug is fixed.** If you
do, kill it with:
```
xcrun devicectl device info processes --device <UDID> | grep BenchmarkApp   # get pid
xcrun devicectl device process signal --device <UDID> --pid <pid> --signal SIGKILL
```

## Discriminating facts for hypothesis-building

- It is NOT prompt length: `quality` uses SHORT prompts and fails; `short-chat` is short
  and works; `long-context` is long and fails.
- It is NOT temperature alone: `long-context` (temp 0.0) fails while `short-chat`
  (temp 0.0) works; `energy` (temp 0.7) also fails.
- `long-context` shows `prefill ~1,000 tok/s` + `ttft_ms 0` + `stopReason "stop"`:
  the model prefills, then the FIRST sampled token already terminates generation
  (or sampling never starts). "stop" (not "length") with 0 tokens smells like an
  immediate stop-token hit or an antiprompt/EOS check firing on step 0.
- Whether `quality` makes one generate call or several (one per question) is UNKNOWN —
  read `QualityTask` + `BenchmarkRunner`. If quality is multi-call and short-chat is
  single-call, "second generate() after reset produces nothing" becomes a strong
  candidate (energy is also multi-call via re-prompting). But long-context is
  single-call and fails — so if multi-call is the mechanism, there must be a second
  factor (long-context's failure could then be a DIFFERENT bug; don't assume one cause).

## Where to look

1. `ios/BenchmarkApp/Sources/Runtimes/LlamaCppRuntime.swift` — the generate loop:
   stop-token/EOS handling, antiprompt logic, sampler init/reset, `n_ctx` /
   context-params (long-context 1,073-token prompt + 256 budget vs whatever n_ctx the
   adapter passes), and whether state carries across generate() calls (reset semantics).
2. `ios/BenchmarkApp/Sources/Benchmark/BenchmarkRunner.swift` — how tasks drive
   generate(); what differs between short-chat's invocation and quality/energy's.
3. `ios/BenchmarkApp/Sources/Benchmark/Tasks/` — per-task `GenerationParameters`
   (maxTokens/temperature/topP) and prompts. Diff short-chat vs quality field by field;
   suspect any parameter unique to the failing set (topP? a penalty? maxTokens
   plumbing — e.g. an int truncation or a 0 sneaking in).
4. Live repro with console (fastest loop; ~40 s per try):
   ```
   xcrun devicectl device process launch --console --terminate-existing \
     --device <UDID> com.daisukemajima.llmbench -- \
     --yardstick-autorun --runtime "llama.cpp" \
     --model-id "unsloth/gemma-4-E2B-it-GGUF/Q4_K_M" --task quality --runs 1
   ```
   Add temporary logging in the adapter (sampled token id, EOS id, stop checks per step).
5. Cross-check on the Mac CLI to avoid device cycles: the SAME adapter builds into the
   `yardstick` macOS tool (`YARDSTICK_BIN=/tmp/yardstick-dd2/.../yardstick` or rebuild:
   scheme `yardstick`, see project.yml) — if quality is empty on the Mac too, the whole
   debug loop is local and fast. CHECK THIS FIRST.

## History note (don't assume regression)

The May-era iPhone llama cells were short-chat only (no quality/long-context/energy runs
exist in `results/raw/superseded/`), so there is no evidence this ever worked — "never
worked" is as likely as "regressed with the b8999 bump". `git log -- ios/BenchmarkApp/Sources/Runtimes/LlamaCppRuntime.swift`
and the LLAMA_TAG history in `ios/BenchmarkApp/scripts/bootstrap.sh` give the timeline.
A cheap discriminator: `LLAMA_TAG=<older> ./scripts/bootstrap.sh` (delete
`Vendored/llama.xcframework` first) and re-test.

## Acceptance criteria (definition of done)

1. `quality` returns the 9 answers (compare litert's: 42 / Tokyo / Cold / Seven / Merci /
   56 / 0.9 / 0.11 / blue).
2. `long-context-1024` decodes ~256 tokens (prefill throughput stays ~1,000 tok/s).
3. `energy --sustain-seconds 600` completes and writes a record (unplugged, ≤90% battery,
   Auto-Lock Never — see `scripts/bench_energy_iphone_gemma_part2.sh` header).
4. Fill the two ⓕ cells and update the annotations in: README Gemma table,
   `results/raw/2026-07-18-gemma4-bestquant/SUMMARY.txt`, and the send-out reports
   `~/Downloads/meeting/gemma4-e2b-post-v2-{ja,en}.md`. Import new captures via the
   flat-convention pattern in `results/raw/2026-07-18-gemma4-bestquant/import_to_flat.py`.
5. Commit with the root cause in the message (this repo's convention: the finding IS the
   commit message).

## Do-not-repeat (this campaign's hard-won lessons)

- The known llama.cpp **Metal cleanup abort on process exit is HARMLESS** and pre-dates
  this bug (`measure_energy.py` even special-cases it) — don't chase it.
- Cold captures must be `initialThermalState == nominal` (fairness rules #2) — downloads
  heat the phone; poll `Documents/results`, don't fixed-sleep (SUMMARY "Operational notes").
- Shared DerivedData carries stale swiftmodules (`BenchmarkApp-coreai` DD has a
  CoreMLLLM.swiftmodule that flips `canImport` back on) — build with a fresh DD.
- The canonical bundle id is unregisterable; use `YARDSTICK_BUNDLE_ID`/`YARDSTICK_TEAM`
  overrides (scripts already support them).
- Background driver scripts can be killed by the session harness — `nohup` + a DONE
  marker file survived where plain background tasks were killed 3×.
