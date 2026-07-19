# SESSION BRIEF — Cactus arm: the 7th row of the Gemma-4-E2B table (written 2026-07-20)

Self-contained. Launch in `~/code/apple-silicon-llm-bench`. Read `CLAUDE.md` and
`methodology/fairness-rules.md` first; the published table this extends lives in README
("Gemma 4 E2B — every runtime at its best *available* build") with its audit trail in
`results/raw/2026-07-18-gemma4-bestquant/SUMMARY.txt`.

## Why this session exists

Lu's engine list for the continuous-benchmarking tool is **LiteRT-LM / llama.cpp /
Core AI / MLX / Cactus** — Cactus is the only listed engine with no arm in this app.
Separately, Shuangfeng (LiteRT team) already asked for a Cactus comparison once
(2026-07-10 session, measurement DONE — see "Prior assets"). The deliverable now is a
**Cactus row in the 5-axis table** (decode / GSM8K / J/tok / deep-context / memory), same
protocols as the other six rows, plus the in-app arm that makes it repeatable.

## Prior assets (2026-07-10 session — do not re-derive)

All in `~/Downloads/ios-llm-benchmark` (the community repo), NOT this repo:
- `docs/cactus-vs-litert.md` (+`.ja.md`) — the full prior report, corrected 2026-07-15.
- `results/raw/2026-07-10-cactus-parity/` — raw runs incl. `cactus_iphone17pro_{auto,cpu,metal}.jsonl`.
- `scripts/cactus_parity_report.py` — parses cactus benchmark output; asserts the
  prefill-not-inflated check every run.
- `ios/BenchmarkApp/Sources/Benchmark/Tasks/CactusParityTask.swift` (untracked) — makes
  OUR runtimes run *Cactus's* protocol (1K prefill / 100 decode). Note the direction: it
  does NOT run Cactus on our protocol. Useful as protocol documentation.
- The cactus source checkout the session used is NOT under ~/code (find returns nothing) —
  re-clone `cactus-compute/cactus` and rebuild its benchmark/engine binaries.

Known results from that session (Gemma-4-E2B, their CQ4 build):
- Speed reproduces: iPhone 17 Pro 704.8 prefill / 36.3 decode / 651 MB (their protocol:
  1000-tok prefill, 100-tok decode, 1 warmup + mean of 3, warm). M4 Max: 2738 / 141.6.
- **CQ4 is quality-dead on multi-step reasoning: GSM8K 3.0% (n=100)** vs bf16 anchor 92 /
  LiteRT int4-QAT 88 / MLX-PTQ 78-as-floor in that session's harness. Confirmed not a
  `####`-extraction artifact and not rescued by their cloud-handoff confidence probe.
- Their Metal path is token-identical iPhone↔Mac (quality transfers; CPU path diverges).

## The build-selection question (answer it FIRST — it shapes the whole row)

Our table's rule is "each arm at its best *usable* build, stated per row". Cactus ships
several Gemma-4-E2B artifacts on HF (`Cactus-Compute/`): `gemma-4-E2B-it`,
`gemma-4-e2b-it-cq` (CQ4), and experimental `gemma4-e2b-grouped-k96/k192[-router]`.
Establish what each actually is (format, bits, which their released SDK/app loads by
default) and measure GSM8K for the candidates on Mac BEFORE committing device time.
If CQ4 is their only phone-deployable format, the row is CQ4 with its 3% — brutal but
per-row-honest. If a better-quality build is deployable, THAT is the row (and the CQ4
number goes in a footnote, like MLX's two-build split). Do not let the row look like a
hit job: same rules as everyone, build stated, best usable wins.

## Measurement plan (protocol parity with the existing 6 rows)

Two integration paths; do B first, A only if warranted:

**Path B — sidecar (fast, gets the row):** drive Cactus's own engine/binary at OUR
protocols, the way litert-mac-verify serves the LiteRT quality column:
1. GSM8K n=100 on Mac through the cactus engine (greedy, thinking-off prompt, same
   extractor — port `run_litertlm`'s shape in `~/code/hf-to-litertlm/scripts/parity_gsm8k.py`).
   Re-run even though 3.0% exists: the table's quality column promises ONE harness.
2. iPhone cold short-chat: their example/bench app driven headless if possible, else
   their benchmark binary with prompt≈short-chat and 128-token budget, 3 thermal-nominal
   colds. If their tooling cannot express our protocol, the cell is "not expressible via
   shipped tooling" with THEIR protocol's number as an annotated reference — that
   finding mirrors the llm-runner/litert-mac-verify quirks and is worth having.
3. Deep-context p=1024/g=256: their parity protocol is already ~1K prefill — closest cell
   to free; capture peak memory.
4. Energy J/tok: needs unplugged battery-delta → only meaningful via an in-app arm or
   their app running a 600 s sustained loop. If not expressible, annotate like the other
   blocked cells (reason in-cell).

**Path A — in-app runtime adapter (the tool milestone):** `CactusRuntime.swift`
conforming to `LLMRuntime`, vendored like LiteRT-LM (local SwiftPM package or
xcframework; `bootstrap.sh` pattern). This unlocks all 5 axes natively including energy.
Session-scale; check their iOS SDK surface (Flutter/RN wrappers exist — find the C++/
Swift-reachable core) before promising it. If A lands, B's cells are superseded by
in-app captures (same-session re-run, cross-session pooling ban applies).

## Wiring checklist (when cells land)

- Import via the flat convention (`results/raw/2026-07-18-gemma4-bestquant/import_to_flat.py`
  pattern; new date-dir for this session's captures + SUMMARY.txt section).
- README Gemma table + charts (`scripts/generate_charts.py` — ROWS lists in
  `chart_iphone`, `chart_thinking` n/a unless thinking measured, `chart_deepcontext`),
  `render_results.py` LOGICAL_MODELS already collapses gemma-4-e2b IDs — verify the
  cactus HF ids collapse correctly (add patterns if not).
- The Lu reports (`~/Downloads/meeting/gemma4-e2b-post-v2-{ja,en}.md`) only if the user
  asks — they may already be sent by the time this session runs.
- Thinking mode: probe whether cactus exposes a thinking toggle (their template handling
  is their own). The thinking table has a LiteRT "locked out" precedent — either result slots in.

## Do-not-repeat (prior session's hard-won corrections — repeating these burns trust)

- **Cactus's headline prefill is NOT inflated.** `cache_state_copy_ms` measures
  0.01–0.03 ms; `prefill_tps` ≈ honest `ttft_prompt_tps` to <0.01%. The parity script
  asserts this — keep the assertion.
- **Do NOT open issues/PRs against Cactus without explicit user approval** (user
  declined outward contact in the prior session).
- The `mlx-community/gemma-4-e2b-it-4bit` re-upload gotcha (2026-07-06, KV-shared
  layout): old-rev numbers and new-rev numbers are different artifacts.
- Cross-session speed pooling ban (`iphone-session-variance`): the 2026-07-10 cactus
  speed numbers and this session's captures must not share a median; re-measure.
- Device A6F3E849 is shared — announce before use; serialize devicectl. Background
  driver scripts get killed — `nohup` + DONE-marker pattern survives.
- Build traps: fresh DerivedData (stale CoreMLLLM.swiftmodule), YARDSTICK_BUNDLE_ID /
  YARDSTICK_TEAM overrides, canonical bundle id unregisterable.

## Acceptance criteria

1. Build-selection memo: which Cactus build is "best usable" and why (with GSM8K numbers
   for the candidates).
2. Cactus row in the README table with ≥3 of 5 cells measured at protocol parity, every
   remaining cell annotated in-cell with its reason.
3. Charts regenerated with the 7th row; RESULTS.md re-rendered.
4. A one-paragraph "Cactus arm status" the user can forward to Lu (his engine list is
   the reason this exists).
5. Commits with findings as messages, raw JSONs + SUMMARY addendum in results/raw/.
