# Matrix cell files

One line per benchmark cell, shared by every platform runner (Mac, iPhone, Android)
and by the regression tooling. This is the model-agnostic replacement for the
per-campaign cell lists in `scripts/cells_*.txt` (those stay frozen as provenance
for their published campaigns).

## Grammar

```
<platform> <runtime> <model-id> <task> [key=value ...]
```

- Whitespace-separated, `#` starts a comment, blank lines ignored.
- `platform`: `ios` | `mac` | `android`. Runners filter to their own platform and
  ignore the rest, so one file can describe a cross-platform matrix.
- `runtime`: a `RuntimeKind` raw value (`mlx-swift`, `llama.cpp`, `litert-lm`,
  `core-ai`, `cactus`, `coreml-llm`, `executorch`, `anemll`, `apple-fm`).
- `model-id`: catalog id (usually the HF repo id; side-loaded ids like
  `litert-local/...` need `local=1`).
- `task`: a `BenchmarkTask` id (`short-chat`, `long-context-1024-gen256`, ...)
  or `native-benchmark-<P>x<D>` for engine-native synthetic benchmarks.

## Options (trailing `key=value`, any order)

| key | meaning |
|---|---|
| `runs=N` | process-launch run count for this cell (default: runner's `RUNS`, 4) |
| `context-tokens=N` | forwarded to the harness `--context-tokens` |
| `max-tokens=N` | forwarded to the harness `--max-tokens` |
| `cooldown=S` | seconds of cooldown BEFORE this cell (default: runner's `BASE_COOLDOWN`) |
| `anchor=1` | session-anchor cell: runners execute it first; `regression_diff.py --anchors` normalizes cross-session deltas through it |
| `exclude=<slug>` | structurally impossible / known-fail cell. Runners skip it; the leaderboard renders it as `— (<slug>)`. Keeping the row is deliberate (fairness rule: failed runs stay in the table) |
| `manual=1` | never run by automated runners (energy cells: the unplug discipline is a human step). `energy` task REQUIRES this flag |
| `local=1` | model is side-loaded, not in the HF catalog — catalog preflight skips it |
| `file=<name>` | artifact filename inside the HF repo (Android GGUF cells: which quant file to fetch/push) |
| `backend=<b>` | engine backend for runtimes that expose one (Android litert-lm: `cpu` \| `gpu`) |

## Conventions

- Order cells light → heavy; give >=3B models `cooldown=300` (SoC heat the 4-level
  thermal gate can't see).
- Anchor choice: a fast, stable cell. When the anchor's own engine is the one under
  test, anchor-normalized verdicts for that engine are confounded — `anchors.cells`
  carries one anchor per platform per runtime family so the diff tool can pick an
  anchor whose runtime differs from the engine under test.
- Interleave arms per model block (fairness rule 11): list the arms of one model
  together, not one arm's full ladder then the next.

## Files

- `apple-warm-matrix.cells` — the standing iPhone warm matrix (migrated from
  `scripts/cells_full_warm_matrix.txt`).
- `anchors.cells` — session-anchor cells, one (or two) per platform.
- `release-regression-litert.cells` — the cells re-measured when a LiteRT-LM
  release ships.

## Validation

```
python3 scripts/validate_cells.py matrices/*.cells            # grammar + semantics
python3 scripts/validate_cells.py --require-anchor FILE       # regression matrices
python3 scripts/validate_cells.py --catalog catalog.json FILE # preflight model ids
```

`catalog.json` comes from `yardstick list --json` (the Mac CLI compiles the same
`ModelCatalog.swift` the iOS app ships, so it validates iPhone cells too).
CI runs the plain form on every push (`cells-valid` job).
