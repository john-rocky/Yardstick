# python-envs — observed specs of the venvs behind published rows

`pip list --format=freeze` snapshots taken 2026-08-05 (continuous-bench gap ①-2: the
Python side of every published number must be reproducible from the repo). These are
records of the environments **as they were when the rows were captured** — including
`pip` itself and, in lt092run's case, the whole conversion stack that shared the venv.

| spec | venv it snapshots | python | role in published rows |
|---|---|---|---|
| `venv-litert015.requirements.txt` | `.venv-litert015` (repo root, git-ignored) | 3.14.6 | `scripts/gsm8k_litert_pip_thinking.py` — released-0.15.0-dylib cross-validation of the ctx4096 yardstick pair (89/92), per-question agreement |
| `lt0150run.requirements.txt` | `~/venvs/lt0150run` | 3.14.6 | 0.15 side of the same-instrument version delta + resident A/B (`litert-lm` CLI; `scripts/bench_litert_015_mac_speed.sh`) |
| `lt092run.requirements.txt` | `~/venvs/lt092run` | 3.10.13 | 0.14 contrast side of that delta (litert-lm 0.14.0). The venv doubles as the model-conversion env — litert-torch **0.9.1 release** (the version int4 conversion requires; dev checkouts produce iOS-delegate-dead artifacts), ai-edge-litert 2.1.6 — so the freeze is 83 packages, most of them conversion-only. |

Recreate (spec = observed state, so install from the file, not by name):

```bash
python3.14 -m venv .venv-litert015 && .venv-litert015/bin/pip install -r tools/python-envs/venv-litert015.requirements.txt
python3.14 -m venv ~/venvs/lt0150run && ~/venvs/lt0150run/bin/pip install -r tools/python-envs/lt0150run.requirements.txt
python3.10 -m venv ~/venvs/lt092run  && ~/venvs/lt092run/bin/pip install -r tools/python-envs/lt092run.requirements.txt
```

Pin registry: `environment.lock.json` → `python_instruments` (update both in the same
commit when a capture changes an environment).
