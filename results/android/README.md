# Android companions

Measurements on Android devices for models and runtimes that also appear in the Apple tables, kept here so the same-model comparison is in one repo.

- `tinyhybridnet-three-runtimes.md` — one small PyTorch conv+transformer model on ExecuTorch (XNNPACK), LiteRT (litert-torch export) and ONNX Runtime, Galaxy S26 and the API 36 emulator, with host parity. Sources and results in `tinyhybridnet-three-runtimes/`.
- `litertlm-qwen2.5-1.5b-mac-galaxy-s26.md` — the official ungated `litert-community/Qwen2.5-1.5B-Instruct` bundle through LiteRT-LM on a Mac (0.16.1 CLI) and a Galaxy S26. Raw logs in `litertlm-qwen2.5-1.5b-raw/`.
