#!/usr/bin/env python3
"""Regenerate prompts/text/*.txt — the canonical task prompts, byte-identical to
what the Swift tasks build (same-budget rule: every arm sees the same prompt).

Sources of truth being mirrored:
  ShortChatTask.swift    prompt literal, maxTokens 128, greedy
  LongContextTask.swift  [i]-tagged lorem blocks (55 tok/block nominal) + tail;
                         long-context-1024-gen256 = forceLongOutput, maxTokens 256

The Android CLI drivers feed these files via --input_prompt_file / llama-cli -f.
If a Swift task changes, regenerate and commit; a follow-up iOS unit test can
assert the Swift-built prompts hash to the same values.
"""
import hashlib
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "prompts", "text")

LOREM = ("Lorem ipsum dolor sit amet, consectetur adipiscing elit. Phasellus bibendum velit non augue "
         "ultricies, a vestibulum ipsum porttitor. Sed at nulla a justo viverra dictum. Vivamus blandit "
         "velit at lectus pulvinar pellentesque. Mauris dictum massa ut nisi tristique consequat.")
TOKENS_PER_BLOCK = 55


def long_context(target_tokens, force_long_output):
    blocks = max(1, target_tokens // TOKENS_PER_BLOCK)
    pieces = [f"[{i}] {LOREM}" for i in range(blocks)]
    if force_long_output:
        pieces.append("\n\nUsing the passage above as context, list 25 distinct things "
                      "on-device AI lets a phone do that a cloud model cannot. Number every item "
                      "and give each one two full sentences of explanation. Do not stop early.")
    else:
        pieces.append("\n\nFinish with one sentence: what on-device AI lets a phone do that a cloud model cannot.")
    return "\n".join(pieces)


PROMPTS = {
    # task id -> (text, decode budget the task pins)
    "short-chat": ("Explain what on-device AI means in simple terms.", 128),
    "long-context-1024-gen256": (long_context(1024, True), 256),
}


def main():
    os.makedirs(OUT, exist_ok=True)
    for task, (text, budget) in PROMPTS.items():
        path = os.path.join(OUT, f"{task}.txt")
        with open(path, "w") as fh:
            fh.write(text)
        sha = hashlib.sha256(text.encode()).hexdigest()
        print(f"{task}: {len(text)} bytes, budget {budget}, sha256 {sha[:16]}…")
    # budgets sidecar so drivers don't hardcode them
    with open(os.path.join(OUT, "budgets.tsv"), "w") as fh:
        for task, (_, budget) in PROMPTS.items():
            fh.write(f"{task}\t{budget}\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
