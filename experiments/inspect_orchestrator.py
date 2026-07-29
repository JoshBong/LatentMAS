"""Measure the orchestrator before trusting any divergence number.

Answers two questions with data instead of guesses:
  1. Does the orchestrator emit N *parseable* briefs, or does it under-produce and
     get padded? (Padding makes workers identical -> divergence reads low -> looks
     like "routing doesn't help" when the prompt/worker-count is the real problem.)
  2. How many tokens does it actually spend? -> set the backstop cap from the
     observed distribution, not a made-up constant.

Run once on the GPU box:
    python experiments/inspect_orchestrator.py --model Qwen/Qwen3-4B --device cuda \
        --n_workers 3 --n_questions 20 --task hotpotqa

Uses enable_thinking=False (as routed_mas does) and a deliberately huge cap, so
nothing truncates -- the point is to observe the NATURAL output length.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import ModelWrapper                       # noqa: E402
from prompts_routed import build_orchestrator_prompt, parse_briefs  # noqa: E402
from utils import auto_device                          # noqa: E402


def _questions(task, n):
    if task == "hotpotqa":
        from data import load_hotpotqa
        it = load_hotpotqa()
    elif task == "gsm8k":
        from data import load_gsm8k
        it = load_gsm8k(split="test")
    else:
        raise SystemExit(f"inspect_orchestrator: unsupported task {task!r}")
    out = []
    for i, item in enumerate(it):
        if i >= n:
            break
        out.append(item)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--task", default="hotpotqa")
    ap.add_argument("--n_workers", type=int, default=3)
    ap.add_argument("--n_questions", type=int, default=20)
    ap.add_argument("--max_new_tokens", type=int, default=1024,
                    help="huge on purpose: don't truncate while measuring")
    a = ap.parse_args()

    dev = auto_device(a.device)
    args = argparse.Namespace(device=str(dev), device2=str(dev), task=a.task,
                              method="routed_mas", use_second_HF_model=False,
                              enable_prefix_caching=False)
    mw = ModelWrapper(a.model, dev, use_vllm=False, args=args)

    lengths, padded = [], 0
    for i, item in enumerate(_questions(a.task, a.n_questions), 1):
        msgs = build_orchestrator_prompt(item["question"], a.n_workers,
                                         item.get("context_docs"), args)
        _, ids, mask, _ = mw.prepare_chat_batch([msgs], add_generation_prompt=True,
                                                enable_thinking=False)
        raw, _ = mw.generate_text_batch(ids, mask, max_new_tokens=a.max_new_tokens,
                                        temperature=0.7, top_p=0.95, past_key_values=None)
        briefs, n_parsed = parse_briefs(raw[0], a.n_workers)
        n_tok = len(mw.tokenizer(raw[0], add_special_tokens=False)["input_ids"])
        lengths.append(n_tok)
        if n_parsed < a.n_workers:
            padded += 1
        print(f"\n=== Q{i}: {item['question'][:80]}")
        print(f"  output tokens: {n_tok} | briefs parsed: {n_parsed}/{a.n_workers}")
        for w, b in enumerate(briefs, 1):
            print(f"  W{w}: {b[:100]}")

    lengths.sort()
    med = lengths[len(lengths) // 2] if lengths else 0
    print("\n" + "=" * 60)
    print(f"padding: {padded}/{a.n_questions} questions under {a.n_workers} real briefs")
    print(f"output tokens: min={min(lengths)} median={med} max={max(lengths)}")
    print(f"suggested orchestrator backstop (4x max): {4 * max(lengths)}")
    if padded:
        print("PADDING > 0 -> fix the orchestrator prompt or worker count BEFORE "
              "interpreting divergence.")


if __name__ == "__main__":
    main()
