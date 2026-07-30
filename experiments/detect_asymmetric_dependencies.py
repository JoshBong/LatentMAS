"""Test script for the orchestrator to detect asymmetric dependencies.

Run this script to evaluate if an LLM can reliably detect 'Cross-Entity Conditional Extraction'
(asymmetric data dependencies) before attempting to route them in parallel.
If the orchestrator detects a dependency, it should ideally degrade to sequential text_mas
or a multi-stage routed_mas.

Usage:
    python experiments/detect_asymmetric_dependencies.py --model Qwen/Qwen2.5-3B-Instruct --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import ModelWrapper
from utils import auto_device


def build_detection_prompt(question: str) -> list[dict]:
    system = (
        "You are an orchestrator analyzing a complex question for a multi-agent system. "
        "Your task is to determine if the question can be perfectly parallelized into independent subtasks, "
        "or if it contains an 'asymmetric dependency' (where one subtask needs the output of another subtask to even begin)."
    )
    user = f"""Question: {question}

Can this question be split into parallel subtasks where workers do not need to communicate with each other?
Or does one worker need to know what another worker discovers?

Analyze the question step-by-step. 
If it requires an intermediate discovery to complete another part of the task, output 'DEPENDENCY DETECTED' at the end.
If it can be completely parallelized, output 'PARALLELIZABLE' at the end.
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    dev = auto_device(a.device)
    
    # Minimal args required by ModelWrapper
    args = argparse.Namespace(
        device=str(dev), 
        device2=str(dev),
        method="routed_mas", 
        use_second_HF_model=False,
        enable_prefix_caching=False,
        latent_space_realign=False
    )
    
    print(f"Loading {a.model} on {dev}...")
    try:
        mw = ModelWrapper(a.model, dev, use_vllm=False, args=args)
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Note: Run this inside the 'latentmas' conda environment with PyTorch installed.")
        return

    questions = [
        # Symmetric (Parallelizable)
        "What is the capital of France and what is the capital of Germany?",
        "Compare the 2022 revenue of Apple with the 2022 revenue of Microsoft.",
        "Which city is more populous: Tokyo or New York?",
        
        # Asymmetric (Dependency Detected)
        "Between Apple and Microsoft, what was Microsoft's revenue in the year Apple filed its first patent?",
        "Who was the president of the US when the founder of Microsoft was born?",
        "Find the highest grossing movie from the director who directed Inception."
    ]

    for q in questions:
        msgs = build_detection_prompt(q)
        _, ids, mask, _ = mw.prepare_chat_batch([msgs], add_generation_prompt=True, enable_thinking=False)
        raw, _ = mw.generate_text_batch(ids, mask, max_new_tokens=150, temperature=0.0, past_key_values=None)
        
        print(f"\n{'='*60}")
        print(f"Question: {q}")
        print(f"{'-'*60}\nAnalysis:\n{raw[0].strip()}")

if __name__ == "__main__":
    main()
