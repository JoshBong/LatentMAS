"""Prompts for routed fan-out (routed_mas).

Kept separate from prompts.py so the fork doesn't touch the baseline builders.
Three roles, matching the three phases in methods/routed_mas.py:

    lead     encode the shared question once  -> base cache S0
    worker   one worker's targeted subtask + its slice of the context
    judger   answer from the combined latent context

`context_docs` is a list of documents (HotpotQA-distractor ships them with the
question). Each worker gets a contiguous slice; on tasks with no documents the
slice is empty and the worker just gets a generic "focus" instruction, which is
why routing should be a no-op on single-domain tasks (the control).
"""

from __future__ import annotations

from typing import List, Optional


def _msgs(system: str, user: str) -> List[dict]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def worker_doc_slice(context_docs: Optional[List[str]], w_idx: int, n_workers: int) -> List[str]:
    """Contiguous split of the documents across workers."""
    if not context_docs:
        return []
    n = len(context_docs)
    lo = (w_idx * n) // n_workers
    hi = ((w_idx + 1) * n) // n_workers
    return context_docs[lo:hi]


LEAD_SYSTEM = "You are the lead of a team solving a question. Read the question carefully."


def build_routed_lead(question: str, args=None) -> List[dict]:
    user = f"""You are coordinating a team of workers who will each investigate part of this question.

Question: {question}

Read and understand the question. Do not answer yet."""
    return _msgs(LEAD_SYSTEM, user)


def build_routed_worker(
    question: str,
    w_idx: int,
    n_workers: int,
    context_docs: Optional[List[str]] = None,
    args=None,
) -> List[dict]:
    docs = worker_doc_slice(context_docs, w_idx, n_workers)
    system = (
        f"You are Worker {w_idx + 1} of {n_workers} on a team solving a question. "
        f"Focus ONLY on your assigned material; other workers cover the rest."
    )
    if docs:
        doc_block = "\n\n".join(f"[Document {w_idx + 1}.{j + 1}]\n{d}" for j, d in enumerate(docs))
        user = f"""Question: {question}

Your assigned documents:
{doc_block}

Extract only the facts from YOUR documents that are relevant to the question.
Note what your documents do and do not establish. Do not guess beyond them."""
    else:
        user = f"""Question: {question}

You are one of {n_workers} workers. Reason about the aspect of this question that
is yours to cover (worker index {w_idx + 1}), and surface what you find. Another
worker will combine everyone's findings."""
    return _msgs(system, user)


def build_routed_judger(question: str, args=None) -> List[dict]:
    system = "You are the judge. Combine the workers' latent findings into a final answer."
    task = getattr(args, "task", None)
    if task in ("gsm8k", "aime2024", "aime2025"):
        fmt = "Reason step by step and output the final answer inside \\boxed{YOUR_FINAL_ANSWER}."
    elif task in ("arc_easy", "arc_challenge", "gpqa", "medqa"):
        fmt = "Select from A,B,C,D and output it inside \\boxed{}, e.g. \\boxed{A}."
    else:  # hotpotqa and free-form
        fmt = ("Give the shortest exact answer (a name, entity, number, or yes/no) "
               "inside \\boxed{YOUR_FINAL_ANSWER}.")
    user = f"""Target Question: {question}

You are given latent findings from several workers, each of whom saw part of the
material. The findings may be partial or contain irrelevant content -- use what
helps and ignore the rest.

{fmt}"""
    return _msgs(system, user)
