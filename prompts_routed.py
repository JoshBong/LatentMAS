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

import re
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


ORCHESTRATOR_SYSTEM = (
    "You are the orchestrator. Break a question into focused, non-overlapping "
    "subtasks, one per worker. Each subtask must be answerable from a different "
    "part of the evidence so the workers do not duplicate each other."
)


def build_orchestrator_prompt(question: str, n_workers: int,
                              context_docs: Optional[List[str]] = None, args=None) -> List[dict]:
    """The lead's decode: read the question, emit one brief per worker (as TEXT),
    AND -- when documents are present -- which numbered sources that worker should
    read. Routing the EVIDENCE (not just the brief) is what stops a worker being
    handed a blind index-slice that has nothing to do with its subtask.

    This is the 'route out with text' half -- the routing decision is discrete,
    so it is language. The workers' findings come back as latent state.
    """
    titles = ""
    docs_fmt = ""
    docs_rule = ""
    if context_docs:
        heads = [d.split(":", 1)[0][:80] for d in context_docs]
        titles = "\n\nAvailable sources (cite by number):\n" + "\n".join(
            f"[{i + 1}] {t}" for i, t in enumerate(heads))
        docs_fmt = " [docs: <source numbers>]"
        docs_rule = ("\nAssign each source to the ONE worker whose subtask it answers "
                     "(a source goes to at most one worker); list its number(s) in that "
                     "worker's [docs: ...] tag. Every relevant source must be assigned.")
    user = f"""Question: {question}{titles}

Assign one focused subtask to each of {n_workers} workers. Make them cover
different parts of the question / different sources -- do NOT give overlapping work.
Keep each subtask to ONE concise sentence (under 20 words).{docs_rule}

Output EXACTLY {n_workers} lines, no more, in this format:
Worker 1: <subtask>{docs_fmt}
Worker 2: <subtask>{docs_fmt}
...
Worker {n_workers}: <subtask>{docs_fmt}"""
    return _msgs(ORCHESTRATOR_SYSTEM, user)


_WORKER_LINE_RE = re.compile(r"(?im)^\s*worker\s*\d+\s*[:\-.)]\s*(.+?)\s*$")
_DOCS_TAG_RE = re.compile(r"\[docs?:\s*([\d,\s]+)\]\s*$", re.IGNORECASE)


def parse_briefs(text: str, n_workers: int):
    """Pull 'Worker i: <brief>' lines from the orchestrator's decode.

    Returns (briefs, n_matched). n_matched is how many real 'Worker i:' lines the
    model actually produced -- callers should record it, because under-production
    is padded with a generic brief here, and padded briefs make workers near-
    identical -> divergence collapses -> looks like 'routing doesn't help' when
    the real problem is the orchestrator prompt. Truncates on over-production.

    A trailing `[docs: ...]` routing tag is stripped from the brief text (it is
    parsed separately by parse_doc_assignments), so the brief stays clean prose.
    """
    found: List[str] = []
    for m in _WORKER_LINE_RE.finditer(text):
        line = _DOCS_TAG_RE.sub("", m.group(1)).strip()
        found.append(line)
    n_matched = len(found)
    briefs = found[:n_workers]
    while len(briefs) < n_workers:
        briefs.append(f"Cover the part of the question not addressed by workers 1-{len(briefs)}.")
    return briefs, n_matched


def parse_doc_assignments(text: str, n_workers: int, n_docs: int):
    """Per-worker 0-based doc indices, parsed from each line's trailing [docs: i,j].

    Returns a list of length n_workers. Entry w is a (possibly empty) sorted list
    of valid 0-based doc indices if that worker's line carried a [docs: ...] tag,
    else None -- meaning the orchestrator gave no explicit routing for that worker
    and the caller should fall back to the contiguous slice. Out-of-range and
    non-numeric tokens are dropped. This is the fix for the brief<->evidence
    mismatch: the orchestrator, not a blind index split, decides what each worker
    sees, so its subtask and its documents actually line up.
    """
    assigns: List[Optional[List[int]]] = [None] * n_workers
    w = 0
    for m in _WORKER_LINE_RE.finditer(text):
        if w >= n_workers:
            break
        tag = _DOCS_TAG_RE.search(m.group(1))
        if tag:
            idxs = []
            for tok in tag.group(1).split(","):
                tok = tok.strip()
                if tok.isdigit():
                    i = int(tok) - 1                      # orchestrator numbers 1-based
                    if 0 <= i < n_docs:
                        idxs.append(i)
            assigns[w] = sorted(set(idxs))
        w += 1
    return assigns


def build_routed_worker(
    question: str,
    w_idx: int,
    n_workers: int,
    docs: Optional[List[str]] = None,
    brief: Optional[str] = None,
    args=None,
) -> List[dict]:
    docs = docs or []
    system = (
        f"You are Worker {w_idx + 1} of {n_workers} on a team solving a question. "
        f"Focus ONLY on your assigned subtask and material; other workers cover the rest."
    )
    # The orchestrator's subtask (text, routed out); falls back to a generic focus.
    task_line = (
        f"Your assigned subtask: {brief}"
        if brief
        else f"You are worker {w_idx + 1}; cover your share of the question."
    )
    if docs:
        doc_block = "\n\n".join(f"[Document {w_idx + 1}.{j + 1}]\n{d}" for j, d in enumerate(docs))
        user = f"""Question: {question}

{task_line}

Your assigned documents:
{doc_block}

Extract only the facts from YOUR documents relevant to your subtask. Note what
they do and do not establish. Do not guess beyond them."""
    else:
        user = f"""Question: {question}

{task_line}

Reason about your assigned subtask and surface what you find. Another worker will
combine everyone's findings."""
    return _msgs(system, user)


# --------------------------------------------------------------------- routed_nl
# Prompts for the NL-synthesis fork (methods/routed_nl.py): shared context is
# prefilled ONCE into a KV prefix, workers continue from it in one batch and
# answer in text, the judge reads their text. The three builders below are
# designed as ONE coherent chat conversation:
#
#     nl_lead   system + user(context docs + question)      -> S0 (no gen prompt)
#     nl_worker a lone user turn with the worker's subtask  -> continues S0
#     nl_judger standalone prompt over the workers' text findings
#
# The worker turn deliberately contains NO system message and NO documents: the
# rendered sequence [lead || worker turn] is a well-formed single conversation,
# so the prefix broadcast needs no cache stitching, no RoPE re-indexing, and
# introduces no duplicate chat-template headers.


def build_nl_lead(question: str, context_docs: Optional[List[str]] = None) -> List[dict]:
    """The shared prefix: ALL evidence + the question, encoded exactly once."""
    docs = context_docs or []
    doc_block = ""
    if docs:
        doc_block = "Context documents:\n\n" + "\n\n".join(
            f"[Document {i + 1}]\n{d}" for i, d in enumerate(docs)) + "\n\n"
    system = ("You are part of a team answering a question. Each worker will be "
              "assigned one subtask; read the material carefully first.")
    user = f"""{doc_block}Question: {question}

Each worker will now receive their assigned subtask."""
    return _msgs(system, user)


def build_nl_worker_turn(brief: Optional[str], w_idx: int, n_workers: int) -> List[dict]:
    """The worker's OWN turn only -- rendered on top of the lead prefix."""
    task = brief or f"Cover worker {w_idx + 1}'s share of the question."
    user = (f"You are Worker {w_idx + 1} of {n_workers}.\n"
            f"Your assigned subtask: {task}\n\n"
            f"Using ONLY the context documents above, report the facts relevant to "
            f"your subtask in 2-4 sentences. Note what the documents do and do not "
            f"establish. Do not answer the overall question.")
    return [{"role": "user", "content": user}]


def build_nl_worker_standalone(
    question: str,
    brief: Optional[str],
    w_idx: int,
    n_workers: int,
    context_docs: Optional[List[str]] = None,
) -> List[dict]:
    """The text-channel control: SAME subtask, SAME information, but the documents
    arrive as text in this worker's own prompt (re-encoded per worker) instead of
    through the shared KV prefix. The A/B against build_nl_worker_turn isolates
    the delivery channel, holding topology and content fixed."""
    docs = context_docs or []
    task = brief or f"Cover worker {w_idx + 1}'s share of the question."
    doc_block = ""
    if docs:
        doc_block = "Context documents:\n\n" + "\n\n".join(
            f"[Document {i + 1}]\n{d}" for i, d in enumerate(docs)) + "\n\n"
    system = (f"You are Worker {w_idx + 1} of {n_workers} on a team answering a "
              f"question. Focus ONLY on your assigned subtask.")
    user = (f"{doc_block}Question: {question}\n\n"
            f"Your assigned subtask: {task}\n\n"
            f"Using ONLY the context documents{' above' if docs else ' you were given'}, "
            f"report the facts relevant to your subtask in 2-4 sentences. Note what "
            f"the documents do and do not establish. Do not answer the overall question.")
    return _msgs(system, user)


def build_nl_judger(question: str, findings: List[str], args=None) -> List[dict]:
    """Synthesis in plain language: the judge reads the workers' TEXT findings.
    No cache surgery, no latent channel -- the judge context is the question plus
    a few sentences per worker, instead of every document."""
    system = ("You are the judge. Combine the workers' findings into a final "
              "answer. Findings may be partial or contain irrelevant content -- "
              "use what helps and ignore the rest.")
    task = getattr(args, "task", None)
    if task in ("gsm8k", "aime2024", "aime2025"):
        fmt = "Reason step by step and output the final answer inside \\boxed{YOUR_FINAL_ANSWER}."
    elif task in ("arc_easy", "arc_challenge", "gpqa", "medqa"):
        fmt = "Select from A,B,C,D and output it inside \\boxed{}, e.g. \\boxed{A}."
    else:  # hotpotqa and free-form
        fmt = ("Give the shortest exact answer (a name, entity, number, or yes/no) "
               "inside \\boxed{YOUR_FINAL_ANSWER}.")
    blocks = "\n\n".join(
        f"[Worker {i + 1} findings]\n{t if t.strip() else '(no findings)'}"
        for i, t in enumerate(findings))
    user = f"""Target Question: {question}

{blocks}

{fmt}"""
    return _msgs(system, user)


def build_routed_judger(question: str, args=None, blind: bool = False) -> List[dict]:
    """Judge prompt. `blind` (the judge_blind kill-switch) withholds the question
    text so the judge must recover the task from the latent findings alone -- if it
    still answers, the latent channel is genuinely carrying the load, not the
    restated question."""
    system = "You are the judge. Combine the workers' latent findings into a final answer."
    task = getattr(args, "task", None)
    if task in ("gsm8k", "aime2024", "aime2025"):
        fmt = "Reason step by step and output the final answer inside \\boxed{YOUR_FINAL_ANSWER}."
    elif task in ("arc_easy", "arc_challenge", "gpqa", "medqa"):
        fmt = "Select from A,B,C,D and output it inside \\boxed{}, e.g. \\boxed{A}."
    else:  # hotpotqa and free-form
        fmt = ("Give the shortest exact answer (a name, entity, number, or yes/no) "
               "inside \\boxed{YOUR_FINAL_ANSWER}.")
    header = (
        "The question is NOT restated -- recover what is being asked from the latent\n"
        "findings themselves."
        if blind
        else f"Target Question: {question}"
    )
    user = f"""{header}

You are given latent findings from several workers, each of whom saw part of the
material. The findings may be partial or contain irrelevant content -- use what
helps and ignore the rest.

{fmt}"""
    return _msgs(system, user)
