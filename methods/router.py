"""The lead's decomposition step: question -> independent lookup units + specs.

The router IS the lead agent -- one LLM call told to decompose. Two principles
from Anthropic's multi-agent guidance:

  * split by CONTEXT (disjoint information each worker needs), never by ROLE.
    "find the entities / find the bridge fact / compare" are three jobs on the
    SAME information -> identical briefs -> zero divergence. "everything about X /
    everything about Y" is disjoint -> the briefs MUST differ.
  * a spec has four fields: objective, output format, sources, boundaries.

n_workers is the router's OUTPUT (= number of units), not a config flag. The
decomposition comes out as TEXT because it is CONTROL FLOW (how many workers,
who's who) -- you cannot spawn a latent number of processes. The per-worker
brief (the payload) is what later becomes latent state.

Scoreable on its own against HotpotQA supporting_facts -- see
experiments/probe_router.py. Nothing here needs a GPU except route().
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


@dataclass
class WorkerSpec:
    unit: str
    objective: str = ""
    output_format: str = ""
    sources: str = ""
    boundaries: str = ""


@dataclass
class RouterPlan:
    units: List[str]
    property: str
    specs: List[WorkerSpec]
    raw: str = ""

    @property
    def n_workers(self) -> int:
        return len(self.specs)


ROUTER_SYSTEM = (
    "You are the lead of a research team. Decompose a question into INDEPENDENT "
    "lookup tasks that can run in parallel with no shared results. Split by what "
    "each worker needs to KNOW (disjoint information), never by what they DO with "
    "it. If answering requires one lookup's result to do the next (a sequential / "
    "bridge question), DO NOT split -- emit a single unit."
)


def build_router_prompt(question: str, args=None) -> List[dict]:
    """Question-only by design: the lead must identify the entities to look up
    from the question itself (that is the routing skill being tested), not pick
    from a provided document list."""
    user = f"""Question: {question}

Identify the independent entities/topics that must be looked up, then write a
task spec for each. Output EXACTLY this format and nothing else:

UNITS: <entity 1> | <entity 2> | ...
PROPERTY: <the shared property or comparison being asked>

WORKER 1
UNIT: <entity 1>
OBJECTIVE: <what to find about this entity, one sentence>
FORMAT: <how to report it>
SOURCES: only passages about <entity 1>
BOUNDARIES: do not research the other entities; do not compare

WORKER 2
UNIT: <entity 2>
... (one block per unit; if the question is a single-hop or bridge question,
emit exactly ONE worker)"""
    return [{"role": "system", "content": ROUTER_SYSTEM},
            {"role": "user", "content": user}]


def _field(block: str, key: str) -> str:
    m = re.search(rf"(?im)^\s*{key}\s*:\s*(.+)$", block)
    return m.group(1).strip() if m else ""


def parse_router_plan(text: str) -> RouterPlan:
    """Pull units + property + per-worker specs from the lead's decode. Tolerant
    of a <think> block, extra prose, and missing fields."""
    text = _THINK.sub("", text or "")
    if "</think>" in text:
        text = text.split("</think>")[-1]

    units: List[str] = []
    m = re.search(r"(?im)^\s*UNITS?\s*:\s*(.+)$", text)
    if m:
        units = [u.strip(" *\"") for u in re.split(r"[|,]", m.group(1)) if u.strip(" *\"")]

    prop = ""
    m = re.search(r"(?im)^\s*PROPERTY\s*:\s*(.+)$", text)
    if m:
        prop = m.group(1).strip()

    specs: List[WorkerSpec] = []
    blocks = re.split(r"(?im)^\s*WORKER\s*\d+\s*$", text)[1:]
    for b in blocks:
        unit = _field(b, "UNIT")
        if not unit and not any(_field(b, k) for k in ("OBJECTIVE", "SOURCES")):
            continue
        specs.append(WorkerSpec(
            unit=unit,
            objective=_field(b, "OBJECTIVE"),
            output_format=_field(b, "FORMAT"),
            sources=_field(b, "SOURCES"),
            boundaries=_field(b, "BOUNDARIES"),
        ))

    # reconcile the UNITS line and the per-worker UNIT fields
    if not units and specs:
        units = [s.unit for s in specs if s.unit]
    if units and not specs:                       # units listed but no blocks
        specs = [WorkerSpec(unit=u) for u in units]

    return RouterPlan(units=units, property=prop, specs=specs, raw=text)


def score_units(pred_units: List[str], gold_titles: List[str]) -> dict:
    """Grade the router against HotpotQA supporting_facts titles. A unit matches
    a title if either contains the other after normalization (entity vs title)."""
    from utils import squad_norm
    P = [squad_norm(u) for u in pred_units if u and squad_norm(u)]
    G = [squad_norm(t) for t in gold_titles if t and squad_norm(t)]

    def match(a: str, bs: List[str]) -> bool:
        return any(a == b or a in b or b in a for b in bs)

    tp_pred = sum(1 for p in P if match(p, G))
    covered = sum(1 for g in G if match(g, P))
    return {
        "count_match": len(P) == len(G),
        "precision": tp_pred / len(P) if P else 0.0,
        "recall": covered / len(G) if G else 0.0,
        "n_pred": len(P),
        "n_gold": len(G),
    }


def route(model, question: str, max_new_tokens: int = 384, args=None) -> RouterPlan:
    """Run the lead's decode (thinking off -> emit the spec directly) and parse it."""
    msgs = build_router_prompt(question, args=args)
    _, ids, mask, _ = model.prepare_chat_batch([msgs], add_generation_prompt=True,
                                               enable_thinking=False)
    gen, _ = model.generate_text_batch(ids, mask, max_new_tokens=max_new_tokens,
                                       temperature=0.7, top_p=0.95, past_key_values=None)
    return parse_router_plan(gen[0])
