"""Routed fan-out over the latent (KV) channel.

The contrast with LatentMAS in one sentence: LatentMAS threads ONE growing cache
through a linear agent chain (every agent sees every predecessor); routed_mas
gives each worker an INDEPENDENT clone of a shared base cache plus its OWN
targeted brief, then concatenates their contributions for the judge.

Three phases (methods/cache_ops.py does the surgery):

    1. lead    : encode the shared question once            -> base cache S0
    2. fan out : for each worker, clone S0, run it from that clone on its brief
                 (workers never see each other)             -> S_w, embeddings
    3. combine : judger decodes from  [S0] ++ [each worker's own tokens]

We deliberately process ONE item at a time (batch of 1) in this first cut: it
keeps the cache surgery exact (no left/right-padding to reconcile across workers).
Batching is a later optimization, not a correctness requirement.

GPU note: the cache primitives are CPU-tested (tests/test_cache_ops.py); this
end-to-end method requires a real model (Qwen) and has NOT been run yet.
"""

from __future__ import annotations

import argparse
import re
import string
from typing import Dict, List, Optional

import torch

from . import Agent
from models import ModelWrapper
from utils import normalize_answer, extract_gsm8k_answer
from prompts_routed import (
    build_orchestrator_prompt,
    build_routed_judger,
    build_routed_lead,
    build_routed_worker,
    parse_briefs,
)
from methods.cache_ops import (
    cache_concat,
    cache_length,
    cache_reindex,
    cache_suffix,
    clone_cache,
    rope_inv_freq,
)

_ARTICLES = re.compile(r"\b(a|an|the)\b")


def _squad_norm(s: str) -> str:
    """SQuAD/HotpotQA answer normalization: lowercase, drop punctuation and
    articles, collapse whitespace. Used for both EM and F1 so they agree."""
    s = (s or "").lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


class RoutedMASMethod:
    def __init__(
        self,
        model: ModelWrapper,
        *,
        latent_steps: int = 10,
        judger_max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        num_workers: int = 3,
        generate_bs: int = 1,
        args: argparse.Namespace = None,
    ) -> None:
        self.args = args
        self.model = model
        self.latent_steps = latent_steps
        self.judger_max_new_tokens = judger_max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.method_name = "routed_mas"
        self.task = getattr(args, "task", None)
        # 'orchestrated' = the lead decodes a brief per worker (route out with
        # text); 'static' = fixed contiguous doc split, generic briefs.
        self.routing = getattr(args, "routing", "orchestrated")
        self.orchestrator_max_new_tokens = getattr(args, "orchestrator_max_new_tokens", 256)
        # De-entangle RoPE positions in the stitched judge cache (default on).
        self.reindex = not getattr(args, "no_reindex", False)
        hf = getattr(model, "HF_model", None) or getattr(model, "model", None)
        # Read the model's actual rotary frequencies (raises if absent) -- only
        # when reindexing, since a wrong value corrupts the cache. --no_reindex
        # skips this entirely.
        self.rope_inv_freq = rope_inv_freq(hf) if self.reindex else None

        # Workers = the non-judger custom agents if supplied, else N generic workers.
        custom = getattr(args, "custom_agents", None)
        if custom:
            self.workers: List[Agent] = [a for a in custom if a.role != "judger"]
        else:
            self.workers = [Agent(name=f"Worker{i + 1}", role=f"worker{i + 1}")
                            for i in range(num_workers)]
        self.n_workers = len(self.workers)

    # ------------------------------------------------------------------ scoring
    def _extract(self, text: str) -> str:
        m = re.search(r"\\boxed\{(.+?)\}", text, flags=re.DOTALL)
        if m:
            return m.group(1).strip()
        if self.task in ("gsm8k", "aime2024", "aime2025"):
            got = extract_gsm8k_answer(text)
            if got:
                return got
        # free-form (hotpotqa): last non-empty line
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return lines[-1] if lines else text.strip()

    def _score(self, pred_text: str, gold: str) -> tuple:
        """Exact match after SQuAD/HotpotQA normalization (the standard EM).

        No substring leniency: 'pred in gold' would score pred='a' correct
        against gold='canada'. Partial credit lives in F1, not here.
        """
        pred = _squad_norm(self._extract(pred_text))
        gold = _squad_norm(gold)
        if not gold:
            return pred, False
        return pred, (pred == gold)

    @staticmethod
    def _f1(pred: str, gold: str) -> float:
        """HotpotQA-style token-overlap F1 between the extracted answer and gold."""
        p = _squad_norm(pred).split()
        g = _squad_norm(gold).split()
        if not p or not g:
            return float(p == g)
        common = 0
        gg = list(g)
        for tok in p:
            if tok in gg:
                common += 1
                gg.remove(tok)
        if common == 0:
            return 0.0
        prec, rec = common / len(p), common / len(g)
        return 2 * prec * rec / (prec + rec)

    # ------------------------------------------------------------------- worker
    @torch.no_grad()
    def _run_item(self, item: Dict) -> Dict:
        model = self.model
        question = item["question"]
        context_docs: Optional[List[str]] = item.get("context_docs")

        # -- Phase 1: lead encodes the shared question -> base cache S0 ----------
        _, lead_ids, lead_mask, _ = model.prepare_chat_batch(
            [build_routed_lead(question, self.args)], add_generation_prompt=True
        )
        S0 = model.generate_latent_batch(
            lead_ids, lead_mask, latent_steps=0, past_key_values=None
        )
        base_len = cache_length(S0)

        # -- Phase 1b: orchestrator decodes one brief per worker (route OUT = text) --
        briefs = None
        n_briefs_parsed = None
        if self.routing == "orchestrated":
            _, o_ids, o_mask, _ = model.prepare_chat_batch(
                [build_orchestrator_prompt(question, self.n_workers, context_docs, self.args)],
                add_generation_prompt=True,
            )
            o_gen, _ = model.generate_text_batch(
                o_ids, o_mask, max_new_tokens=self.orchestrator_max_new_tokens,
                temperature=self.temperature, top_p=self.top_p, past_key_values=None,
            )
            # n_briefs_parsed < n_workers => some briefs were padded (generic) =>
            # workers won't differentiate; logged so it can't hide as low divergence.
            briefs, n_briefs_parsed = parse_briefs(o_gen[0], self.n_workers)

        # -- Phase 2: fan out; each worker runs from an independent clone of S0 --
        worker_caches = []
        worker_embeds = []
        traces = []
        for w_idx, worker in enumerate(self.workers):
            brief_text = briefs[w_idx] if briefs else None
            worker_msgs = build_routed_worker(
                question, w_idx, self.n_workers, context_docs=context_docs,
                brief=brief_text, args=self.args,
            )
            _, b_ids, b_mask, _ = model.prepare_chat_batch([worker_msgs], add_generation_prompt=True)
            S0_w = clone_cache(S0)
            S_w, emb = model.generate_latent_batch_hidden_state(
                b_ids, b_mask, latent_steps=self.latent_steps, past_key_values=S0_w
            )
            worker_caches.append(S_w)
            worker_embeds.append(emb)
            traces.append({"name": worker.name, "role": worker.role, "brief": brief_text,
                           "latent_steps": self.latent_steps, "output": ""})

        # -- Phase 3: combine -> [S0] ++ each worker's own tokens, then judge ----
        # Each worker was encoded starting at base_len, so their keys carry
        # overlapping RoPE positions. Shift worker w by the total length of the
        # workers before it -> the stitched cache is positionally a sequential read.
        suffixes = []
        offset = 0
        for S_w in worker_caches:
            suf = cache_suffix(S_w, base_len)
            if self.reindex and offset > 0:
                suf = cache_reindex(suf, offset, inv_freq=self.rope_inv_freq)
            suffixes.append(suf)
            offset += cache_length(suf)
        combined = cache_concat([S0] + suffixes)

        _, j_ids, j_mask, _ = model.prepare_chat_batch(
            [build_routed_judger(question, self.args)], add_generation_prompt=True
        )
        # Manual decode over the stitched cache -- HF generate() can't take a
        # non-prefix past (see ModelWrapper.decode_from_cache).
        final_text = model.decode_from_cache(
            j_ids, combined, max_new_tokens=self.judger_max_new_tokens
        ).strip()

        pred, ok = self._score(final_text, item.get("gold", ""))
        f1 = self._f1(pred, item.get("gold", ""))
        traces.append({"name": "Judger", "role": "judger", "output": final_text})

        return {
            "question": question,
            "gold": item.get("gold", ""),
            "solution": item.get("solution", ""),
            "prediction": pred,
            "raw_prediction": final_text,
            "agents": traces,
            "correct": ok,
            "f1": f1,
            "routing": self.routing,
            "briefs": briefs,
            "n_briefs_parsed": n_briefs_parsed,
            "worker_divergence": self._divergence(worker_embeds),
            "n_workers": self.n_workers,
        }

    # -------------------------------------------------------------- diagnostic
    @staticmethod
    def _divergence(worker_embeds: List[torch.Tensor]) -> Optional[float]:
        """Mean pairwise (1 - cosine) between workers' latent outputs.

        ~0 => the workers produced near-identical states: routing did not
        differentiate them, and any flat accuracy is boring, not a finding.
        Larger => the briefs actually pulled the workers apart. This is the
        number that turns 'it didn't help' into 'it didn't help *and here is
        whether routing even happened*'.
        """
        if len(worker_embeds) < 2:
            return None
        vecs = [e.float().mean(dim=1).flatten() for e in worker_embeds]  # one vector per worker
        sims = []
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                sims.append(torch.cosine_similarity(vecs[i], vecs[j], dim=0).item())
        return float(1.0 - sum(sims) / len(sims))

    # ------------------------------------------------------------------- batch
    def run_batch(self, items: List[Dict]) -> List[Dict]:
        return [self._run_item(item) for item in items]

    def run_item(self, item: Dict) -> Dict:
        return self._run_item(item)
