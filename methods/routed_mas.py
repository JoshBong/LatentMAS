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
from typing import Dict, List, Optional

import torch

from . import Agent
from models import ModelWrapper
from utils import extract_answer, extract_gsm8k_answer, squad_norm, token_f1
from prompts_routed import (
    build_orchestrator_prompt,
    build_routed_judger,
    build_routed_lead,
    build_routed_worker,
    parse_briefs,
    parse_doc_assignments,
    worker_doc_slice,
)
from methods.cache_ops import (
    cache_concat,
    cache_length,
    cache_reindex,
    cache_suffix,
    clone_cache,
    noise_cache,
    rope_inv_freq,
)

ARMS = ("normal", "judge_blind", "empty_cache", "noise_blocks")

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
        # The worker latent encoding IS the mechanism; latent_steps == 0 silently
        # turns it off and every downstream number becomes noise. Fail loudly.
        if latent_steps <= 0:
            raise ValueError(
                f"routed_mas requires latent_steps > 0 (the worker latent channel is "
                f"the mechanism under test); got latent_steps={latent_steps}"
            )
        self.latent_steps = latent_steps
        # Kill-switch ablation arm (see ARMS / experiments/run_suite.py).
        self.arm = getattr(args, "arm", "normal")
        if self.arm not in ARMS:
            raise ValueError(f"unknown arm {self.arm!r}; choose from {ARMS}")
        self.judger_max_new_tokens = judger_max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.method_name = "routed_mas"
        self.task = getattr(args, "task", None)
        # 'orchestrated' = the lead decodes a brief per worker (route out with
        # text); 'static' = fixed contiguous doc split, generic briefs.
        self.routing = getattr(args, "routing", "orchestrated")
        # Backstop only -- with enable_thinking=False the orchestrator hits EOS well
        # inside this. If experiments/inspect_orchestrator.py shows truncation, raise
        # it; a cap should never be the normal terminator.
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
        # math: pull the boxed value / final number
        if self.task in ("gsm8k", "aime2024", "aime2025"):
            m = re.search(r"\\boxed\{(.+?)\}", text, flags=re.DOTALL)
            if m:
                return m.group(1).strip()
            got = extract_gsm8k_answer(text)
            if got:
                return got
        # free-form (hotpotqa): strip <think>, take boxed / 'answer is' / first sentence
        return extract_answer(text)

    def _score(self, pred_text: str, gold: str) -> tuple:
        """Strict, extraction-based EM on BOTH paths.

        Score the EXTRACTED answer, never substring-match the raw response: a
        rambling judge that merely names the gold entity among six others must not
        get credit (the old free-form path did `answer_hit(pred_text, gold)`, whole-
        word recall over the whole response -- inflated, and incomparable to
        published HotpotQA). This is SQuAD-normalized EM, matching probe.py::em, so
        the two harnesses finally agree. token_f1 stays as the softer secondary
        metric (computed in _run_item, on the same extracted pred).
        """
        pred = self._extract(pred_text)
        if not gold:
            return pred, False
        ok = squad_norm(pred) == squad_norm(gold)
        return pred, bool(ok)

    def _ntok(self, text: str) -> int:
        """Token count of a decoded string, for the accounting summary. Uses the
        model tokenizer; falls back to a whitespace count for the CPU test stub
        (which has no tokenizer). Prompt tokens are counted exactly from the input
        ids/masks at each phase, not here."""
        if not text:
            return 0
        tok = getattr(self.model, "tokenizer", None)
        if tok is None:
            return len(text.split())
        return len(tok(text, add_special_tokens=False).input_ids)

    # ------------------------------------------------------------------- worker
    @torch.no_grad()
    def _run_item(self, item: Dict) -> Dict:
        model = self.model
        question = item["question"]
        context_docs: Optional[List[str]] = item.get("context_docs")
        arm = self.arm
        prompt_tokens = 0        # exact: summed from input ids/masks at each phase
        gen_tokens = 0           # decoded text tokens (orchestrator + judge)

        # -- Phase 1: lead encodes the shared question -> base cache S0 ----------
        _, lead_ids, lead_mask, _ = model.prepare_chat_batch(
            [build_routed_lead(question, self.args)], add_generation_prompt=True
        )
        prompt_tokens += int(lead_mask.sum().item())
        S0 = model.generate_latent_batch(
            lead_ids, lead_mask, latent_steps=0, past_key_values=None
        )
        base_len = cache_length(S0)

        # -- Phase 1b: orchestrator decodes one brief (+ doc assignment) per worker --
        # empty_cache drops the workers entirely, so it needs no briefs -- skip the
        # orchestrator (and its padded-brief guard) for that control arm.
        briefs = None
        n_briefs_parsed = None
        doc_assign = None
        if self.routing == "orchestrated" and arm != "empty_cache":
            _, o_ids, o_mask, _ = model.prepare_chat_batch(
                [build_orchestrator_prompt(question, self.n_workers, context_docs, self.args)],
                add_generation_prompt=True,
                enable_thinking=False,   # emit briefs directly; don't spend the budget thinking
            )
            prompt_tokens += int(o_mask.sum().item())
            o_gen, _ = model.generate_text_batch(
                o_ids, o_mask, max_new_tokens=self.orchestrator_max_new_tokens,
                temperature=self.temperature, top_p=self.top_p, past_key_values=None,
            )
            gen_tokens += self._ntok(o_gen[0])
            briefs, n_briefs_parsed = parse_briefs(o_gen[0], self.n_workers)
            if context_docs:
                doc_assign = parse_doc_assignments(o_gen[0], self.n_workers, len(context_docs))
            # A padded brief = a worker with no real subtask -> the fan-out is
            # compromised and any number it produces is noise. FAIL the cell rather
            # than warn-and-continue: a shipped log with the mechanism half-off is
            # exactly the failure this guard exists to prevent.
            if n_briefs_parsed < self.n_workers:
                raise RuntimeError(
                    f"[routed_mas] orchestrator emitted only {n_briefs_parsed}/{self.n_workers} "
                    f"real briefs for question={question!r}; refusing to run padded workers "
                    f"(fix the orchestrator prompt / model or drop this item)."
                )

        # -- Phase 2: fan out; each worker runs from an independent clone of S0 --
        # empty_cache: judge sees S0 + question text only, workers dropped. If that
        # ties `routed`, the workers contribute nothing and every cell is noise.
        worker_caches = []
        worker_embeds = []
        traces = []
        if arm != "empty_cache":
            for w_idx, worker in enumerate(self.workers):
                brief_text = briefs[w_idx] if briefs else None
                # Route the EVIDENCE, not just the brief: use the orchestrator's doc
                # assignment when it gave one; else fall back to the contiguous slice.
                if doc_assign is not None and doc_assign[w_idx] is not None:
                    w_docs = [context_docs[i] for i in doc_assign[w_idx]]
                elif context_docs:
                    w_docs = worker_doc_slice(context_docs, w_idx, self.n_workers)
                else:
                    w_docs = []
                worker_msgs = build_routed_worker(
                    question, w_idx, self.n_workers, docs=w_docs,
                    brief=brief_text, args=self.args,
                )
                _, b_ids, b_mask, _ = model.prepare_chat_batch([worker_msgs], add_generation_prompt=True)
                prompt_tokens += int(b_mask.sum().item())
                S0_w = clone_cache(S0)
                S_w, emb = model.generate_latent_batch_hidden_state(
                    b_ids, b_mask, latent_steps=self.latent_steps, past_key_values=S0_w
                )
                worker_caches.append(S_w)
                worker_embeds.append(emb)
                traces.append({"name": worker.name, "role": worker.role, "brief": brief_text,
                               "docs": (doc_assign[w_idx] if doc_assign else None),
                               "latent_steps": self.latent_steps, "output": ""})

        # -- Phase 3: combine -> [S0] ++ each worker's own tokens, then judge ----
        # Each worker was encoded starting at base_len, so their keys carry
        # overlapping RoPE positions. Shift worker w by the total length of the
        # workers before it -> the stitched cache is positionally a sequential read.
        suffixes = []
        offset = 0
        for S_w in worker_caches:
            suf = cache_suffix(S_w, base_len)
            # noise_blocks: shape/norm-matched noise in place of the real suffix,
            # re-indexed identically -> only information removed, not positions.
            if arm == "noise_blocks":
                suf = noise_cache(suf)
            if self.reindex and offset > 0:
                suf = cache_reindex(suf, offset, inv_freq=self.rope_inv_freq)
            suffixes.append(suf)
            offset += cache_length(suf)
        combined = cache_concat([S0] + suffixes) if suffixes else S0

        _, j_ids, j_mask, _ = model.prepare_chat_batch(
            [build_routed_judger(question, self.args, blind=(arm == "judge_blind"))],
            add_generation_prompt=True,
        )
        prompt_tokens += int(j_mask.sum().item())
        # Manual decode over the stitched cache -- HF generate() can't take a
        # non-prefix past (see ModelWrapper.decode_from_cache).
        final_text = model.decode_from_cache(
            j_ids, combined, max_new_tokens=self.judger_max_new_tokens
        ).strip()
        gen_tokens += self._ntok(final_text)

        pred, ok = self._score(final_text, item.get("gold", ""))
        f1 = token_f1(pred, item.get("gold", ""))
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
            "arm": arm,
            "briefs": briefs,
            "doc_assign": doc_assign,
            "n_briefs_parsed": n_briefs_parsed,
            "worker_divergence": self._divergence(worker_embeds),
            "n_workers": self.n_workers,
            # mechanism-on + token accounting (surfaced into summary.json)
            "latent_steps": self.latent_steps,
            "prompt_tokens": prompt_tokens,
            "gen_tokens": gen_tokens,
            "latent_tokens": self.latent_steps * len(worker_caches),
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
