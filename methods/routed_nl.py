"""Decomposition MAS: latent broadcast DOWN, natural-language synthesis UP.

The topology Josh sketched:

    prompt ("find thing 1 and thing 2 in relation to thing 3")
      -> orchestrator decodes one subtask per worker (text; routing is discrete)
      -> shared context (all documents + question) prefilled ONCE into a KV
         prefix S0, broadcast to every worker as a batch-N expanded view
      -> workers decode their findings IN ONE BATCH (true parallelism: one
         forward pass per decode step for all workers), each continuing the
         same prefix with its own subtask turn
      -> judge reads the workers' TEXT findings and answers in plain language

Contrast with the repo's other methods, in one line each:
  latent_mas  one growing cache through a sequential role chain (same task 4x)
  routed_mas  doc-partitioned workers, latent-step encodings, judge decodes over
              a STITCHED cache (reindex + concat -- the fragile part)
  routed_nl   subtask-decomposed workers over ONE shared prefix, judge reads text

What this design deletes, on purpose: cache_suffix / cache_reindex /
cache_concat / decode_from_cache. Every forward here is a positionally honest
continuation of a real prefix, so there is nothing to stitch and nothing to
re-index. The mechanism under test is the PREFIX BROADCAST: the documents are
encoded once and reach the workers only through the KV channel.

Channels (--channel) give the controlled comparison, same topology throughout:
  kv      docs live ONLY in the shared prefix; worker turns carry just the brief
  text    docs repeated as text in every worker's own prompt (no prefix at all)
  nodocs  kill-switch: docs reach nobody -- if kv ~= nodocs, the broadcast
          carries nothing and the kv numbers are noise

The honest cost story this enables: `kv` pays the documents once
(s0_prompt_tokens), `text` pays them once per worker; the judge context is a
few sentences per worker in both. Accounting fields per item make the claim
checkable from trials.jsonl alone.
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional

import torch

from models import ModelWrapper
from utils import score_prediction, token_f1
from prompts_routed import (
    build_nl_judger,
    build_nl_lead,
    build_nl_worker_standalone,
    build_nl_worker_turn,
    build_orchestrator_prompt,
    parse_briefs,
)
from methods.cache_ops import cache_length, expand_cache

CHANNELS = ("kv", "text", "nodocs")


class RoutedNLMethod:
    def __init__(
        self,
        model: ModelWrapper,
        *,
        judger_max_new_tokens: int = 256,
        worker_max_new_tokens: int = 160,
        temperature: float = 0.7,
        top_p: float = 0.95,
        num_workers: int = 3,
        args: argparse.Namespace = None,
    ) -> None:
        self.args = args
        self.model = model
        self.task = getattr(args, "task", None)
        self.channel = getattr(args, "channel", "kv")
        if self.channel not in CHANNELS:
            raise ValueError(f"unknown channel {self.channel!r}; choose from {CHANNELS}")
        self.n_workers = num_workers
        self.judger_max_new_tokens = judger_max_new_tokens
        self.worker_max_new_tokens = worker_max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.orchestrator_max_new_tokens = getattr(args, "orchestrator_max_new_tokens", 256)
        self.method_name = "routed_nl"

    # ------------------------------------------------------------------ scoring
    def _score(self, pred_text: str, gold: str) -> tuple:
        pred, ok, _f1 = score_prediction(pred_text, gold, self.task)
        return pred, ok

    def _ntok(self, text: str) -> int:
        if not text:
            return 0
        tok = getattr(self.model, "tokenizer", None)
        if tok is None:
            return len(text.split())
        return len(tok(text, add_special_tokens=False).input_ids)

    # -------------------------------------------------------------------- item
    @torch.no_grad()
    def _run_item(self, item: Dict) -> Dict:
        model = self.model
        question = item["question"]
        context_docs: Optional[List[str]] = item.get("context_docs") or []
        channel = self.channel
        prompt_tokens = 0
        gen_tokens = 0
        s0_prompt_tokens = 0

        # -- Phase 1: orchestrator decodes one subtask per worker (text) --------
        _, o_ids, o_mask, _ = model.prepare_chat_batch(
            [build_orchestrator_prompt(question, self.n_workers,
                                       context_docs if channel != "nodocs" else None,
                                       self.args)],
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_tokens += int(o_mask.sum().item())
        o_gen, _ = model.generate_text_batch(
            o_ids, o_mask, max_new_tokens=self.orchestrator_max_new_tokens,
            temperature=self.temperature, top_p=self.top_p, past_key_values=None,
        )
        gen_tokens += self._ntok(o_gen[0])
        briefs, n_briefs_parsed = parse_briefs(o_gen[0], self.n_workers)
        if n_briefs_parsed < self.n_workers:
            # Same fail-loudly stance as routed_mas: padded briefs collapse the
            # decomposition and every downstream number becomes noise.
            raise RuntimeError(
                f"[routed_nl] orchestrator emitted only {n_briefs_parsed}/{self.n_workers} "
                f"real briefs for question={question!r}; refusing to run padded workers."
            )

        # -- Phase 2: fan out, ONE batched decode for all workers ---------------
        if channel == "kv":
            # Shared prefix: all docs + question, encoded exactly once.
            _, lead_ids, lead_mask, _ = model.prepare_chat_batch(
                [build_nl_lead(question, context_docs)], add_generation_prompt=False,
            )
            s0_prompt_tokens = int(lead_mask.sum().item())
            prompt_tokens += s0_prompt_tokens          # paid ONCE, not per worker
            S0 = model.generate_latent_batch(
                lead_ids, lead_mask, latent_steps=0, past_key_values=None
            )
            base_len = cache_length(S0)
            S0_batch = expand_cache(S0, self.n_workers)  # broadcast view, no copy

            worker_msgs = [build_nl_worker_turn(briefs[w], w, self.n_workers)
                           for w in range(self.n_workers)]
            _, w_ids, w_mask, _ = model.prepare_chat_batch(
                worker_msgs, add_generation_prompt=True, padding_side="left",
            )
            prompt_tokens += int(w_mask.sum().item())
            worker_texts = model.decode_text_batch_from_prefix(
                w_ids, w_mask, S0_batch,
                max_new_tokens=self.worker_max_new_tokens,
                temperature=self.temperature, top_p=self.top_p,
            )
        else:
            # text / nodocs: no prefix anywhere; docs (if any) arrive as text in
            # each worker's own standalone prompt.
            base_len = 0
            docs_for_workers = context_docs if channel == "text" else []
            worker_msgs = [
                build_nl_worker_standalone(question, briefs[w], w, self.n_workers,
                                           docs_for_workers)
                for w in range(self.n_workers)
            ]
            _, w_ids, w_mask, _ = model.prepare_chat_batch(
                worker_msgs, add_generation_prompt=True, padding_side="left",
            )
            prompt_tokens += int(w_mask.sum().item())
            worker_texts = model.decode_text_batch_from_prefix(
                w_ids, w_mask, None,
                max_new_tokens=self.worker_max_new_tokens,
                temperature=self.temperature, top_p=self.top_p,
            )
        gen_tokens += sum(self._ntok(t) for t in worker_texts)

        # -- Phase 3: judge reads the findings as TEXT --------------------------
        _, j_ids, j_mask, _ = model.prepare_chat_batch(
            [build_nl_judger(question, worker_texts, self.args)],
            add_generation_prompt=True,
        )
        judge_ctx_tokens = int(j_mask.sum().item())
        prompt_tokens += judge_ctx_tokens
        j_gen, _ = model.generate_text_batch(
            j_ids, j_mask, max_new_tokens=self.judger_max_new_tokens,
            temperature=self.temperature, top_p=self.top_p, past_key_values=None,
        )
        final_text = j_gen[0].strip()
        gen_tokens += self._ntok(final_text)

        pred, ok = self._score(final_text, item.get("gold", ""))
        f1 = token_f1(pred, item.get("gold", ""))

        traces = [{"name": "Orchestrator", "role": "orchestrator", "output": o_gen[0]}]
        for w, t in enumerate(worker_texts):
            traces.append({"name": f"Worker{w + 1}", "role": f"worker{w + 1}",
                           "brief": briefs[w], "output": t})
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
            "channel": channel,
            "briefs": briefs,
            "n_briefs_parsed": n_briefs_parsed,
            "n_workers": self.n_workers,
            "worker_texts": worker_texts,
            # accounting: the cost claim, checkable per item
            "prompt_tokens": prompt_tokens,
            "gen_tokens": gen_tokens,
            "s0_prompt_tokens": s0_prompt_tokens,   # docs paid once (kv) or 0
            "s0_cache_len": base_len,
            "judge_ctx_tokens": judge_ctx_tokens,
        }

    # ------------------------------------------------------------------- batch
    def run_batch(self, items: List[Dict]) -> List[Dict]:
        return [self._run_item(item) for item in items]

    def run_item(self, item: Dict) -> Dict:
        return self._run_item(item)
