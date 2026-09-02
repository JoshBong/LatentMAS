"""End-to-end wiring smoke for RoutedNLMethod on CPU, all three channels.

Same stance as test_routed_smoke.py: a tiny random GPT-2 behind a stand-in for
the ModelWrapper methods routed_nl calls. Answer quality is meaningless (random
weights); what is exercised is the real integration surface:

    orchestrator decode -> parse_briefs -> [kv] lead prefill -> expand_cache
    (batch-N broadcast VIEW of S0) -> ONE batched left-padded worker decode
    continuing the shared prefix -> NL judge over the workers' text.

The kv-channel worker decode runs real forward passes over the expanded cache,
so a broken expand/mask/position recipe fails here, not on the GPU box.
"""

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from methods.routed_nl import RoutedNLMethod
from methods.cache_ops import cache_length, expand_cache


class _FakeWrapper:
    """Stand-in for models.ModelWrapper (the methods routed_nl uses)."""

    def __init__(self):
        torch.manual_seed(0)
        self.vocab = 64
        cfg = GPT2Config(vocab_size=self.vocab, n_positions=1024, n_embd=32, n_layer=2, n_head=2)
        self.model = GPT2LMHeadModel(cfg).eval()
        self.device = torch.device("cpu")
        self.calls = {"batched_worker_decodes": 0, "worker_batch_sizes": [],
                      "worker_had_prefix": []}
        self.judge_prompts = []

    def _encode_one(self, messages):
        text = " ".join(m["content"] for m in messages)
        ids = [ord(c) % self.vocab for c in text][:24] or [0]
        return ids

    def prepare_chat_batch(self, batch_messages, add_generation_prompt=True,
                           enable_thinking=True, padding_side=None):
        rows = [self._encode_one(m) for m in batch_messages]
        width = max(len(r) for r in rows)
        ids, mask = [], []
        for r in rows:
            pad = width - len(r)
            if padding_side == "left":
                ids.append([0] * pad + r)
                mask.append([0] * pad + [1] * len(r))
            else:
                ids.append(r + [0] * pad)
                mask.append([1] * len(r) + [0] * pad)
        self.judge_prompts.append(batch_messages)
        return (["p"] * len(rows), torch.tensor(ids), torch.tensor(mask),
                [["t"]] * len(rows))

    @torch.no_grad()
    def generate_latent_batch(self, input_ids, attention_mask=None, *, latent_steps,
                              past_key_values=None):
        out = self.model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
        return out.past_key_values

    @torch.no_grad()
    def generate_text_batch(self, input_ids, attention_mask=None, *, max_new_tokens,
                            temperature=0.7, top_p=0.95, past_key_values=None):
        # Orchestrator or judge decode (past=None). Return parseable briefs; the
        # judge call just needs any text with a boxed answer.
        return ["Worker 1: find the director\nWorker 2: find their nationality\n"
                "Worker 3: cross-check the dates\n\\boxed{paris}"], None

    @torch.no_grad()
    def decode_text_batch_from_prefix(self, input_ids, attention_mask,
                                      past_key_values=None, *, max_new_tokens,
                                      temperature=0.0, top_p=0.95):
        # The batched fan-out. Run a REAL forward over the (possibly expanded)
        # cache so a bad batch/mask/position recipe explodes here.
        n = input_ids.shape[0]
        self.calls["batched_worker_decodes"] += 1
        self.calls["worker_batch_sizes"].append(n)
        self.calls["worker_had_prefix"].append(past_key_values is not None)
        if past_key_values is not None:
            plen = cache_length(past_key_values)
            full_mask = torch.cat(
                [torch.ones(n, plen, dtype=attention_mask.dtype), attention_mask], dim=-1)
            pos = (attention_mask.long().cumsum(-1) - 1 + plen).clamp_min(0)
            out = self.model(input_ids=input_ids, attention_mask=full_mask,
                             position_ids=pos, past_key_values=past_key_values,
                             use_cache=True)
            assert out.logits.shape[0] == n          # batched forward accepted
        return [f"worker {i + 1} findings" for i in range(n)]


class _Args:
    def __init__(self, channel="kv"):
        self.task = "hotpotqa"
        self.channel = channel


def _item():
    return {
        "question": "What nationality was the director of Inception?",
        "context_docs": [f"Document {i} body text." for i in range(6)],
        "gold": "paris",
    }


def test_kv_channel_broadcasts_one_prefix_to_a_single_batched_decode():
    wrapper = _FakeWrapper()
    method = RoutedNLMethod(wrapper, judger_max_new_tokens=4, num_workers=3,
                            args=_Args("kv"))
    res = method.run_item(_item())
    assert res["channel"] == "kv"
    assert res["correct"] is True and res["prediction"].lower() == "paris"
    assert len(res["briefs"]) == 3 and res["n_briefs_parsed"] == 3
    # ONE batched decode of all 3 workers, WITH the shared prefix
    assert wrapper.calls["batched_worker_decodes"] == 1
    assert wrapper.calls["worker_batch_sizes"] == [3]
    assert wrapper.calls["worker_had_prefix"] == [True]
    # docs were paid once, into the prefix
    assert res["s0_prompt_tokens"] > 0 and res["s0_cache_len"] > 0
    assert len(res["agents"]) == 5                      # orchestrator + 3 workers + judge
    assert res["judge_ctx_tokens"] > 0


def test_text_channel_runs_without_any_prefix():
    wrapper = _FakeWrapper()
    method = RoutedNLMethod(wrapper, judger_max_new_tokens=4, num_workers=3,
                            args=_Args("text"))
    res = method.run_item(_item())
    assert res["channel"] == "text"
    assert wrapper.calls["worker_had_prefix"] == [False]
    assert res["s0_prompt_tokens"] == 0 and res["s0_cache_len"] == 0


def test_nodocs_killswitch_withholds_documents_everywhere():
    wrapper = _FakeWrapper()
    method = RoutedNLMethod(wrapper, judger_max_new_tokens=4, num_workers=3,
                            args=_Args("nodocs"))
    res = method.run_item(_item())
    assert res["channel"] == "nodocs"
    assert res["s0_prompt_tokens"] == 0
    # no document text may reach any worker prompt
    for batch in wrapper.judge_prompts:
        for msgs in batch:
            for m in msgs:
                assert "Document 0 body" not in m["content"]


def test_unknown_channel_rejected():
    with pytest.raises(ValueError):
        RoutedNLMethod(_FakeWrapper(), num_workers=2, args=_Args("latent"))


def test_expand_cache_is_a_view_and_batches_forward():
    torch.manual_seed(0)
    cfg = GPT2Config(vocab_size=64, n_positions=128, n_embd=32, n_layer=2, n_head=2)
    model = GPT2LMHeadModel(cfg).eval()
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.no_grad():
        base = model(input_ids=ids, use_cache=True).past_key_values
        S3 = expand_cache(base, 3)
        assert cache_length(S3) == cache_length(base)
        # a genuine broadcast view: no storage copy for the shared prefix
        from methods.cache_ops import _to_legacy
        b0 = _to_legacy(base)[0][0]
        e0 = _to_legacy(S3)[0][0]
        assert e0.shape[0] == 3 and e0.data_ptr() == b0.data_ptr()
        # continuing all 3 rows from the expanded cache must equal continuing
        # batch-1 from the original, row for row
        nxt = torch.tensor([[7], [7], [7]])
        pos = torch.full((3, 1), 5)
        mask3 = torch.ones(3, 6, dtype=torch.long)
        out3 = model(input_ids=nxt, attention_mask=mask3, position_ids=pos,
                     past_key_values=S3, use_cache=True)
        out1 = model(input_ids=nxt[:1], attention_mask=mask3[:1], position_ids=pos[:1],
                     past_key_values=base, use_cache=True)
        assert torch.allclose(out3.logits[0], out1.logits[0], atol=1e-5)
        assert torch.allclose(out3.logits[1], out1.logits[0], atol=1e-5)


def test_orchestrator_underproduction_fails_loudly():
    class _Lazy(_FakeWrapper):
        @torch.no_grad()
        def generate_text_batch(self, *a, **k):
            return ["Worker 1: only one brief"], None
    method = RoutedNLMethod(_Lazy(), judger_max_new_tokens=4, num_workers=3,
                            args=_Args("kv"))
    with pytest.raises(RuntimeError):
        method.run_item(_item())
