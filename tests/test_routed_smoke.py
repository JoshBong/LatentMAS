"""End-to-end wiring smoke test for RoutedMASMethod on CPU.

No Qwen, no download: a tiny random GPT-2 behind a minimal stand-in that mimics
the four ModelWrapper methods routed_mas calls. This does NOT test answer quality
(random weights => gibberish). It tests that the three phases compose:

    lead -> S0 -> clone per worker -> run -> cache_suffix -> cache_concat ->
    judger decodes from the CONCATENATED cache

The concatenated-cache decode (correct cache_position over a stitched-together
past) is the real integration risk beyond the primitives, so exercising it end
to end with real forward passes is the point.
"""

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from methods.routed_mas import RoutedMASMethod
from methods.cache_ops import cache_length


class _FakeWrapper:
    """Minimal stand-in for models.ModelWrapper (the 4 methods routed_mas uses)."""

    def __init__(self):
        torch.manual_seed(0)
        self.vocab = 64
        cfg = GPT2Config(vocab_size=self.vocab, n_positions=1024, n_embd=32, n_layer=2, n_head=2)
        self.model = GPT2LMHeadModel(cfg).eval()
        self.device = torch.device("cpu")

    def _encode(self, messages):
        text = " ".join(m["content"] for m in messages)
        ids = [ord(c) % self.vocab for c in text][:24] or [0]
        return torch.tensor([ids])

    def prepare_chat_batch(self, batch_messages, add_generation_prompt=True):
        ids = self._encode(batch_messages[0])
        mask = torch.ones_like(ids)
        return ["p"], ids, mask, [["t"]]

    @torch.no_grad()
    def generate_latent_batch(self, input_ids, attention_mask=None, *, latent_steps, past_key_values=None):
        out = self.model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
        return out.past_key_values

    @torch.no_grad()
    def generate_latent_batch_hidden_state(self, input_ids, attention_mask=None, *, latent_steps, past_key_values=None):
        out = self.model(input_ids=input_ids, past_key_values=past_key_values,
                         use_cache=True, output_hidden_states=True)
        return out.past_key_values, out.hidden_states[0]      # [B, seq, D] input embeddings

    @torch.no_grad()
    def generate_text_batch(self, input_ids, attention_mask=None, *, max_new_tokens,
                            temperature=0.7, top_p=0.95, past_key_values=None):
        # Prove the judger prompt is consumed on top of the CONCATENATED cache at
        # the right positions (a plain forward; the real wrapper uses generate()).
        past_len = cache_length(past_key_values) if past_key_values is not None else 0
        pos = torch.arange(past_len, past_len + input_ids.shape[-1], dtype=torch.long).unsqueeze(0)
        full_mask = torch.ones(1, past_len + input_ids.shape[-1], dtype=torch.long)
        out = self.model(input_ids=input_ids, attention_mask=full_mask,
                         past_key_values=past_key_values, position_ids=pos, use_cache=True)
        assert out.logits.shape[1] == input_ids.shape[-1]      # stitched cache accepted
        return ["reasoning... \\boxed{paris}"], None


class _Args:
    task = "hotpotqa"
    custom_agents = None


def test_routed_pipeline_runs_end_to_end():
    method = RoutedMASMethod(
        _FakeWrapper(), latent_steps=0, judger_max_new_tokens=4,
        num_workers=3, args=_Args(),
    )
    item = {
        "question": "What nationality was the director of Inception?",
        "context_docs": [f"Document {i} body text about something." for i in range(6)],
        "gold": "paris",
    }
    res = method.run_item(item)

    # the pipeline completed and produced the expected record shape
    assert res["n_workers"] == 3
    assert res["prediction"] == "paris" and res["correct"] is True   # from the stubbed decode
    assert isinstance(res["worker_divergence"], float)               # 3 workers -> diagnostic computed
    assert len(res["agents"]) == 4                                   # 3 workers + judger


def test_divergence_none_for_single_worker():
    method = RoutedMASMethod(_FakeWrapper(), latent_steps=0, num_workers=1, args=_Args())
    res = method.run_item({"question": "q", "context_docs": ["a", "b"], "gold": "z"})
    assert res["worker_divergence"] is None                         # needs >=2 workers
