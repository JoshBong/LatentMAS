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

    def prepare_chat_batch(self, batch_messages, add_generation_prompt=True, enable_thinking=True):
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
        # Orchestrator decode (past=None). Return parseable worker briefs.
        return ["Worker 1: find the director\nWorker 2: find their nationality\n"
                "Worker 3: cross-check"], None

    @torch.no_grad()
    def decode_from_cache(self, input_ids, past_key_values, max_new_tokens=64, eos_id=None):
        # Judge decode over the CONCATENATED cache -- exercise that a stitched,
        # RoPE-free cache is consumed at the continuing positions (a plain forward).
        past_len = cache_length(past_key_values) if past_key_values is not None else 0
        pos = torch.arange(past_len, past_len + input_ids.shape[-1], dtype=torch.long).unsqueeze(0)
        full_mask = torch.ones(1, past_len + input_ids.shape[-1], dtype=torch.long)
        out = self.model(input_ids=input_ids, attention_mask=full_mask,
                         past_key_values=past_key_values, position_ids=pos, use_cache=True)
        assert out.logits.shape[1] == input_ids.shape[-1]      # stitched cache accepted
        return "The answer is Paris."


class _Args:
    def __init__(self, routing="orchestrated", arm="normal"):
        self.task = "hotpotqa"
        self.custom_agents = None
        self.routing = routing
        self.arm = arm
        self.no_reindex = True          # GPT-2 has no RoPE; reindex is tested separately on Llama


def test_orchestrated_pipeline_runs_end_to_end():
    method = RoutedMASMethod(
        _FakeWrapper(), latent_steps=1, judger_max_new_tokens=4,
        num_workers=3, args=_Args("orchestrated"),
    )
    res = method.run_item({
        "question": "What nationality was the director of Inception?",
        "context_docs": [f"Document {i} body text." for i in range(6)],
        "gold": "paris",
    })
    assert res["n_workers"] == 3
    assert res["routing"] == "orchestrated"
    assert res["arm"] == "normal"
    assert res["prediction"].lower() == "paris" and res["correct"] is True   # stubbed decode
    assert isinstance(res["f1"], float)
    assert res["briefs"] is not None and len(res["briefs"]) == 3     # lead's briefs parsed + kept
    assert isinstance(res["worker_divergence"], float)
    assert len(res["agents"]) == 4                                   # 3 workers + judger
    # token accounting is populated
    assert res["prompt_tokens"] > 0 and res["latent_tokens"] == 3    # 1 latent step x 3 workers


def test_latent_steps_zero_is_rejected():
    # The mechanism can't be silently off: the ctor must refuse latent_steps<=0.
    import pytest
    with pytest.raises(ValueError):
        RoutedMASMethod(_FakeWrapper(), latent_steps=0, num_workers=2, args=_Args("orchestrated"))


def test_empty_cache_arm_drops_workers():
    method = RoutedMASMethod(
        _FakeWrapper(), latent_steps=1, judger_max_new_tokens=4,
        num_workers=3, args=_Args("orchestrated", arm="empty_cache"),
    )
    res = method.run_item({"question": "q", "context_docs": ["a", "b", "c"], "gold": "z"})
    assert res["arm"] == "empty_cache"
    assert len(res["agents"]) == 1                                  # judger only, no workers
    assert res["latent_tokens"] == 0                                # workers dropped
    assert res["briefs"] is None                                    # orchestrator skipped
    assert res["worker_divergence"] is None


def test_noise_blocks_arm_runs():
    method = RoutedMASMethod(
        _FakeWrapper(), latent_steps=1, judger_max_new_tokens=4,
        num_workers=3, args=_Args("orchestrated", arm="noise_blocks"),
    )
    res = method.run_item({"question": "q", "context_docs": ["a", "b", "c"], "gold": "z"})
    assert res["arm"] == "noise_blocks"
    assert len(res["agents"]) == 4                                  # workers still run, suffixes noised


def test_static_routing_has_no_briefs():
    method = RoutedMASMethod(_FakeWrapper(), latent_steps=1, num_workers=2, args=_Args("static"))
    res = method.run_item({"question": "q", "context_docs": ["a", "b"], "gold": "z"})
    assert res["routing"] == "static"
    assert res["briefs"] is None                                    # static -> no lead decode


def test_divergence_none_for_single_worker():
    method = RoutedMASMethod(_FakeWrapper(), latent_steps=1, num_workers=1, args=_Args("static"))
    res = method.run_item({"question": "q", "context_docs": ["a", "b"], "gold": "z"})
    assert res["worker_divergence"] is None                         # needs >=2 workers
