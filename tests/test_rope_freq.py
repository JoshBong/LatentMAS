"""rope_inv_freq + proof that the wrong frequencies corrupt the reindexed cache.

Reading the frequencies off the model (not reconstructing them from a config key
that moved in transformers 5) can't drift. test_wrong_inv_freq_corrupts_the_keys
is the check the old Llama-theta-1e4 reindex test structurally could not catch:
there the wrong default happened to equal the right value.
"""

import pytest
import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM

from methods.cache_ops import _to_legacy, cache_reindex, rope_inv_freq


def _llama(theta):
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=2,
                      num_key_value_heads=2, rope_theta=theta)
    return LlamaForCausalLM(cfg).eval()


def test_finds_inv_freq_on_a_rope_model():
    inv = rope_inv_freq(_llama(1_000_000))
    assert inv.ndim == 1 and inv.shape[0] == 8      # head_dim/2 = (32/2)/2


def test_raises_without_rope():
    class NoRope(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 4)
    with pytest.raises(ValueError):
        rope_inv_freq(NoRope())


def test_raises_on_dynamic_rope():
    class Dyn(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("inv_freq", torch.ones(4))
            self.rope_type = "dynamic"            # inv_freq mutates with seq len -> unsafe
    with pytest.raises(ValueError):
        rope_inv_freq(Dyn())


def _maxkeyerr(a, b):
    la, lb = _to_legacy(a), _to_legacy(b)
    return max((la[i][0] - lb[i][0]).abs().max().item() for i in range(len(la)))


@torch.no_grad()
def test_wrong_inv_freq_corrupts_the_keys():
    m = _llama(1_000_000)                          # a Qwen-like theta
    inv_correct = rope_inv_freq(m)
    inv_wrong = rope_inv_freq(_llama(10_000))      # the old default's frequencies

    ids = torch.randint(0, 64, (1, 6))
    p, d = 3, 5
    cache_p = m(input_ids=ids, position_ids=torch.arange(p, p + 6).unsqueeze(0),
                use_cache=True).past_key_values
    ground = m(input_ids=ids, position_ids=torch.arange(p + d, p + d + 6).unsqueeze(0),
               use_cache=True).past_key_values
    key_mag = _to_legacy(ground)[0][0].abs().max().item()

    assert _maxkeyerr(cache_reindex(cache_p, d, inv_correct), ground) < 1e-3   # exact
    assert _maxkeyerr(cache_reindex(cache_p, d, inv_wrong), ground) > 0.5 * key_mag  # noise
