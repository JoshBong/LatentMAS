"""read_rope_theta + proof that a wrong theta corrupts the reindexed cache.

The bug this guards: transformers 5 moved rope_theta from `config.rope_theta` to
`config.rope_parameters['rope_theta']`, so a `getattr(cfg, "rope_theta", 10000.0)`
silently returned the 10000 default for every model. On a Qwen (theta 1e6) that
rotates every key by a 100x-wrong angle -> the cache becomes noise.

test_wrong_theta_corrupts_the_keys is the one that would have caught it: the old
reindex test used Llama (theta 1e4), where the wrong default happened to be right.
"""

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from methods.cache_ops import _to_legacy, cache_reindex, read_rope_theta


# ---- reader: both config layouts + fail-loud ----

class _CfgAttr:            # transformers 4 style
    rope_theta = 12345.0

class _CfgParams:          # transformers 5 style
    rope_parameters = {"rope_theta": 1_000_000, "rope_type": "default"}

class _CfgNone:
    pass


def test_reads_transformers4_attr():
    assert read_rope_theta(_CfgAttr()) == 12345.0


def test_reads_transformers5_rope_parameters():
    assert read_rope_theta(_CfgParams()) == 1_000_000.0


def test_missing_raises_not_defaults():
    import pytest
    with pytest.raises(ValueError):
        read_rope_theta(_CfgNone())
    with pytest.raises(ValueError):
        read_rope_theta(None)


# ---- the corruption proof ----

def _maxkeyerr(a, b):
    la, lb = _to_legacy(a), _to_legacy(b)
    return max((la[i][0] - lb[i][0]).abs().max().item() for i in range(len(la)))


@torch.no_grad()
def test_wrong_theta_corrupts_the_keys():
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=2,
                      num_key_value_heads=2, rope_theta=1_000_000)
    assert read_rope_theta(cfg) == 1_000_000.0     # reader gets the real value
    m = LlamaForCausalLM(cfg).eval()

    ids = torch.randint(0, 64, (1, 6))
    p, d = 3, 5
    # no prefix -> hidden states identical, keys differ by exactly R(d)
    cache_p = m(input_ids=ids, position_ids=torch.arange(p, p + 6).unsqueeze(0),
                use_cache=True).past_key_values
    ground_truth = m(input_ids=ids, position_ids=torch.arange(p + d, p + d + 6).unsqueeze(0),
                     use_cache=True).past_key_values
    key_mag = _to_legacy(ground_truth)[0][0].abs().max().item()

    correct = cache_reindex(cache_p, d, rope_theta=1_000_000)   # matches the model
    wrong = cache_reindex(cache_p, d, rope_theta=10_000)        # the old silent default

    assert _maxkeyerr(correct, ground_truth) < 1e-3            # exact
    assert _maxkeyerr(wrong, ground_truth) > 0.5 * key_mag     # error ~ the signal itself
