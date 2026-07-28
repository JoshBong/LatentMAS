"""CPU sanity for the KV-cache surgery behind routed fan-out.

No GPU, no model download -- a tiny random GPT-2 built from config. These assert
the *primitives* are correct, which is the one real bug-risk in routed_mas; the
full pipeline (Qwen + latent steps) is GPU-only and unexercised here.

The load-bearing one is `test_handoff_identity`: running [A;B] in one pass must
equal running [A], cloning the cache, and running [B] from the clone. If that
breaks, a routed worker starting from a cloned base cache is not what it claims.
"""

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from methods.cache_ops import cache_concat, cache_length, cache_suffix, clone_cache


def _legacy(cache):
    return cache.to_legacy_cache() if hasattr(cache, "to_legacy_cache") else cache


def _model():
    torch.manual_seed(0)
    cfg = GPT2Config(vocab_size=64, n_positions=128, n_embd=32, n_layer=2, n_head=2)
    return GPT2LMHeadModel(cfg).eval()


@torch.no_grad()
def test_handoff_identity_split_equals_whole():
    m = _model()
    ids = torch.randint(0, 64, (1, 24))
    split = 10

    full = m(ids, use_cache=True).logits

    out_a = m(ids[:, :split], use_cache=True)
    cache = clone_cache(out_a.past_key_values)
    pos_b = torch.arange(split, ids.shape[1]).unsqueeze(0)
    attn = torch.ones(1, ids.shape[1], dtype=torch.long)
    logits_b = m(
        ids[:, split:], past_key_values=cache, use_cache=True,
        position_ids=pos_b, attention_mask=attn,
    ).logits

    assert torch.allclose(full[:, split:], logits_b, atol=1e-4), \
        (full[:, split:] - logits_b).abs().max().item()


@torch.no_grad()
def test_clone_is_independent():
    m = _model()
    out = m(torch.randint(0, 64, (1, 12)), use_cache=True)
    orig = out.past_key_values
    clone = clone_cache(orig)

    _legacy(clone)[0][0].add_(99.0)                 # scribble on the copy's layer-0 K
    assert _legacy(orig)[0][0].abs().max() < 50     # original untouched


@torch.no_grad()
def test_length_suffix_concat_reconstruct_the_whole():
    m = _model()
    ids = torch.randint(0, 64, (1, 20))
    split = 8

    full = m(ids, use_cache=True).past_key_values
    assert cache_length(full) == 20

    prefix = m(ids[:, :split], use_cache=True).past_key_values  # == full[:split] (causal)
    suffix = cache_suffix(full, split)
    assert cache_length(suffix) == 12

    recon = cache_concat([prefix, suffix])
    assert cache_length(recon) == 20
    assert torch.allclose(_legacy(recon)[0][0], _legacy(full)[0][0], atol=1e-5)
    assert torch.allclose(_legacy(recon)[1][1], _legacy(full)[1][1], atol=1e-5)
