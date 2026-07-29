"""Ground-truth test for the RoPE de-entanglement fix (methods.cache_ops.cache_reindex).

Uses a tiny RoPE model (Llama) on CPU. The clean invariant:

    Encode the SAME chunk with NO prefix at positions [p..) vs [p+d..). With no
    prefix, intra-chunk attention depends only on RELATIVE positions, which are
    identical for any p -- so the hidden states (and pre-RoPE keys) are identical,
    and the cached keys differ by EXACTLY R(d). Therefore:

        cache_reindex(cache_at_p, d)  ==  cache_at_(p+d)

If that holds, shifting a worker's keys really does relocate them to a new
position without re-running the model -- which is what removes the entanglement
in the stitched judge cache.
"""

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from methods.cache_ops import _to_legacy, cache_reindex, clone_cache, rope_inv_freq

THETA = 10000.0


def _legacy(c):
    return _to_legacy(c)   # shim: works on transformers 4 and 5


def _model():
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=512, rope_theta=THETA,
    )
    return LlamaForCausalLM(cfg).eval()


@torch.no_grad()
def _cache_at(model, ids, p):
    pos = torch.arange(p, p + ids.shape[1]).unsqueeze(0)
    return model(input_ids=ids, position_ids=pos, use_cache=True).past_key_values


@torch.no_grad()
def test_reindex_equals_encoding_at_shifted_position():
    m = _model()
    ids = torch.randint(0, 64, (1, 6))
    p, d = 5, 7

    inv = rope_inv_freq(m)
    cache_p = _cache_at(m, ids, p)
    cache_pd = _cache_at(m, ids, p + d)
    shifted = cache_reindex(cache_p, d, inv)

    for (ks, vs), (kb, vb) in zip(_legacy(shifted), _legacy(cache_pd)):
        assert torch.allclose(ks, kb, atol=1e-4), (ks - kb).abs().max().item()  # keys relocated
        assert torch.allclose(vs, vb, atol=1e-6)                                # values untouched


@torch.no_grad()
def test_reindex_is_a_correct_rotation():
    m = _model()
    inv = rope_inv_freq(m)
    cache = _cache_at(m, torch.randint(0, 64, (1, 5)), 0)

    # identity at delta 0
    for (k0, _), (k, _) in zip(_legacy(cache), _legacy(cache_reindex(cache, 0, inv))):
        assert torch.allclose(k0, k, atol=1e-6)

    # composition: R(a) then R(b) == R(a+b)
    ab = cache_reindex(cache_reindex(cache, 3, inv), 4, inv)
    a_plus_b = cache_reindex(cache, 7, inv)
    for (k1, _), (k2, _) in zip(_legacy(ab), _legacy(a_plus_b)):
        assert torch.allclose(k1, k2, atol=1e-4)

    # inverse: shift by d then -d is identity
    round_trip = cache_reindex(cache_reindex(cache, 9, inv), -9, inv)
    for (k0, _), (k, _) in zip(_legacy(cache), _legacy(round_trip)):
        assert torch.allclose(k0, k, atol=1e-4)
