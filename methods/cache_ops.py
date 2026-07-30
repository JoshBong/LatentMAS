"""KV-cache surgery for routed fan-out.

LatentMAS threads ONE cache through a linear agent chain. Routed fan-out instead
needs to (1) share one base cache across independent workers and (2) recombine
their contributions for the judger. Three primitives do that, all via the same
legacy round-trip (`to_legacy_cache` / `from_legacy_cache`) the repo already uses
in `LatentMASMethod._truncate_past`, so they work on a `transformers.Cache`
object or a raw legacy tuple identically.

    clone_cache(S)          deep, independent copy      (fork a base per worker)
    cache_length(S)         sequence length of the cache
    cache_suffix(S, start)  keep only positions >= start (a worker's own tokens)
    cache_concat([S1,S2..]) concatenate along sequence   (question once + each worker)

Layout: legacy cache is a tuple over layers, each layer a tuple (K, V) with
K, V shaped [batch, heads, seq, head_dim]; the sequence axis is dim -2.
"""

from __future__ import annotations

from typing import List, Sequence

import torch


def _to_legacy(cache):
    """-> tuple[layer] of (K, V). Handles a raw legacy tuple, a transformers-4
    Cache (to_legacy_cache), and a transformers-5 Cache (.layers[i].keys/.values).

    Fails CLOSED: an unrecognized cache raises rather than being passed through
    as if it were a tuple -- the old feature-detection guard silently mis-
    classified a v5 Cache (whose legacy methods were removed) as a plain tuple,
    which then blew up downstream.
    """
    if cache is None or isinstance(cache, (tuple, list)):
        return cache
    if hasattr(cache, "to_legacy_cache"):                    # transformers 4 Cache
        return cache.to_legacy_cache()
    if hasattr(cache, "layers"):                             # transformers 5 Cache
        return tuple((layer.keys, layer.values) for layer in cache.layers)
    raise TypeError(f"unrecognized cache type: {type(cache)!r}")


def _from_legacy(legacy, like):
    """Rebuild a cache of the same kind as `like` from a legacy tuple."""
    if isinstance(like, (tuple, list)):
        return legacy
    if hasattr(type(like), "from_legacy_cache"):             # transformers 4 Cache
        return type(like).from_legacy_cache(legacy)
    if hasattr(like, "layers"):                              # transformers 5 Cache
        rebuilt = type(like)()
        for i, (k, v) in enumerate(legacy):
            rebuilt.update(k, v, i)
        return rebuilt
    raise TypeError(f"cannot rebuild cache type: {type(like)!r}")


def cache_length(cache) -> int:
    legacy = _to_legacy(cache)
    if not legacy or legacy[0] is None:
        return 0
    return int(legacy[0][0].shape[-2])


def clone_cache(cache):
    """A deep, independent copy: mutating the copy never touches the original."""
    legacy = _to_legacy(cache)
    cloned = tuple(
        tuple(t.detach().clone() for t in layer)
        for layer in legacy
    )
    return _from_legacy(cloned, cache)


def cache_suffix(cache, start: int):
    """Keep only sequence positions [start:] -- i.e. a worker's own tokens after
    the shared prefix. `start` is typically the base-cache length."""
    legacy = _to_legacy(cache)
    sliced = tuple(
        tuple(t[..., start:, :].contiguous() for t in layer)
        for layer in legacy
    )
    return _from_legacy(sliced, cache)


def rope_inv_freq(model):
    """The rotary frequencies the model ACTUALLY uses, read off its buffer.

    Strictly better than reconstructing them from rope_theta: no config-key
    guessing (the key moved to config.rope_parameters in transformers 5), no
    head_dim assumption, and correct under rope_scaling -- linear/YaRN fold their
    factor into inv_freq, which rope_theta alone does not capture.

    Raises on dynamic NTK rope, whose inv_freq mutates with sequence length so a
    single-offset key rotation would be wrong; and raises if there is no rotary
    embedding at all (rather than guessing a default that silently corrupts).
    """
    for mod in model.modules():
        inv = getattr(mod, "inv_freq", None)
        if inv is not None:
            if getattr(mod, "rope_type", "default") == "dynamic":
                raise ValueError(
                    "dynamic NTK rope mutates inv_freq with sequence length; "
                    "cache key rotation is not valid for this model"
                )
            return inv.detach().float()
    raise ValueError("no rotary embedding (inv_freq) found -- is this a RoPE model?")


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def cache_reindex(cache, delta: int, inv_freq):
    """Shift a cache's KEYS forward by `delta` positions, via RoPE composition.

    A RoPE'd key at position p is R(p)·k0; because rotations compose,
    R(delta)·R(p)·k0 = R(p+delta)·k0 -- i.e. multiplying the cached (post-RoPE)
    key by the rotation for `delta` relocates it to position p+delta WITHOUT
    re-running the model. Values carry no position, so they are left untouched.

    This de-entangles a stitched cache: each worker was encoded starting at the
    same base position, so its keys carry overlapping rotary phases; shifting
    worker w by the total length of the workers before it makes the concatenated
    cache positionally identical to a single sequential read.

    `inv_freq` is the model's own rotary frequencies (rope_inv_freq(model)) --
    read off the model, not reconstructed from a config key, so it stays correct
    across transformers versions and under rope_scaling. Any per-frequency
    attention_scaling is already baked into the cached keys and cancels in the
    composition R(delta)·s·R(p)·k = s·R(p+delta)·k, so it is intentionally not
    reapplied here.
    """
    if delta == 0:
        return clone_cache(cache)
    legacy = _to_legacy(cache)
    K0 = legacy[0][0]
    inv = inv_freq.to(device=K0.device, dtype=torch.float32)    # [head_dim/2]
    ang = float(delta) * inv
    emb = torch.cat([ang, ang], dim=-1)                         # [head_dim]
    cos = emb.cos().to(K0.dtype)
    sin = emb.sin().to(K0.dtype)
    out = tuple(
        ((k * cos) + (_rotate_half(k) * sin), v)               # rotate K, leave V
        for (k, v) in legacy
    )
    return _from_legacy(out, cache)


def noise_cache(cache):
    """Replace every layer's K and V with Gaussian noise of the SAME shape and the
    SAME per-layer Frobenius norm, content destroyed.

    For the `noise_blocks` kill-switch: stitch shape/norm-matched noise into the
    judge in place of the real worker suffixes (re-indexed identically by the
    caller, so ONLY the information is removed, not the positional bookkeeping). If
    the judge scores the same on noise as on real states, the latent channel is
    carrying nothing -- publish the negative result.
    """
    legacy = _to_legacy(cache)
    out = []
    for (k, v) in legacy:
        nk = torch.randn_like(k)
        nv = torch.randn_like(v)
        nk = nk * (k.norm() / (nk.norm() + 1e-8))       # match this layer's K norm
        nv = nv * (v.norm() / (nv.norm() + 1e-8))       # match this layer's V norm
        out.append((nk, nv))
    return _from_legacy(tuple(out), cache)


def cache_concat(caches: Sequence):
    """Concatenate several caches along the sequence axis, layer by layer.

    Used to build the judger's context: [shared question] ++ [worker 1 tokens] ++
    [worker 2 tokens] ++ ... All caches must share batch/head/head_dim and layer
    count; only the sequence length differs.
    """
    if not caches:
        raise ValueError("cache_concat needs at least one cache")
    legacies = [_to_legacy(c) for c in caches]
    n_layers = len(legacies[0])
    if any(len(lg) != n_layers for lg in legacies):
        raise ValueError("caches disagree on layer count")

    out_layers: List[tuple] = []
    for li in range(n_layers):
        k = torch.cat([lg[li][0] for lg in legacies], dim=-2)
        v = torch.cat([lg[li][1] for lg in legacies], dim=-2)
        out_layers.append((k, v))
    return _from_legacy(tuple(out_layers), caches[0])
