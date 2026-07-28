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


def _is_cache_obj(cache) -> bool:
    return hasattr(cache, "to_legacy_cache") and hasattr(type(cache), "from_legacy_cache")


def _to_legacy(cache):
    """-> tuple[layer] of (K, V). Accepts a Cache object or an already-legacy tuple."""
    if _is_cache_obj(cache):
        return cache.to_legacy_cache()
    return cache


def _from_legacy(legacy, like):
    """Rebuild a cache of the same kind as `like` from a legacy tuple."""
    if _is_cache_obj(like):
        return type(like).from_legacy_cache(legacy)
    return legacy


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
