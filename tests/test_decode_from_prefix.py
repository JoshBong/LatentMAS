"""The REAL ModelWrapper.decode_text_batch_from_prefix, exercised on CPU.

test_routed_nl_smoke.py stubs this method out, so the actual batched decode
recipe (left-padding, per-row positions, per-row EOS, prefix mask growth) would
otherwise ship untested -- the exact mistake the repo audit flagged for the
reindex path (§4.7). Load-bearing assertion: a LEFT-PADDED BATCHED greedy decode
from a shared prefix must produce, row for row, the SAME tokens as an unpadded
one-row-at-a-time greedy decode. If padding leaked into positions or attention,
the rows diverge.
"""

import types

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from models import ModelWrapper
from methods.cache_ops import expand_cache


class _Tok:
    eos_token_id = 63
    pad_token_id = 0

    def decode(self, toks, skip_special_tokens=True):
        return " ".join(str(t) for t in toks)


def _wrapper():
    torch.manual_seed(0)
    cfg = GPT2Config(vocab_size=64, n_positions=256, n_embd=32, n_layer=2, n_head=2)
    w = types.SimpleNamespace()
    w.model = GPT2LMHeadModel(cfg).eval()
    w.device = torch.device("cpu")
    w.tokenizer = _Tok()
    # bind the real implementation, unmodified
    w.decode_text_batch_from_prefix = types.MethodType(
        ModelWrapper.decode_text_batch_from_prefix, w)
    return w


def _reference_row(model, prefix_ids, row_ids, max_new):
    """Unbatched, unpadded greedy continuation: [prefix ; row] one token at a time."""
    with torch.no_grad():
        full = torch.cat([prefix_ids, row_ids], dim=-1)
        out = model(input_ids=full, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        toks = []
        for _ in range(max_new):
            if int(nxt.item()) == _Tok.eos_token_id:
                break
            toks.append(int(nxt.item()))
            plen = full.shape[-1] + len(toks) - 1
            out = model(input_ids=nxt, position_ids=torch.tensor([[plen]]),
                        past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1].argmax(-1, keepdim=True)
    return toks


def test_batched_leftpad_decode_matches_unbatched_rows():
    w = _wrapper()
    prefix_ids = torch.tensor([[5, 9, 12, 33, 7, 21]])           # shared prefix, len 6
    rows = [torch.tensor([[3, 14, 15]]),                         # different lengths
            torch.tensor([[42, 8, 30, 11, 2]])]
    max_new = 8

    with torch.no_grad():
        S0 = w.model(input_ids=prefix_ids, use_cache=True).past_key_values
    S0_batch = expand_cache(S0, 2)

    # left-pad the two rows into one batch
    width = max(r.shape[-1] for r in rows)
    ids, mask = [], []
    for r in rows:
        pad = width - r.shape[-1]
        ids.append([0] * pad + r[0].tolist())
        mask.append([0] * pad + [1] * r.shape[-1])
    texts = w.decode_text_batch_from_prefix(
        torch.tensor(ids), torch.tensor(mask), S0_batch,
        max_new_tokens=max_new, temperature=0.0,
    )

    for text, row in zip(texts, rows):
        ref = _reference_row(w.model, prefix_ids, row, max_new)
        assert text == " ".join(str(t) for t in ref), (
            f"batched left-padded decode diverged from unbatched reference: "
            f"{text!r} vs {ref!r}")


def test_no_prefix_path_and_early_eos():
    w = _wrapper()
    ids = torch.tensor([[0, 0, 3, 14], [42, 8, 30, 11]])
    mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])
    texts = w.decode_text_batch_from_prefix(
        ids, mask, None, max_new_tokens=4, temperature=0.0)
    assert len(texts) == 2
    for t in texts:
        assert isinstance(t, str)
