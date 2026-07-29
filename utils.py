import os
import random
import re
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def auto_device(device: Optional[str] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# this is to extract answer in \boxed{}
def extract_gsm8k_answer(text: str) -> Optional[str]:
    boxes = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxes:
        content = boxes[-1]
        number = re.search(r"[-+]?\d+(?:\.\d+)?", content)
        return number.group(0) if number else content.strip()

    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1]
    return None


def extract_gold(text: str) -> Optional[str]:
    match = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", text)
    return match.group(1) if match else None


def normalize_answer(ans: Optional[str]) -> Optional[str]:
    if ans is None:
        return None
    return ans.strip().lower()


import string as _string

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_BOXED_RE = re.compile(r"\\boxed\{(.+?)\}", re.DOTALL)
_ANSWER_MARK_RE = re.compile(
    r"(?i)(?:the\s+)?(?:final\s+)?answer(?:\s+is|:)\s*\**\s*(.+?)\s*\**\s*(?:[.\n]|$)"
)
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")


def squad_norm(s: Optional[str]) -> str:
    """SQuAD/HotpotQA normalization: lowercase, drop punctuation + articles, collapse ws."""
    s = (s or "").lower()
    s = s.translate(str.maketrans("", "", _string.punctuation))
    s = _ARTICLES_RE.sub(" ", s)
    return " ".join(s.split())


def extract_answer(text: str) -> str:
    """Pull a short answer out of a verbose judge response.

    The judge writes a <think> block then prose that LEADS with the answer
    ("Yes, both American." / "The director is X, based in Y."). Naive number-
    grabbing scored these wrong, so: strip the CoT, then take a \\boxed{} value,
    else an explicit 'answer is/:' span, else the first sentence.
    """
    if not text:
        return ""
    text = _THINK_RE.sub("", text)
    if "</think>" in text:                       # unclosed/truncated CoT
        text = text.split("</think>")[-1]
    text = text.strip()
    m = _BOXED_RE.search(text)
    if m:
        return m.group(1).strip()
    marks = list(_ANSWER_MARK_RE.finditer(text))
    if marks:
        return marks[-1].group(1).strip().strip('"*')
    for line in text.splitlines():
        line = line.strip(" -*#")
        if line:
            return re.split(r"(?<=[.!?])\s", line)[0].strip()   # first sentence
    return text.strip()


def answer_hit(response_or_pred: str, gold: str) -> bool:
    """Free-form (HotpotQA-style) recall: does the gold answer appear, as whole
    word(s), in the model's answer? Padded so 'no' does not match 'nobody'."""
    g = squad_norm(gold)
    a = squad_norm(response_or_pred)
    if not g:
        return False
    return f" {g} " in f" {a} " or g == a


def token_f1(pred: str, gold: str) -> float:
    """HotpotQA token-overlap F1 (same normalization for both sides)."""
    p, g = squad_norm(pred).split(), squad_norm(gold).split()
    if not p or not g:
        return float(p == g)
    common, gg = 0, list(g)
    for tok in p:
        if tok in gg:
            common += 1
            gg.remove(tok)
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def extract_markdown_python_block(text: str) -> Optional[str]:
    pattern = r"```python(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return None


# to run python
import traceback
from multiprocessing import Process, Manager
def run_with_timeout(code, timeout):
    def worker(ns, code):
        try:
            local_ns = {}
            exec(code, local_ns)
            ns['ok'] = True
            ns['error'] = None
        except Exception:
            ns['ok'] = False
            ns['error'] = traceback.format_exc()
    with Manager() as manager:
        ns = manager.dict()
        p = Process(target=worker, args=(ns, code))
        p.start()
        p.join(timeout)
        if p.is_alive():
            p.terminate()
            ns['ok'] = False
            ns['error'] = f"TimeoutError: Execution exceeded {timeout} seconds"
        return ns.get('ok', False), ns.get('error', None)

