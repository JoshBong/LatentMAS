#!/usr/bin/env python3
"""Text-baseline probes for routed multi-agent QA on HotpotQA comparison questions.

Consolidates three throwaway scripts (router_check / worker_check / pipeline_run)
into one program with ONE model load, ONE set of helpers, and a resumable log.

Three phases, run any subset with --phases:

  router    Score the router ALONE against HotpotQA `supporting_facts` titles.
            No workers, no synthesis. Does the lead name the right entities?

  worker    Does the routed spec change what a worker DOES? Three arms on
            identical passages:
              A  raw question only        (unrouted baseline)
              B  routed spec              (yours)
              C  routed spec, entity name MASKED
            C is the control that separates "the spec structure helps" from
            "you just told it which entity to look at."

  pipeline  End-to-end, scored on FINAL ANSWERS with HotpotQA EM + token F1:
              single  1 call, question + passages          THE REAL BASELINE
              hier    N generic workers + synthesizer      unrouted fan-out
              routed  router -> N entity workers + synth   yours

This is the TEXT baseline. It deliberately shares no code with the repo's latent
(KV-cache) pipeline -- it exists to tell you whether routing helps *at all*
before you attribute anything to the latent channel.

Everything appends to one JSONL keyed by (phase, seed, qid, arm), so a crash or
a Colab disconnect costs you one trial. Re-run the same command to continue.

    python probe.py --phases router --n-router 12
    python probe.py --phases pipeline --n-pipeline 30 --seeds 0 1 2
    python probe.py --phases router worker pipeline        # one model load

GPU note: this loads a model INTO THIS PROCESS. Never run it in the same Colab
kernel as experiments/run_suite.py, which spawns subprocesses that each need the
whole card.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import os
import re
import string
import sys
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# --------------------------------------------------------------------- scoring
# Standard SQuAD/HotpotQA normalization. Do not invent a variant -- EM and F1
# must agree with published numbers or they are not comparable to anything.

_PUNCT = str.maketrans("", "", string.punctuation)
_ARTICLES = re.compile(r"\b(a|an|the)\b")

STOPWORDS = set(
    """a an the of in on at to for and or but is are was were be been being it its
    this that these those with as by from about into over after before he she they
    his her their which who whom whose what when where why how not no do does did
    done has have had also both same more most than then there here you your i we
    """.split()
)


def normalize(s: str) -> str:
    s = (s or "").lower().translate(_PUNCT)
    return " ".join(_ARTICLES.sub(" ", s).split())


def em(pred: str, gold: str) -> float:
    return float(normalize(pred) == normalize(gold))


def f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    same = sum((collections.Counter(p) & collections.Counter(g)).values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def content_words(s: str) -> set:
    return {w for w in normalize(s).split() if w not in STOPWORDS and len(w) > 2}


def overlap(pred: str, gold: str) -> float:
    """Fraction of the gold sentence's content words present in pred."""
    g = content_words(gold)
    return len(g & content_words(pred)) / len(g) if g else 0.0


def mentions(text: str, entity: str) -> bool:
    """Does text refer to this entity? Match on its distinctive words."""
    words = [w for w in content_words(entity) if len(w) > 3] or list(content_words(entity))
    if not words:
        return False
    hit = content_words(text)
    return sum(1 for w in words if w in hit) / len(words) >= 0.5


def loose_match(a: str, b: str) -> bool:
    """Entity names vs Wikipedia titles are rarely exact."""
    na, nb = normalize(a), normalize(b)
    return bool(na and nb) and (na == nb or na in nb or nb in na)


def extract_answer(raw: str) -> str:
    """Pull the answer span out of a generation. Ordered fallbacks, never None."""
    t = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S).strip()
    for pat in (r"ANSWER\s*:\s*(.+)", r"\\boxed\{([^}]*)\}", r"answer is\s*:?\s*(.+)"):
        m = re.search(pat, t, flags=re.I)
        if m:
            return m.group(1).strip().split("\n")[0].strip(" .*\"'")
    lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
    return (lines[-1] if lines else t)[:120].strip(" .*\"'")


def extract_json(raw: str):
    """First JSON object in a generation, or None."""
    t = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S)
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.M).strip()
    m = re.search(r"\{.*\}", t, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


# ------------------------------------------------------------------------ data


def build_context(row) -> str:
    ctx = row["context"]
    return "\n\n".join(
        f"[{t}]\n" + " ".join(ss) for t, ss in zip(ctx["title"], ctx["sentences"])
    )


def sentence_lookup(row) -> dict:
    ctx = row["context"]
    return dict(zip(ctx["title"], ctx["sentences"]))


def gold_titles(row) -> list:
    """Deduped supporting_facts titles = the documents the question needs."""
    sf = row["supporting_facts"]
    titles = sf.get("title", []) if isinstance(sf, dict) else [d["title"] for d in sf]
    seen, out = set(), []
    for t in titles:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def gold_sentences(row) -> dict:
    """title -> the gold supporting sentence(s) for that title."""
    sf = row["supporting_facts"]
    pairs = (
        zip(sf["title"], sf["sent_id"])
        if isinstance(sf, dict)
        else [(d["title"], d["sent_id"]) for d in sf]
    )
    lookup, out = sentence_lookup(row), {}
    for t, sid in pairs:
        ss = lookup.get(t, [])
        if 0 <= sid < len(ss):
            out.setdefault(t, []).append(ss[sid])
    return {t: " ".join(v) for t, v in out.items()}


def load_rows(n: int, split: str = "validation"):
    """HotpotQA COMPARISON questions only -- bridge questions are sequential by
    construction, so a parallel fan-out cannot do them and including them would
    stack the deck against routing for the wrong reason."""
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split, streaming=True)
    return list(itertools.islice((r for r in ds if r.get("type") == "comparison"), n))


# ----------------------------------------------------------------------- model


class LM:
    """One model, one chat entry point, one token counter."""

    def __init__(self, name: str, device: str = "cuda"):
        print(f"loading {name} ...", flush=True)
        self.tok = AutoTokenizer.from_pretrained(name)
        # bf16 only where it's native (Ampere+). On a T4 (Turing) bf16 is emulated
        # -> slower and no memory win, so fall back to fp16 there.
        if device == "cuda":
            kw = {"dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16}
        else:
            kw = {"dtype": torch.float32}
        self.model = AutoModelForCausalLM.from_pretrained(name, **kw).to(device).eval()
        self.device = device
        self.tokens = 0

    def _render(self, msgs):
        try:  # Qwen3 honours enable_thinking; older templates ignore the kwarg
            return self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            return self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )

    @torch.no_grad()
    def chat(self, msgs, max_new: int, greedy: bool = False,
             temperature: float = 0.6, top_p: float = 0.95) -> str:
        ids = self.tok(self._render(msgs), return_tensors="pt").to(self.device)
        kw = dict(max_new_tokens=max_new, pad_token_id=self.tok.eos_token_id)
        kw.update({"do_sample": False} if greedy
                  else {"do_sample": True, "temperature": temperature, "top_p": top_p})
        out = self.model.generate(**ids, **kw)
        n_in = ids["input_ids"].shape[1]
        self.tokens += n_in + (out.shape[1] - n_in)
        gen = self.tok.decode(out[0][n_in:], skip_special_tokens=True)
        return re.sub(r"<think>.*?</think>", "", gen, flags=re.S).strip()


# ---------------------------------------------------------------------- router

ROUTER_SYS_FULL = """You are the lead agent in a multi-agent research system.
Given a question, break it into independent lookup tasks that can be done IN \
PARALLEL. A task is only valid if a worker could complete it WITHOUT seeing any \
other worker's results.

Split by WHAT INFORMATION each worker needs, not by what job it does. One worker \
per distinct entity or thing that must be looked up.
The number of workers is determined by the question. Do not pad it.

For each worker give exactly five fields:
  unit           the specific entity/thing this worker looks up
  objective      what to find out about it
  output_format  how the worker should report back
  sources        where to look
  boundaries     what this worker must NOT do

Reply with ONLY a JSON object. No prose, no markdown fences."""

ROUTER_SYS_MIN = """You are the lead agent. Break the question into independent \
lookup tasks that can be done IN PARALLEL -- a task is only valid if a worker \
could finish it without seeing another worker's results.
One worker per distinct entity that must be looked up. The number of workers is \
set by the question; do not pad it.
Reply with ONLY JSON: {"property": "...", "units": ["...", "..."]}"""

EX_Q = "Which film came out first, Blade Runner or Alien?"

EX_A_FULL = json.dumps({"property": "release year", "workers": [
    {"unit": "Blade Runner",
     "objective": "Find the release year of the film Blade Runner.",
     "output_format": "RELEASE_YEAR: <year>, plus the sentence it came from.",
     "sources": "Only passages about Blade Runner.",
     "boundaries": "Do not research Alien. Do not compare the two films."},
    {"unit": "Alien",
     "objective": "Find the release year of the film Alien.",
     "output_format": "RELEASE_YEAR: <year>, plus the sentence it came from.",
     "sources": "Only passages about Alien.",
     "boundaries": "Do not research Blade Runner. Do not compare."}]})

EX_A_MIN = '{"property": "release year", "units": ["Blade Runner", "Alien"]}'


class Router:
    """Memoized so the three phases don't each pay for the same routing call."""

    def __init__(self, lm: LM, max_new: int = 400):
        self.lm, self.max_new = lm, max_new
        self._cache = {}

    def __call__(self, question: str, full: bool = False):
        key = (question, full)
        if key not in self._cache:
            sys_p, ex_a = ((ROUTER_SYS_FULL, EX_A_FULL) if full
                           else (ROUTER_SYS_MIN, EX_A_MIN))
            raw = self.lm.chat([
                {"role": "system", "content": sys_p},
                {"role": "user", "content": f"Question: {EX_Q}"},
                {"role": "assistant", "content": ex_a},
                {"role": "user", "content": f"Question: {question}"},
            ], self.max_new, greedy=True)
            plan = extract_json(raw)
            self._cache[key] = (plan, raw)
        return self._cache[key]


def plan_units(plan) -> list:
    """Units from either router schema (`units` list or `workers[].unit`)."""
    if not plan:
        return []
    if isinstance(plan.get("units"), list):
        return [str(u) for u in plan["units"] if str(u).strip()]
    return [str(w.get("unit", "")) for w in (plan.get("workers") or [])
            if str(w.get("unit", "")).strip()]


# ------------------------------------------------------------------------ arms

WORKER_SYS = ("You are a research worker. Use only the source passages. "
              "Follow your task exactly and stay inside your boundaries. Be concise.")

FINAL_SYS = ("You answer multi-hop questions. Answer with the shortest correct span "
             "-- a name, a place, a year, or yes/no. "
             "End with a line exactly: ANSWER: <answer>")

HIER_ROLES = [
    "Gather every fact from the passages that bears on the question.",
    "Gather supporting details from the passages that a first pass might miss.",
]


def arm_single(lm, router, row, context, gen_final, **_):
    raw = lm.chat([{"role": "system", "content": FINAL_SYS},
                   {"role": "user", "content":
                    f"Source passages:\n{context}\n\nQuestion: {row['question']}"}],
                  gen_final)
    return extract_answer(raw), raw, {"calls": 1, "workers": 0}


def arm_hier(lm, router, row, context, gen_final, gen_worker=180, **_):
    """Unrouted fan-out: workers differ by generic role, NOT by entity. Matched
    to `routed` on call count so the comparison isn't a compute confound."""
    outs = [lm.chat([{"role": "system", "content": WORKER_SYS},
                     {"role": "user", "content":
                      f"Source passages:\n{context}\n\nQuestion: {row['question']}"
                      f"\n\nYour task: {role}"}], gen_worker)
            for role in HIER_ROLES]
    notes = "\n\n".join(f"Researcher {i+1}:\n{o}" for i, o in enumerate(outs))
    raw = lm.chat([{"role": "system", "content": FINAL_SYS},
                   {"role": "user", "content":
                    f"Researcher notes:\n{notes}\n\nQuestion: {row['question']}"}],
                  gen_final)
    return extract_answer(raw), raw, {"calls": len(outs) + 1, "workers": len(outs)}


def arm_routed(lm, router, row, context, gen_final, gen_worker=180, **_):
    plan, router_raw = router(row["question"])
    units = plan_units(plan)
    prop = (plan or {}).get("property") or "the relevant property"
    if not units:  # router failed -> fall back to single, and SAY SO
        pred, raw, meta = arm_single(lm, router, row, context, gen_final)
        meta.update({"router_failed": True, "router_raw": router_raw[:400]})
        return pred, raw, meta
    outs = []
    for u in units:
        others = [x for x in units if x != u]
        outs.append((u, lm.chat([
            {"role": "system", "content": WORKER_SYS},
            {"role": "user", "content":
             f"Source passages:\n{context}\n\nYOUR TASK: find the {prop} of {u}.\n"
             f"Only use passages about {u}. "
             f"Do not research {', '.join(others) if others else 'anything else'}."}],
            gen_worker)))
    notes = "\n\n".join(f"[{u}]\n{o}" for u, o in outs)
    raw = lm.chat([{"role": "system", "content": FINAL_SYS},
                   {"role": "user", "content":
                    f"Worker findings:\n{notes}\n\nQuestion: {row['question']}"}],
                  gen_final)
    return extract_answer(raw), raw, {"calls": len(units) + 2, "workers": len(units),
                                      "units": units, "property": prop,
                                      "router_failed": False}


ARMS = {"single": arm_single, "hier": arm_hier, "routed": arm_routed}


# ------------------------------------------------------------------- trial log


class TrialLog:
    """Append-only JSONL keyed by (phase, seed, qid, arm). Resume for free."""

    def __init__(self, path: str):
        self.path, self.done = path, set()
        if os.path.exists(path):
            with open(path) as fh:
                for line in fh:
                    try:
                        r = json.loads(line)
                        self.done.add((r["phase"], r.get("seed", 0), r["qid"],
                                       r.get("arm", "-")))
                    except Exception:
                        pass
            print(f"  resuming -- {len(self.done)} trials already logged")
        self.fh = open(path, "a")

    def has(self, phase, seed, qid, arm="-"):
        return (phase, seed, qid, arm) in self.done

    def write(self, rec):
        self.fh.write(json.dumps(rec) + "\n")
        self.fh.flush()
        os.fsync(self.fh.fileno())

    def close(self):
        self.fh.close()

    def read(self, phase=None):
        with open(self.path) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if phase is None or r.get("phase") == phase:
                    yield r


def mean(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def mean_sd(xs):
    if not xs:
        return float("nan"), 0.0
    m = sum(xs) / len(xs)
    sd = (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5 if len(xs) > 1 else 0.0
    return m, sd


# ---------------------------------------------------------------- phase: router


def phase_router(lm, router, rows, log, args):
    print(f"\n{'='*78}\nPHASE router -- {len(rows)} questions\n{'='*78}")
    for qid, row in enumerate(rows):
        if log.has("router", 0, qid):
            continue
        gold = gold_titles(row)
        plan, raw = router(row["question"], full=True)
        units = plan_units(plan)
        hit_g = [any(loose_match(u, g) for u in units) for g in gold]
        hit_u = [any(loose_match(u, g) for g in gold) for u in units]
        rec = {"phase": "router", "seed": 0, "qid": qid, "arm": "-",
               "question": row["question"], "gold_answer": row["answer"],
               "gold_titles": gold, "units": units,
               "parse_failed": plan is None,
               "recall": (sum(hit_g) / len(gold)) if gold else 0.0,
               "precision": (sum(hit_u) / len(units)) if units else 0.0,
               "count_match": len(units) == len(gold),
               "raw": raw[:1500]}
        log.write(rec)
        flag = "PARSE-FAIL" if plan is None else f"p={rec['precision']:.2f} r={rec['recall']:.2f}"
        print(f"  q{qid:02d} {flag:24s} gold={gold} units={units}")


def report_router(log):
    rs = list(log.read("router"))
    if not rs:
        return
    n = len(rs)
    fails = sum(r["parse_failed"] for r in rs)
    prec, rec = mean([r["precision"] for r in rs]), mean([r["recall"] for r in rs])
    cnt = mean([float(r["count_match"]) for r in rs])
    print(f"\n{'='*78}\nROUTER SCORE\n{'='*78}")
    print(f"  questions            {n}")
    print(f"  JSON parse failures  {fails}  ({fails/n*100:.0f}%)")
    print(f"  worker-count correct {cnt*100:.0f}%")
    print(f"  unit precision       {prec:.2f}")
    print(f"  unit recall          {rec:.2f}")
    if fails / n > 0.3:
        print("  >>> Model can't hold the output format. Add a second few-shot "
              "example or go up to 4B/8B before reading anything else.")
    elif rec < 0.7:
        print("  >>> ROUTER IS THE BOTTLENECK. Wrong entities means no worker can "
              "succeed. Fix here; ignore downstream.")
    elif cnt < 0.7:
        print("  >>> Entities right, count wrong -- padding or merging. Tighten "
              "the 'do not pad' instruction.")
    else:
        print("  >>> ROUTER IS GOOD. Move on to the workers.")


# ---------------------------------------------------------------- phase: worker


def mask_entity(spec: dict, unit: str) -> dict:
    pat = re.compile(re.escape(unit), re.I)
    return {k: pat.sub("THE ASSIGNED SUBJECT", str(v)) for k, v in spec.items()}


def _worker_from_spec(lm, spec, context, gen):
    return lm.chat([{"role": "system", "content": WORKER_SYS},
                    {"role": "user", "content":
                     f"Source passages:\n{context}\n\nYOUR TASK\n"
                     f"  objective:   {spec.get('objective')}\n"
                     f"  report as:   {spec.get('output_format')}\n"
                     f"  sources:     {spec.get('sources')}\n"
                     f"  boundaries:  {spec.get('boundaries')}\n"}], gen)


def phase_worker(lm, router, rows, log, args):
    print(f"\n{'='*78}\nPHASE worker -- {len(rows)} questions, arms A/B/C\n{'='*78}")
    for qid, row in enumerate(rows):
        context, golds = build_context(row), gold_sentences(row)
        plan, _ = router(row["question"], full=True)
        workers = (plan or {}).get("workers") or []
        if not workers:
            print(f"  q{qid:02d} router produced no worker specs, skipping")
            continue
        units = [str(w.get("unit", "")) for w in workers]
        out_a = None  # arm A is per-question, shared across workers
        for wi, spec in enumerate(workers):
            unit = str(spec.get("unit", ""))
            arm_key = f"w{wi}"
            if log.has("worker", 0, qid, arm_key):
                continue
            others = [u for u in units if u != unit]
            gt = next((t for t in golds if content_words(t) & content_words(unit)), None)
            gold_s = golds.get(gt, "") if gt else ""
            if out_a is None:
                out_a = lm.chat([{"role": "system", "content": WORKER_SYS},
                                 {"role": "user", "content":
                                  f"Source passages:\n{context}\n\n"
                                  f"Question: {row['question']}\n\n"
                                  f"Find the information needed to answer this."}],
                                args.gen_worker)
            out_b = _worker_from_spec(lm, spec, context, args.gen_worker)
            out_c = _worker_from_spec(lm, mask_entity(spec, unit), context, args.gen_worker)
            rec = {"phase": "worker", "seed": 0, "qid": qid, "arm": arm_key,
                   "question": row["question"], "unit": unit, "others": others,
                   "gold_sentence": gold_s,
                   "outputs": {"A": out_a, "B": out_b, "C": out_c},
                   "gold_recall": {k: (overlap(v, gold_s) if gold_s else float("nan"))
                                   for k, v in (("A", out_a), ("B", out_b), ("C", out_c))},
                   "crosstalk": {k: float(any(mentions(v, o) for o in others))
                                 for k, v in (("A", out_a), ("B", out_b), ("C", out_c))}}
            log.write(rec)
            gr = rec["gold_recall"]
            print(f"  q{qid:02d} {arm_key} [{unit[:28]:28s}] "
                  f"gold_recall A={gr['A']:.2f} B={gr['B']:.2f} C={gr['C']:.2f}")


def report_worker(log):
    rs = list(log.read("worker"))
    if not rs:
        return
    print(f"\n{'='*78}\nWORKER ARMS\n{'='*78}")
    print(f"\n  {'arm':22s} {'gold recall':>12s} {'crosstalk':>11s}")
    print("  " + "-" * 47)
    labels = {"A": "A  raw question", "B": "B  routed spec", "C": "C  masked spec"}
    for k in "ABC":
        print(f"  {labels[k]:22s} {mean([r['gold_recall'][k] for r in rs]):12.3f} "
              f"{mean([r['crosstalk'][k] for r in rs]):11.2f}")
    print("""
  HOW TO READ THIS
    B > A on gold recall        routing helps the worker focus.
    B ~ A                       routing does nothing at the worker level, and a
                                perfect router score doesn't matter.
    B crosstalk << A crosstalk  boundaries obeyed -- context isolation, measured.
                                (arm A was never told to isolate, so high
                                crosstalk there is EXPECTED, not a failure.)
    B ~ C                       the spec STRUCTURE is doing the work.
    B >> C                      it was just the entity name; the four extra
                                fields are decoration. Worth knowing before you
                                write about them.""")


# -------------------------------------------------------------- phase: pipeline


def phase_pipeline(lm, router, rows, log, args):
    total = len(rows) * len(args.arms) * len(args.seeds)
    print(f"\n{'='*78}\nPHASE pipeline -- {len(rows)} q x {len(args.arms)} arms "
          f"x {len(args.seeds)} seeds = {total} runs\n{'='*78}")
    for seed in args.seeds:
        for qid, row in enumerate(rows):
            context, gold = build_context(row), row["answer"]
            for arm in args.arms:
                if log.has("pipeline", seed, qid, arm):
                    continue
                torch.manual_seed(seed * 100_000 + qid * 10 + args.arms.index(arm))
                lm.tokens = 0
                t0 = time.time()
                try:
                    pred, raw, meta = ARMS[arm](lm, router, row, context,
                                                args.gen_final,
                                                gen_worker=args.gen_worker)
                    err = None
                except Exception as exc:  # one bad item must not kill the run
                    pred, raw, meta, err = "", "", {}, f"{type(exc).__name__}: {exc}"
                rec = {"phase": "pipeline", "seed": seed, "qid": qid, "arm": arm,
                       "question": row["question"], "gold": gold, "pred": pred,
                       "raw_prediction": raw,
                       "em": em(pred, gold) if pred else 0.0,
                       "f1": f1(pred, gold) if pred else 0.0,
                       "tokens": lm.tokens, "wall_s": round(time.time() - t0, 2),
                       "error": err, **meta}
                log.write(rec)
                print(f"  s{seed} q{qid:02d} {arm:7s} em={rec['em']:.0f} "
                      f"f1={rec['f1']:.2f} tok={lm.tokens:5d} "
                      f"| pred={pred[:38]!r} gold={gold[:26]!r}")


def report_pipeline(log, arms):
    rs = [r for r in log.read("pipeline")]
    if not rs:
        return
    by = collections.defaultdict(list)
    for r in rs:
        by[(r["arm"], r["seed"])].append(r)
    print(f"\n{'='*78}\nPIPELINE RESULTS\n{'='*78}")
    print(f"\n  {'arm':8s} {'EM':>16s} {'F1':>16s} {'tokens':>9s} {'calls':>6s} {'n':>5s}")
    print("  " + "-" * 68)
    summary = {}
    for arm in arms:
        seeds = [v for k, v in sorted(by.items()) if k[0] == arm and v]
        if not seeds:
            continue
        me, se = mean_sd([mean([t["em"] for t in v]) for v in seeds])
        mf, sf = mean_sd([mean([t["f1"] for t in v]) for v in seeds])
        allt = [r for r in rs if r["arm"] == arm]
        tk = mean([r["tokens"] for r in allt])
        cl = mean([r.get("calls", 1) for r in allt])
        summary[arm] = (me, se, tk)
        print(f"  {arm:8s} {me:.3f} ± {se:.3f}   {mf:.3f} ± {sf:.3f}  "
              f"{tk:9.0f} {cl:6.1f} {len(allt):5d}")

    routed = [r for r in rs if r["arm"] == "routed"]
    rf = [r for r in routed if r.get("router_failed")]
    if rf:
        print(f"\n  router JSON failures: {len(rf)}/{len(routed)} "
              f"({len(rf)/len(routed)*100:.0f}%) -- these fell back to single-agent, "
              f"so they dilute the routed arm toward the baseline")

    if "single" in summary and "routed" in summary:
        s_em, s_sd, s_tk = summary["single"]
        r_em, r_sd, r_tk = summary["routed"]
        gap, noise = r_em - s_em, max(s_sd, r_sd)
        print(f"\n{'='*78}\nVERDICT\n{'='*78}")
        print(f"\n  routed - single  =  {gap:+.3f} EM")
        print(f"  seed spread      =  ±{noise:.3f}")
        print(f"  token cost       =  {r_tk/max(s_tk,1):.1f}x\n")
        if abs(gap) < 2 * noise:
            print("  NO DEFENSIBLE DIFFERENCE -- the gap is inside seed noise.\n"
                  "  Most likely the task is too easy to need decomposition: one\n"
                  "  agent holds two lookups and a comparison fine. Find a task\n"
                  "  where a single agent genuinely runs out of context.")
        elif gap > 0:
            print(f"  ROUTING WINS by {gap:.3f} EM, outside noise, at "
                  f"{r_tk/max(s_tk,1):.1f}x the tokens.\n"
                  "  Real result. Next: is it worth the extra calls, and does it\n"
                  "  hold at 200 questions and on a second model?")
        else:
            print(f"  ROUTING LOSES by {abs(gap):.3f} EM at "
                  f"{r_tk/max(s_tk,1):.1f}x the cost.\n"
                  "  Decomposition is hurting -- likely information loss at the\n"
                  "  worker->synthesizer handoff. Read raw_prediction for routed\n"
                  "  items where single got it right.")


# ------------------------------------------------------------------------- main


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--phases", nargs="+", default=["router", "worker", "pipeline"],
                    choices=["router", "worker", "pipeline"])
    ap.add_argument("--arms", nargs="+", default=["single", "hier", "routed"],
                    choices=list(ARMS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n-router", type=int, default=12)
    ap.add_argument("--n-worker", type=int, default=4)
    ap.add_argument("--n-pipeline", type=int, default=30)
    ap.add_argument("--gen-worker", type=int, default=180)
    ap.add_argument("--gen-final", type=int, default=120)
    ap.add_argument("--gen-router", type=int, default=400)
    ap.add_argument("--out", default="probe_trials.jsonl")
    ap.add_argument("--report-only", action="store_true",
                    help="re-print reports from the existing JSONL, no GPU needed")
    args = ap.parse_args(argv)

    log = TrialLog(args.out)
    if args.report_only:
        if "router" in args.phases:
            report_router(log)
        if "worker" in args.phases:
            report_worker(log)
        if "pipeline" in args.phases:
            report_pipeline(log, args.arms)
        log.close()
        return 0

    n_rows = max(args.n_router if "router" in args.phases else 0,
                 args.n_worker if "worker" in args.phases else 0,
                 args.n_pipeline if "pipeline" in args.phases else 0)
    print(f"loading {n_rows} HotpotQA comparison questions ...")
    rows = load_rows(n_rows)
    print(f"  got {len(rows)} (comparison only -- bridge questions are sequential, "
          f"a parallel fan-out cannot do them)")

    lm = LM(args.model, args.device)
    router = Router(lm, args.gen_router)

    t0 = time.time()
    if "router" in args.phases:
        phase_router(lm, router, rows[:args.n_router], log, args)
        report_router(log)
    if "worker" in args.phases:
        phase_worker(lm, router, rows[:args.n_worker], log, args)
        report_worker(log)
    if "pipeline" in args.phases:
        phase_pipeline(lm, router, rows[:args.n_pipeline], log, args)
        report_pipeline(log, args.arms)
    print(f"\ndone in {(time.time()-t0)/60:.1f} min -> {args.out}")
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
