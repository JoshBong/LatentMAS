"""Score the ROUTER on its own -- no workers, no judge, no answer extraction.

HotpotQA hands you the router's answer key: `supporting_facts` lists the document
titles needed, which for a comparison question are the entities to look up. So we
can grade the lead's decomposition directly:

    count_match   did it emit the right NUMBER of units?
    precision     of the units it named, how many are real gold titles?
    recall        of the gold titles, how many did it name?

    python experiments/probe_router.py --model Qwen/Qwen3-1.7B --device cuda --n 20

Comparison-only by default (bridge questions are sequential -> should NOT fan out;
the router is told to emit a single unit for those, which this probe would score
against 2 gold titles -- run with --type bridge to inspect that separately).
"""

from __future__ import annotations

import argparse
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import load_hotpotqa                       # noqa: E402
from methods.router import route, score_units        # noqa: E402
from models import ModelWrapper                      # noqa: E402
from utils import auto_device                        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--type", default="comparison", help="'comparison' | 'bridge' | 'all'")
    ap.add_argument("--max_new_tokens", type=int, default=384)
    a = ap.parse_args()

    dev = auto_device(a.device)
    args = argparse.Namespace(device=str(dev), device2=str(dev), task="hotpotqa",
                              method="routed_mas", use_second_HF_model=False,
                              enable_prefix_caching=False)
    mw = ModelWrapper(a.model, dev, use_vllm=False, args=args)

    rows = []
    for item in load_hotpotqa():
        if len(rows) >= a.n:
            break
        if a.type != "all" and item.get("type") != a.type:
            continue
        plan = route(mw, item["question"], max_new_tokens=a.max_new_tokens)
        s = score_units(plan.units, item["supporting_titles"])
        rows.append(s)
        print(f"\n=== {item['question'][:80]}")
        print(f"  gold titles : {item['supporting_titles']}")
        print(f"  router units: {plan.units}   (n_workers={plan.n_workers})")
        print(f"  count_match={s['count_match']}  prec={s['precision']:.2f}  rec={s['recall']:.2f}")
        for sp in plan.specs:
            print(f"    - {sp.unit}: obj={sp.objective[:60]!r} bounds={sp.boundaries[:40]!r}")

    if not rows:
        print("no items matched; try --type all")
        return
    print("\n" + "=" * 60)
    print(f"ROUTER over {len(rows)} {a.type} questions ({a.model})")
    print(f"  count-match rate : {sum(r['count_match'] for r in rows) / len(rows):.2f}")
    print(f"  mean precision   : {st.mean(r['precision'] for r in rows):.2f}")
    print(f"  mean recall      : {st.mean(r['recall'] for r in rows):.2f}")
    print("If recall is high, the router names the right things -> move to the workers.\n"
          "If low, fix the router prompt; nothing downstream can work until it's right.")


if __name__ == "__main__":
    main()
