"""Draw the answers out of a run_suite folder: comparison table + triple-rule verdict.

    python experiments/analyze.py --out results/exp1

Reads every <arm>__<task>__seed<s>/summary.json, aggregates mean +/- sd over
seeds, prints a table, and adjudicates routed vs latent_mas by the triple rule:

    routed WINS iff  (Δacc on the decomposable task > seed spread)
                AND  (workers actually diverged)
                AND  (no effect on the control task)
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

DECOMPOSABLE = "hotpotqa"
CONTROL = "gsm8k"


def load(out: Path) -> dict:
    runs = {}
    for d in sorted(out.glob("*__*__seed*")):
        f = d / "summary.json"
        if f.exists():
            runs[d.name] = json.loads(f.read_text())
    return runs


def by_arm_task(runs, metric):
    g = defaultdict(list)
    for r in runs.values():
        v = r.get(metric)
        if v is not None:
            g[(r["arm"], r["task"])].append(v)
    return g


def ms(xs):
    if not xs:
        return None, None
    return st.mean(xs), (st.pstdev(xs) if len(xs) > 1 else 0.0)


def fmt(x):
    return f"{x:.3f}" if isinstance(x, float) else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/exp")
    args = ap.parse_args()
    out = Path(args.out)

    runs = load(out)
    if not runs:
        print(f"no completed runs in {out}")
        return
    acc = by_arm_task(runs, "accuracy")
    f1 = by_arm_task(runs, "mean_f1")
    div = by_arm_task(runs, "mean_worker_divergence")
    arms = sorted({k[0] for k in acc})
    tasks = sorted({k[1] for k in acc})

    lines = [f"# Results ({len(runs)} runs, {out})", ""]
    header = f"{'task':<10} {'arm':<18} {'accuracy':<16} {'f1':<10} {'divergence':<10} {'seeds':<6}"
    lines += [header, "-" * len(header)]
    for t in tasks:
        for arm in arms:
            xs = acc.get((arm, t))
            if not xs:
                continue
            am, asd = ms(xs)
            fm, _ = ms(f1.get((arm, t), []))
            dm, _ = ms(div.get((arm, t), []))
            lines.append(f"{t:<10} {arm:<18} {am:.3f} ± {asd:.3f}    "
                         f"{fmt(fm):<10} {fmt(dm):<10} {len(xs):<6}")
    lines.append("")

    # ---- triple rule: routed vs latent_mas ----
    def cell(arm, task, g):
        return ms(g.get((arm, task), []))

    verdict = ["## Triple-rule verdict (routed vs latent_mas)"]
    r_acc, r_sd = cell("routed", DECOMPOSABLE, acc)
    l_acc, l_sd = cell("latent_mas", DECOMPOSABLE, acc)
    r_div, _ = cell("routed", DECOMPOSABLE, div)
    if r_acc is None or l_acc is None:
        verdict.append(f"  need both 'routed' and 'latent_mas' on {DECOMPOSABLE} — incomplete.")
    else:
        delta = r_acc - l_acc
        spread = max(r_sd or 0, l_sd or 0)
        c1 = delta > spread
        c2 = (r_div or 0) > 0.05                     # workers meaningfully diverged
        # control: routed should NOT beat the chain on the non-decomposable task
        rc_acc, _ = cell("routed", CONTROL, acc)
        lc_acc, _ = cell("latent_mas", CONTROL, acc)
        c3 = (rc_acc is None or lc_acc is None) or ((rc_acc - lc_acc) <= spread)
        verdict += [
            f"  {DECOMPOSABLE}:  routed {r_acc:.3f} vs chain {l_acc:.3f}  (Δ={delta:+.3f}, seed spread {spread:.3f})",
            f"  [{'PASS' if c1 else 'fail'}] Δacc > seed spread",
            f"  [{'PASS' if c2 else 'fail'}] workers diverged (routed divergence {fmt(r_div)} > 0.05)",
            f"  [{'PASS' if c3 else 'fail'}] no win on the {CONTROL} control",
            "",
            f"  VERDICT: {'ROUTING HELPS (all three hold)' if (c1 and c2 and c3) else 'not established — see which check failed'}",
        ]
    lines += verdict

    report = "\n".join(lines)
    print(report)
    (out / "report.md").write_text(report + "\n")
    print(f"\n[written to {out}/report.md]")


if __name__ == "__main__":
    main()
