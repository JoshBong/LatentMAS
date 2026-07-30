"""Run the full arm x task x seed matrix and record EVERYTHING to a folder.

Each cell is a `python run.py ...` call. For each we save, under
<out>/<arm>__<task>__seed<s>/ :

    trials.jsonl   per-item results (prediction, correct, f1, divergence, briefs)
    console.log    full stdout/stderr of the run
    summary.json   the final one-line summary (accuracy / f1 / divergence / timing)

Resumable: a cell whose summary.json exists is skipped, so a crash mid-matrix
doesn't lose completed runs. Analyze with experiments/analyze.py.

Example (real run on a GPU box):
    python experiments/run_suite.py --model Qwen/Qwen3-4B --device cuda \
        --tasks hotpotqa gsm8k --arms latent_mas routed --seeds 42 43 44 \
        --max_samples 100 --out results/exp1

Example (plumbing smoke on a laptop):
    python experiments/run_suite.py --model Qwen/Qwen2.5-0.5B-Instruct --device cpu \
        --tasks gsm8k --arms latent_mas routed --seeds 42 --max_samples 2 \
        --max_new_tokens 24 --latent_steps 2 --out results/smoke
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# arm name -> the run.py flags that define it. Everything else is common.
ARM_FLAGS = {
    "single":           ["--method", "baseline"],   # 1 call, full context — the arm to beat
    "baseline":         ["--method", "baseline"],
    "latent_mas":       ["--method", "latent_mas", "--prompt", "hierarchical"],
    "routed":           ["--method", "routed_mas", "--routing", "orchestrated"],
    "routed_static":    ["--method", "routed_mas", "--routing", "static"],
    "routed_noreindex": ["--method", "routed_mas", "--routing", "orchestrated", "--no_reindex"],
}


def parse_summary(stdout: str):
    """The last line that is a JSON object with an 'accuracy' key."""
    found = None
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("{") and '"accuracy"' in line:
            try:
                found = json.loads(line)
            except json.JSONDecodeError:
                pass
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--device", default="cuda", help="cuda | mps | cpu")
    ap.add_argument("--tasks", nargs="+", default=["hotpotqa", "gsm8k"])
    ap.add_argument("--arms", nargs="+", default=["latent_mas", "routed"],
                    help=f"any of: {', '.join(ARM_FLAGS)}")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--max_samples", type=int, default=100)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=3)
    ap.add_argument("--latent_steps", type=int, default=10)
    ap.add_argument("--out", default="results/exp")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    bad = [a for a in args.arms if a not in ARM_FLAGS]
    if bad:
        sys.exit(f"unknown arm(s): {bad}. known: {list(ARM_FLAGS)}")

    runs = [(a, t, s) for a in args.arms for t in args.tasks for s in args.seeds]
    print(f"{len(runs)} runs -> {out}\n")

    for i, (arm, task, seed) in enumerate(runs, 1):
        # worker count in the id so `--num_workers 2` vs `3` don't collide
        nw = f"__nw{args.num_workers}" if arm.startswith("routed") else ""
        rid = f"{arm}{nw}__{task}__seed{seed}"
        rdir = out / rid
        rdir.mkdir(exist_ok=True)
        if (rdir / "summary.json").exists():
            print(f"[{i}/{len(runs)}] skip {rid} (already done)")
            continue

        (rdir / "trials.jsonl").unlink(missing_ok=True)   # fresh, no double-append
        cmd = [
            sys.executable, str(root / "run.py"),
            *ARM_FLAGS[arm],
            "--model_name", args.model, "--device", args.device,
            "--task", task, "--seed", str(seed),
            "--max_samples", str(args.max_samples),
            "--max_new_tokens", str(args.max_new_tokens),
            "--num_workers", str(args.num_workers),
            "--latent_steps", str(args.latent_steps),
            "--generate_bs", "1", "--do_not_enforce_qwen",
            "--log_file", str(rdir / "trials.jsonl"),
        ]
        print(f"[{i}/{len(runs)}] {rid} ...")
        proc = subprocess.run(cmd, cwd=root, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True)
        (rdir / "console.log").write_text(proc.stdout)

        summary = parse_summary(proc.stdout)
        if summary is not None:
            (rdir / "summary.json").write_text(json.dumps({**summary, "arm": arm}, indent=2))
            print(f"    acc={summary.get('accuracy')}  f1={summary.get('mean_f1')}  "
                  f"div={summary.get('mean_worker_divergence')}")
        else:
            (rdir / "FAILED").write_text("no summary parsed; see console.log")
            print(f"    FAILED (return {proc.returncode}) -> {rdir}/console.log")

    print(f"\ndone. analyze:  python experiments/analyze.py --out {out}")


if __name__ == "__main__":
    main()
