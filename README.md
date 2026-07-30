<a name="readme-top"></a>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo.png">
    <img alt="Routed LatentMAS" src="assets/logo.png" width=500>
  </picture>
</p>

<h3 align="center">
Latent Communication for Routed Multi-Agent Systems
</h3>
<p align="center"><i>A research fork of LatentMAS adding <b>routed</b> (orchestrator-worker, context-sectioned) latent fan-out.</i></p>

<p align="center">
    <a href="https://arxiv.org/abs/2511.20639"><img src="https://img.shields.io/badge/Base-LatentMAS%20(2511.20639)-B31B1B.svg?logo=arxiv" alt="Base paper"></a>
    <a href="./HANDOFF.md"><img src="https://img.shields.io/badge/Status-research%20fork-informational.svg" alt="Status"></a>
</p>

---

## 💡 Introduction

**LatentMAS** moves multi-agent collaboration from token space into the model's **latent space**: instead of writing textual reasoning, agents pass **latent thoughts** through their working memory (hidden states + KV cache), which is faster and cheaper than text-based multi-agent systems. This fork builds on the [Science-LatentMAS](https://github.com/Gen-Verse/LatentMAS) branch (customizable agent roles + hybrid text–latent generation).

Both existing LatentMAS collaboration modes hand every agent the **same** information:

- **Sequential** — one chain: `Plan → Critique → Refine → Solve`.
- **Hierarchical** — the same question answered from several perspectives, then aggregated. In the standard multi-agent taxonomy this is **voting / ensembling**.

**Routed LatentMAS asks the opposite question:** *what if each worker gets a different, disjoint slice of the context?* A **lead** decomposes the task, spins up one worker per slice, the workers run **in parallel on non-overlapping context**, and a **judge** synthesizes the answer over their combined latent state. In established terms this is the **orchestrator-worker** pattern ([Anthropic, *Building Effective Agents*](https://www.anthropic.com/research/building-effective-agents)) / **sectioned** parallelization — not input "routing" (classify-and-dispatch), and not voting-style "hierarchical."

> **The load-bearing idea:** *split by **CONTEXT**, not by **ROLE**.*

| | Base: Sequential | Base: "Hierarchical" | **Fork: Routed** |
|---|---|---|---|
| Info per agent | full, accumulated | full, same question | **disjoint slice** |
| Pattern | chain / workflow | voting / ensembling | **orchestrator-worker / sectioning** |
| Flow | `A → B → C → D` | `A,B,C → D` (same input) | `lead → {A,B,…} → judge` (different inputs) |
| Best for | iterative refinement | multi-perspective | **decomposable, comparison-style tasks** |

---

## 🔬 Routed Fan-Out (Experimental)

The contribution lives in **two files** — `methods/routed_mas.py` + `methods/cache_ops.py` — and works entirely at the **KV-cache level**:

1. **Lead** reads the shared context once → base cache `S0`.
2. **Fan out:** each worker gets an independent `clone_cache(S0)` plus its own targeted brief, and runs **mutually-blind** in parallel, producing a suffix KV block.
3. **Stitch:** `cache_concat([S0] + suffixes)` for the judge, with **RoPE re-indexing** (`cache_reindex` + `rope_inv_freq`) so the independently-computed blocks sit at correct absolute positions. (`--no_reindex` disables it, to A/B whether it matters — it does.)
4. **Judge** decodes the final answer over the stitched cache.

**Run it:**

```bash
# Routed fan-out on comparison HotpotQA (Qwen3-1.7B fits a single mid-size GPU)
python run.py --method routed_mas --model_name Qwen/Qwen3-1.7B --task hotpotqa \
  --num_workers 2 --routing orchestrated --latent_steps 10
```

- `--routing orchestrated` — the lead decodes a targeted brief per worker (default).
- `--routing static` — fixed even split of the context docs (no lead decode).
- `--num_workers N` — workers in the fan-out (comparison HotpotQA ⇒ 2).
- `--no_reindex` — skip RoPE re-indexing of the stitched cache (diagnostic).

### Scope & honest status

- **Target task = comparison-only, entity-partitioned HotpotQA.** Bridge / multi-hop questions deliberately **do not** fan out. Validity condition: the per-worker extractions must be **symmetrically independent** — worker B's target must not depend on a value only worker A discovers (see below).
- The standalone **router** (`methods/router.py`: decompose → units + per-worker specs) is **built and probed but not yet wired into `routed_mas`** — for now `--num_workers` and briefs come from the flag / orchestrator. Probe it alone:
  ```bash
  python experiments/probe_router.py --model Qwen/Qwen3-1.7B --device cuda --n 20 --type comparison
  ```

### The open question: cross-worker attention

Because workers run mutually-blind, worker B's tokens never attend to worker A's. For **independent** subtasks the judge recovers the join at its own layers, so the loss is benign — but for **Cross-Entity Conditional Extraction** (e.g. *"what was company Y's revenue in the year company X filed its first patent?"*) the value B needs was never materialized, and no post-hoc stitch can recover it. The full analysis (Claude ↔ Gemini research relay) lives in [`DEBATE_kv_cross_attention.md`](./DEBATE_kv_cross_attention.md).

A diagnostic for detecting these dependencies **before** routing:

```bash
python experiments/detect_asymmetric_dependencies.py --model Qwen/Qwen2.5-7B-Instruct --device cuda
```

It asks the model to classify questions as `PARALLELIZABLE` vs `DEPENDENCY DETECTED`; if reliable, the logic can be folded into the orchestrator to trigger a sequential fallback or multi-stage fan-out.

---

## 🛠️ Getting Started

```bash
conda create -n latentmas python=3.10 -y
conda activate latentmas
pip install -r requirements.txt
# optional, for the base method's fast path:
pip install vllm
```

Recommended: point your HF cache at a stable location to avoid repeated downloads:

```bash
export HF_HOME=/path/to/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
```

## 🚀 Repository Structure

```
LatentMAS/
│── run.py                 # Main entry for experiments
│── models.py              # Wrapper for HF + vLLM + latent realignment
│── methods/
│   ├── baseline.py        # Single-agent baseline
│   ├── text_mas.py        # Token-space multi-agent
│   ├── latent_mas.py      # Latent-space multi-agent (base method)
│   ├── routed_mas.py      # [fork] Routed fan-out: lead → clone-per-worker → stitch → judge
│   ├── router.py          # [fork] Standalone decompose step (units + per-worker specs)
│   └── cache_ops.py       # [fork] KV surgery: clone / suffix / concat / RoPE re-index (tf4 + tf5)
│── prompts.py             # Base prompt constructors
│── prompts_routed.py      # [fork] Orchestrator / worker / judge prompts + brief parsing
│── data.py                # Dataset loaders (incl. load_hotpotqa — distractor, comparison)
│── experiments/           # [fork] probe_router · run_suite · analyze · inspect_orchestrator
│   └── detect_asymmetric_dependencies.py   # [fork] pre-routing dependency classifier
│── tests/                 # [fork] 26 tests: cache ops, RoPE reindex, router parse/score (tf 4.46 + 5.14)
│── utils.py               # Answer parsing / timeout / free-form extract + F1 (HotpotQA)
│── HANDOFF.md             # [fork] current state — read first when resuming
│── DEBATE_kv_cross_attention.md   # [fork] Claude↔Gemini relay on cross-worker attention
│── requirements.txt
```

## 🧪 Running Experiments

```bash
# Baseline (single model)
python run.py --method baseline  --model_name Qwen/Qwen3-14B --task gsm8k --max_samples -1

# TextMAS (token-space multi-agent)
python run.py --method text_mas  --model_name Qwen/Qwen3-14B --task gsm8k --prompt sequential --max_samples -1

# LatentMAS (base latent multi-agent)
python run.py --method latent_mas --model_name Qwen/Qwen3-14B --task gsm8k --prompt sequential --latent_steps 10

# Routed LatentMAS (this fork)
python run.py --method routed_mas --model_name Qwen/Qwen3-1.7B --task hotpotqa --num_workers 2 --routing orchestrated --latent_steps 10
```

Notes: `--latent_steps ∈ [0, 80]` (tune per task); `--latent_space_realign` toggles latent→embedding alignment; `--do_not_enforce_qwen` to run non-Qwen HF models.

## 🧩 Inherited Science-LatentMAS Features

Fully compatible with the Science-LatentMAS branch this fork extends:

- **Custom agents & ordering** via `--custom_prompt_file prompts.json` (define `"agents": [...]`; the **last** agent always emits the final text).
- **Custom thinking tokens** via `--think "<think>\n"`.
- **Hybrid text–latent** via `--first_agent_text` (first agent emits text, middle agents reason in latent space, last agent answers).
- **vLLM hybrid pipeline** via `--use_vllm --use_second_HF_model` (vLLM decodes; a HF model does latent rollout). See notes below.

> vLLM does not officially support latent-embedding KV injection; the base repo patches vLLM internals for this. Use the **HF backend** to reproduce published numbers.

## 📊 Base LatentMAS Results (preserved)

The base method's headline results across 9 math/science/commonsense/code tasks:

<p align="center"><img src="assets/main_table1.png" width="900"></p>
<p align="center"><img src="assets/main_table2.png" width="900"></p>
<p align="center"><img src="assets/main_table3.png" width="900"></p>

Base LatentMAS reduces **~50–80% tokens** and **~3×–7× wall-clock** vs Text-MAS / chain-of-thought.

## 📚 Citation

If you use this work, please cite the original LatentMAS paper:

```bibtex
@article{zou2025latentmas,
  title={Latent Collaboration in Multi-Agent Systems},
  author={Zou, Jiaru and Yang, Xiyuan and Qiu, Ruizhong and Li, Gaotang and Tieu, Katherine and Lu, Pan and Shen, Ke and Tong, Hanghang and Choi, Yejin and He, Jingrui and Zou, James and Wang, Mengdi and Yang, Ling},
  journal={arXiv preprint arXiv:2511.20639},
  year={2025}
}
```

## 🤝 Acknowledgement

This repository is a research fork of **[LatentMAS](https://github.com/Gen-Verse/LatentMAS)** (Zou et al., 2025), built on its **Science-LatentMAS** branch, and partially based on the amazing work of **[vLLM](https://github.com/vllm-project/vllm)**. The **routed fan-out** method, cache-surgery ops, router, and cross-worker-attention analysis are additions of this fork.
