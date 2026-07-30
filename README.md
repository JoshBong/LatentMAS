<a name="readme-top"></a>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo.png">
    <img alt="Routed LatentMAS" src="assets/logo.png" width=500>
  </picture>
</p>

# ⚡️ Routed LatentMAS (Fork)

> **Fork of [LatentMAS](https://github.com/Gen-Verse/LatentMAS)** (built on the Science-LatentMAS branch). Adds a **routed** latent multi-agent method — a lead decomposes a task, workers reason **in parallel over disjoint context**, and a judge synthesizes over their combined latent state. Research fork; see [`HANDOFF.md`](./HANDOFF.md).

<p align="center">
    <a href="https://arxiv.org/abs/2511.20639"><img src="https://img.shields.io/badge/Base-LatentMAS%20(2511.20639)-B31B1B.svg?logo=arxiv" alt="Base paper"></a>
    <a href="./HANDOFF.md"><img src="https://img.shields.io/badge/Status-research%20fork-informational.svg" alt="Status"></a>
    <a href="./DEBATE_kv_cross_attention.md"><img src="https://img.shields.io/badge/Design-cross--attention%20analysis-8A2BE2.svg" alt="Design notes"></a>
</p>

---

## 🌟 Contribution: Routed Latent Communication

The original LatentMAS hands **every agent the same information**. Its two modes are:
- **Sequential** — one chain: `Plan → Critique → Refine → Solve`.
- **Hierarchical** — the same question from several perspectives, then aggregated. In the standard multi-agent taxonomy this is **voting / ensembling**.

**Routed LatentMAS asks the opposite question:** what if each worker gets a **different, disjoint slice** of the context? A **lead** decomposes the task, one worker runs per slice **in parallel on non-overlapping context**, and a **judge** synthesizes. In established terms this is the **orchestrator-worker** pattern ([Anthropic, *Building Effective Agents*](https://www.anthropic.com/research/building-effective-agents)) / **sectioned** parallelization — *not* input "routing" (classify-and-dispatch), and *not* voting-style "hierarchical."

> **The load-bearing idea:** *split by **CONTEXT**, not by **ROLE**.*

### Motivation

Across the LatentMAS ecosystem, every derivative still threads **one** shared cache down a chain — nobody fans out *different* context to *different* workers and recombines. That is the open slot this fork fills. For genuinely **decomposable** tasks (e.g. comparison questions with disjoint entities), a worker only needs its own slice, so making workers read the whole context is wasted compute and cross-talk. Routing the right slice to each worker is the natural latent analog of map-reduce.

### ⚠️ Limitations (honest)

- **Comparison-only / entity-partitioned.** Valid **iff** the per-worker extractions are *symmetrically independent* — worker B's target must not depend on a value only worker A discovers. Bridge / multi-hop questions deliberately do **not** fan out (see [`DEBATE_kv_cross_attention.md`](./DEBATE_kv_cross_attention.md)).
- **Router not yet wired in.** The standalone router (`methods/router.py`) is built and probed, but `--num_workers` and briefs currently come from the flag / orchestrator, not `route()`.
- **No verified accuracy numbers yet.** End-to-end benchmarking (routed vs. chain on comparison HotpotQA, seeded) is the current work — see Status below.
- **Cross-worker attention is lost by construction** (workers run mutually-blind). Benign for independent subtasks; fatal outside that regime. Detail in *The Math*.

### 🔬 Method: Routed Fan-Out

Everything happens at the **KV-cache level** (`methods/routed_mas.py` + `methods/cache_ops.py`):

1. **Lead** reads the shared context once → base cache `S0`.
2. **Fan out.** Each worker gets an independent `clone_cache(S0)` + its own targeted brief, runs **mutually-blind in parallel**, and produces a suffix KV block.
3. **Stitch.** `cache_concat([S0] + suffixes)` for the judge, with **RoPE re-indexing** (`cache_reindex` + `rope_inv_freq`) so the independently-computed blocks sit at correct absolute positions.
4. **Judge** decodes the final answer over the stitched cache.

### The Math

**Judge cache size.** With `N` workers, the judge decodes over

```
L_judge = |S0| + Σ_{w=1..N} |suffix_w|
```

The shared context lives in `S0` (counted once); each worker adds only its suffix.

**Position re-indexing.** Worker `w` was computed at positions `[|S0|, |S0| + |suffix_w|)` (it forked from `S0`), but in the concatenation it must sit at offset `o_w = |S0| + Σ_{j<w} |suffix_j|`. Because RoPE encodes *relative* position and its rotations compose, each key is corrected by a single closed-form rotation

```
k'  =  R_{Θ, Δ_w} · k ,      Δ_w = o_w − |S0|
```

— negligible next to a forward pass. (`--no_reindex` disables this to A/B whether it matters — it does.)

**What attention is lost.** In a joint pass over `[S0, A, B]`, causal masking already forbids `A → B`; the **only** edge dropped by parallel fan-out is `B → A`. The judge's own query tokens attend to the *entire* stitched cache, so a shallow read-and-join is preserved. The residual loss decomposes as

```
cross-attention loss  =  (i) distributional miscalibration   [O(1) fixable, training-free — APE, arXiv:2502.05431]
                       +  (ii) content / information loss     [information-theoretic — unrecoverable post-hoc]
```

Term (ii) is zero **iff** the extraction predicate is symmetrically independent — which is exactly the validity condition above.

### Usage

```bash
# Routed fan-out on comparison HotpotQA (Qwen3-1.7B fits a single mid-size GPU)
python run.py --method routed_mas --model_name Qwen/Qwen3-1.7B --task hotpotqa \
  --num_workers 2 --routing orchestrated --latent_steps 10
```

`--routing orchestrated` = lead decodes a brief per worker (default); `--routing static` = fixed doc split; `--no_reindex` = skip RoPE re-indexing (diagnostic).

---

## 🛠️ Getting Started

```bash
conda create -n latentmas python=3.10 -y
conda activate latentmas
pip install -r requirements.txt
pip install vllm   # optional, for the base method's fast path
```

### ⚙️ Setup Environment Variables

```bash
export HF_HOME=/path/to/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
```

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone -b routed-mas https://github.com/JoshBong/LatentMAS.git
cd LatentMAS
```

### 2. Repository Structure

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
│── tests/                 # [fork] 26 tests: cache ops, RoPE reindex, router (tf 4.46 + 5.14)
│── utils.py               # Answer parsing / timeout / free-form extract + F1 (HotpotQA)
│── HANDOFF.md             # [fork] current state — read first when resuming
│── DEBATE_kv_cross_attention.md   # [fork] Claude↔Gemini design relay on cross-worker attention
│── requirements.txt
```

## 🧪 Running Experiments (standard HF backend)

### 🔹 **Baseline (single model)**
```bash
python run.py --method baseline --model_name Qwen/Qwen3-14B --task gsm8k --max_samples -1
```

### 🔹 **TextMAS (text-based multi-agent system)**
```bash
python run.py --method text_mas --model_name Qwen/Qwen3-14B --task gsm8k --prompt sequential --max_samples -1
```

### 🔹 **LatentMAS (base latent multi-agent method)**
```bash
python run.py --method latent_mas --model_name Qwen/Qwen3-14B --task gsm8k --prompt sequential --latent_steps 10
```

### 🔹 **Routed LatentMAS (this fork)**
```bash
python run.py --method routed_mas --model_name Qwen/Qwen3-1.7B --task hotpotqa \
  --num_workers 2 --routing orchestrated --latent_steps 10

# Probe the router's decomposition alone (count / precision / recall vs supporting_facts)
python experiments/probe_router.py --model Qwen/Qwen3-1.7B --device cuda --n 20 --type comparison

# Classify a question as PARALLELIZABLE vs DEPENDENCY DETECTED before routing
python experiments/detect_asymmetric_dependencies.py --model Qwen/Qwen2.5-7B-Instruct --device cuda
```

#### Notes
- `--latent_steps ∈ [0, 80]` — tune per task.
- `--latent_space_realign` — toggles latent→embedding alignment (treat as a hyperparameter).
- `--do_not_enforce_qwen` — run non-Qwen HF models.

## 📊 Status / Results

Benchmarking is **in progress**. The headline experiment — routed vs. the latent chain on comparison-only HotpotQA, fixed free-form scorer, `--num_workers 2`, multiple seeds — is the current baseline. Diagnostics already in place: `worker_divergence` (are briefs actually differentiating?), a `--no_reindex` A/B on RoPE, and the dependency classifier above. Base LatentMAS reference numbers (the method this forks): ~50–80% fewer tokens and ~3×–7× wall-clock vs Text-MAS / CoT.

## ⚡ vLLM Integration (inherited)

The base method supports a hybrid HF + vLLM pipeline (`--use_vllm --use_second_HF_model`): vLLM decodes final text, a HF model handles latent rollout. The routed method runs on the **HF backend**.

> vLLM does not officially support latent-embedding KV injection; the base repo patches vLLM internals for this. Use the HF backend to reproduce published numbers.

## 🧩 Inherited Science-LatentMAS Features

Fully compatible with the branch this fork extends: custom agents & ordering (`--custom_prompt_file`), custom thinking tokens (`--think`), and hybrid text–latent generation (`--first_agent_text`).

## 🌐 Related Works based on LatentMAS

- **KNN-LatentMAS** ([Bookmaster9](https://github.com/Bookmaster9/kNN-latentMAS)) — kNN prune of the shared KV cache (selective KV).
- **Hybrid-LatentMAS** ([nhminle](https://github.com/nhminle/LatentMAS-Hybrid)) — heterogeneous *models* via closed-form cross-vocab alignment.
- **LatentMAS-SLoRA** ([Arifuzzamanjoy](https://github.com/Arifuzzamanjoy/latent_mas_slora)) — per-role LoRA adapters + domain router (routes *weights*).
- **Routed LatentMAS** (this fork) — routes *context*: parallel workers over disjoint slices → judge.

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
