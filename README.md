<a name="readme-top"></a>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo.png">
    <img alt="Routed LatentMAS" src="assets/logo.png" width=500>
  </picture>
</p>

# ⚡️ Routed LatentMAS (Fork)

> **Fork of [LatentMAS](https://github.com/Gen-Verse/LatentMAS).** Adds a **routed** latent multi-agent method: a lead decomposes a task, workers reason **in parallel over disjoint context**, and a judge synthesizes over their combined KV cache.

<p align="center">
    <a href="https://arxiv.org/abs/2511.20639"><img src="https://img.shields.io/badge/Base-LatentMAS%20(2511.20639)-B31B1B.svg?logo=arxiv" alt="Base paper"></a>
    <img src="https://img.shields.io/badge/Status-research%20fork-informational.svg" alt="Status">
</p>

---

## 🌟 What's New: Routed Fan-Out

The base LatentMAS gives every agent the **same** context (a sequential chain, or several perspectives on the same question). **Routed LatentMAS gives each worker a different, disjoint slice** — the **orchestrator-worker** pattern: lead decomposes → workers run in parallel on non-overlapping context → judge combines.

> **The idea:** split by **CONTEXT**, not by **ROLE**.

**How it works** (`methods/routed_mas.py` + `methods/cache_ops.py`, all at the KV-cache level):

1. **Lead** reads the shared context once → base cache `S0`.
2. **Fan out** — each worker gets an independent `clone_cache(S0)` + its own brief, runs in parallel, produces a suffix KV block.
3. **Stitch** — `cache_concat([S0] + suffixes)` with RoPE re-indexing so the blocks sit at correct positions.
4. **Judge** decodes the final answer over the stitched cache.

```bash
python run.py --method routed_mas --model_name Qwen/Qwen3-1.7B --task hotpotqa \
  --num_workers 2 --routing orchestrated --latent_steps 10
```

`--routing orchestrated` = lead decodes a brief per worker · `--routing static` = fixed doc split.

---

## 🛠️ Getting Started

```bash
conda create -n latentmas python=3.10 -y
conda activate latentmas
pip install -r requirements.txt

git clone -b routed-mas https://github.com/JoshBong/LatentMAS.git
cd LatentMAS
```

Optionally set your HF cache: `export HF_HOME=/path/to/huggingface`.

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

Inherited from the base: `--latent_steps ∈ [0,80]`, `--latent_space_realign`, custom agents via `--custom_prompt_file`, `--do_not_enforce_qwen` for non-Qwen models, and the hybrid HF+vLLM path (`--use_vllm`).

## 📦 Repository Structure

```
LatentMAS/
│── run.py                 # Main entry
│── models.py              # HF + vLLM wrapper + latent realignment
│── methods/
│   ├── baseline.py · text_mas.py · latent_mas.py   # base methods
│   ├── routed_mas.py      # [fork] lead → clone-per-worker → stitch → judge
│   ├── router.py          # [fork] decompose step (units + per-worker specs)
│   └── cache_ops.py       # [fork] KV surgery: clone / concat / RoPE re-index
│── prompts_routed.py      # [fork] orchestrator / worker / judge prompts
│── data.py                # loaders (incl. load_hotpotqa)
│── experiments/           # [fork] probe_router · run_suite · analyze · inspect_orchestrator
│── tests/                 # [fork] 26 tests (cache ops, RoPE reindex, router)
```

## 🌐 Related Works based on LatentMAS

- **KNN-LatentMAS** ([Bookmaster9](https://github.com/Bookmaster9/kNN-latentMAS)) — kNN prune of the shared KV cache.
- **Hybrid-LatentMAS** ([nhminle](https://github.com/nhminle/LatentMAS-Hybrid)) — heterogeneous models via cross-vocab alignment.
- **LatentMAS-SLoRA** ([Arifuzzamanjoy](https://github.com/Arifuzzamanjoy/latent_mas_slora)) — per-role LoRA adapters + domain router.
- **Routed LatentMAS** (this fork) — routes *context*: parallel workers over disjoint slices → judge.

## 📚 Citation

```bibtex
@article{zou2025latentmas,
  title={Latent Collaboration in Multi-Agent Systems},
  author={Zou, Jiaru and Yang, Xiyuan and Qiu, Ruizhong and Li, Gaotang and Tieu, Katherine and Lu, Pan and Shen, Ke and Tong, Hanghang and Choi, Yejin and He, Jingrui and Zou, James and Wang, Mengdi and Yang, Ling},
  journal={arXiv preprint arXiv:2511.20639},
  year={2025}
}
```

## 🤝 Acknowledgement

Research fork of **[LatentMAS](https://github.com/Gen-Verse/LatentMAS)** (Zou et al., 2025), built on its Science-LatentMAS branch, partially based on **[vLLM](https://github.com/vllm-project/vllm)**. The routed method, cache-surgery ops, and router are additions of this fork.
