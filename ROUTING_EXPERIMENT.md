# Routing experiment — findings & plan

> Standalone exploration (for fun, KV-caching model, **not** the DeltaMAS artifact).
> Question: LatentMAS's fan-out looks suboptimal for orchestrating *different* tasks
> across agents — does a real routed fan-out beat its chain? Idk if it improves;
> that's why it's a quick fork + test.
>
> Fork: `github.com/JoshBong/LatentMAS` (origin), upstream = `Gen-Verse/LatentMAS`.
> Cloned at `~/LatentMAS`. Read-only upstream reference also at
> `~/deltamas-workspace/reference/LatentMAS`.

## Finding 1 — LatentMAS doesn't route (code-grounded)

`methods/latent_mas.py::run_batch`: a single `for agent in self.agents:` loop
threading **one** `past_kv` (KV cache) through `[Planner, Critic, Refiner, Judger]`.
Each non-judger agent appends its latent thoughts to the same growing cache;
judger decodes from the accumulation.

- **No fan-out** — it's a linear refinement chain (critic critiques planner, etc.).
- **No routing** — every agent gets `context=""` + the same shared cache; the only
  per-agent difference is its role prompt.
- **"Hierarchical" is cosmetic** — `--prompt sequential|hierarchical` only swaps the
  prompt template (`build_agent_message_{sequential,hierarchical}_latent_mas`). The
  control flow is byte-identical. Confirmed in both base and Science branches.
- **None of the 9 tasks decompose** (gsm8k, aime, gpqa, arc, mbpp, humaneval, medqa).
  All single-question. This is the real build cost, not the fork.

## Finding 2 — ecosystem scan (6 derivatives)

| Repo | What it actually adds | Real parallel? | Borrow for routing |
|---|---|---|---|
| **Science-LatentMAS** (lamm-mit `flexible_agents` / Gen-Verse `Science-LatentMAS` branch) | JSON-configurable agents (`--custom_prompt_file`), arbitrary roles/count, `--first_agent_text`. **README claims parallel+combine; the code is still a threaded chain.** | ❌ (spec≠reality) | The **JSON agent-config system** — define agents without editing code. Base the fork here. |
| **KNN-LatentMAS** (Bookmaster9) | kNN prune of the shared KV cache: top-k similarity / bottom-k diversity / random. −40% KV mem, −29% latency, ~same acc. Layer-dependent compressibility. | ❌ (chain) | **Selective-KV**: route only the relevant cache slice to each worker. |
| **Hybrid-LatentMAS** (nhminle) | Heterogeneous *models* (diff Qwen ckpts) via cross-vocab align `W_cross=(WoutᵀWout+λI)⁻¹WoutᵀWin`. `--agent_models`. | ❌ (chain) | The cross-model W (= the LatentMAS `W_a` / DeltaMAS `bridge.py`, cross-model). Not routing. |
| **LatentMAS-SLoRA** (Arifuzzaman Joy) | Per-role **LoRA adapters** + **domain router** (keyword/semantic/hybrid → medical/math/code → `set_adapter()`). Critic/Refiner in hidden-state space. | ❌ (sequential swap) | **Domain-router idea** (how to decide the decomposition) + `adapter_manager` pattern. Routes *weights*, not *context* — orthogonal to us. |
| **AVP — Agent Vector Protocol** (VectorArc) | Productized handoff: `think()`/`generate()`, `context.to_bytes()/from_bytes()` **across processes**, auto mode-negotiation (same-model full KV ~390MB / cross-model hidden-state ~6KB / text fallback). +12.4pp HumanEval cross-model. Connectors HF/Ollama/vLLM + LangChain/CrewAI/AutoGen. | handoff primitive, not an orchestrator | **Serialization** solves the "separate processes can't share tensors" problem I raised. Reference for the channel API. |
| **Awareness** (everest-an) | Not really on LatentMAS — a local MCP markdown-memory daemon. | — | skip |

## Finding 3 — the open slot (our contribution)

**Real parallel routed fan-out with differentiated per-agent context is unbuilt across the whole ecosystem.** Everyone threads one cache. Even the "flexible/hierarchical" work is a configurable *chain*. The un-taken combinations:

- routing by **weights** → SLoRA (done)
- routing by **model** → Hybrid (done)
- **selective** KV → KNN (done)
- configurable **chain** → Science (done)
- **parallel fan-out, each worker gets DIFFERENT targeted context/subtask, then combine** → **nobody. ours.**

Note the distinction that keeps it ours: Science's "hierarchical" (even if it worked) is *same question, N perspectives*. Ours is *decompose the question, route different sub-work to N workers*. Different mechanism.

## The plan

**Base:** fork from the `Science-LatentMAS` branch (JSON agent config for free), add a new method — don't hand-edit the fixed chain.

**What changes**
- NEW `methods/routed_mas.py` (fork of `latent_mas.py`): replace the threaded chain with
  1. **lead** reads question → base cache `S0`;
  2. **fan-out** — clone `S0` per worker (deep-copy via `to_legacy_cache`/`from_legacy_cache`, the round-trip `_truncate_past` already uses), each worker runs `generate_latent_batch` **independently** from its copy + its **routed brief**;
  3. **judger** concatenates the workers' hidden embeddings (`embedding_record` / `torch.cat`, already exists) → decodes.
- NEW routed prompt builders (start **static** decomposition; dynamic lead-writes-briefs later).
- NEW **decomposable task** in `data.py` + `run.py` dispatch: **HotpotQA-distractor or 2WikiMultihopQA** (context ships with the question → no retriever). ~half the build.
- NEW **divergence diagnostic** logger: pairwise distance between worker contexts.
- FIRST: a `clone→run == run-from-original` sanity check (the cache-clone is the only real bug risk; analog of DeltaMAS's handoff-equals-continuous test).

**Reused unchanged:** `ModelWrapper` (all of `models.py`), judger decode, scoring/`utils.py`, `run.py` harness, batching, Science's JSON agent config.

**Arms** (same model/task/seeds): `baseline` · `latent_mas` (chain, the one to beat) · `routed_mas` (ours) · `text_mas` (English ceiling, optional).

**Substrate:** Qwen instruct (their vLLM path asserts Qwen) — Qwen2.5-7B-Instruct or Qwen3-4B. GPU required.

**How we judge — the triple rule (a win needs all three):**
1. Δacc(routed − chain) **> seed spread** on the decomposable task (3 seeds/arm).
2. Workers **actually diverged** (pairwise distance high). If ≈identical → routing never happened (boring null). Diverged but flat → real finding (routing doesn't help).
3. Effect **absent on GSM8K** (control). If routed also beats chain there, it's a confound.

**Timeline / cost** (GPU, Qwen):
- Build ~1 day (routed_mas + prompts ~½; HotpotQA + divergence logger ~½). Cache-clone is the debug risk.
- Fastest signal: 2 arms, 1 seed, ~100 HotpotQA Q on the HF `run_batch` path → ~10–30 GPU-min, ~$1, same-day directional read.
- Full judged result: 4 arms × 3 seeds × {HotpotQA, GSM8K} × ~300 Q on vLLM → ~½ day GPU, ~$10–20.
- **Directional yes/no in ~1 day; judged result in ~2–3.**

**Risks:** hinges on a task that genuinely decomposes (the build's center of gravity); static routing may not differentiate workers (diagnostic #2 catches early); it may simply not help — a fine, informative outcome under the triple rule.

## Build status — first cut DONE (branch `routed-mas`, off Science-LatentMAS)

**Shipped:**
- `methods/cache_ops.py` — `clone_cache` / `cache_length` / `cache_suffix` / `cache_concat` (KV surgery via the legacy round-trip, works on Cache obj or tuple).
- `methods/routed_mas.py` — `RoutedMASMethod`: lead → clone-per-worker fan-out → suffix+concat → judge, + `worker_divergence` diagnostic, + scoring. Processes one item at a time (bs=1) so the cache surgery is exact (no padding to reconcile) — batching is a later optimization.
- `prompts_routed.py` — lead / worker / judger prompts; contiguous `worker_doc_slice`.
- `data.py::load_hotpotqa` — distractor config; yields `context_docs` (for routed) + `question_full` (for non-routed arms so they see the same evidence).
- `run.py` — `--method routed_mas`, `--task hotpotqa`, `--num_workers`; non-routed arms auto-get `question_full`.
- `tests/` — `test_cache_ops.py` (3) + `test_routed_smoke.py` (2), **all green on CPU**.

**Verified on CPU (no GPU/download):** the KV handoff identity (split==whole), clone independence, suffix/concat reconstruction, and the full three-phase pipeline wiring incl. the concatenated-cache decode — via a tiny GPT-2 stand-in. This covers the one real bug-risk (cache surgery + phase composition).

**NOT verified (GPU-only, held):** real accuracy on a Qwen instruct model. `methods/latent_mas.py` pulls in vLLM (needs CUDA), so the CPU venv here can't run the real end-to-end — that's the GPU box.

**Run on the box:**
```bash
pip install -e . ; pip install vllm      # or the repo's requirements
# routed vs the chain, fastest signal:
python run.py --method routed_mas --task hotpotqa --model_name Qwen/Qwen2.5-7B-Instruct \
  --num_workers 3 --max_samples 100 --seed 42 --do_not_enforce_qwen
python run.py --method latent_mas --task hotpotqa --model_name Qwen/Qwen2.5-7B-Instruct \
  --prompt hierarchical --max_samples 100 --seed 42
```
Then judge by the triple rule above (Δacc>seed-spread ∧ workers diverged ∧ no effect on GSM8K). First thing to watch: `worker_divergence` in the routed output — if ~0, the briefs aren't differentiating and nothing downstream matters.

**Known limits of the first cut:** bs=1 (slow, fine for a first read); static routing (contiguous doc split — not lead-generated briefs); free-form EM scoring on HotpotQA (no F1 yet).
