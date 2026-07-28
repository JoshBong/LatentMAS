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

## Build status — DONE (branch `routed-mas`, off Science-LatentMAS)

The channel design: **route OUT with text** (orchestrator decodes a brief per worker),
**come BACK with latent state** (workers' KV/hidden state, concatenated for the judge).
This is the Anthropic orchestrator-worker graph with the *return* channel swapped
text→latent.

**Shipped:**
- `methods/cache_ops.py` — `clone_cache` / `cache_length` / `cache_suffix` / `cache_concat` (KV surgery via the legacy round-trip; Cache obj or tuple).
- `methods/routed_mas.py` — `RoutedMASMethod`, four phases: (1) lead encodes the question → S0; (1b) **orchestrator decodes one text brief per worker** (`--routing orchestrated`, the default) or falls back to a static contiguous doc split (`--routing static`); (2) each worker clones S0 + gets its brief/slice, runs independently; (3) judge decodes from `concat(S0, each worker's own tokens)`. Plus `worker_divergence`, EM + token-**F1**, and `briefs` logged. bs=1 so the cache surgery is exact.
- `prompts_routed.py` — orchestrator / lead / worker / judger prompts, `parse_briefs` (robust to model slop), `worker_doc_slice`.
- `data.py::load_hotpotqa` — distractor; yields `context_docs` (routed) + `question_full` (non-routed arms see the same evidence).
- `run.py` — `--method routed_mas`, `--task hotpotqa`, `--num_workers`, `--routing`, and **`--log_file`** (per-item JSONL: prediction/correct/f1/worker_divergence/briefs). Summary prints mean_f1 + mean_worker_divergence.
- `tests/` — cache_ops (3) + routed smoke incl. orchestrated + static (4) + parse_briefs/doc-slice (4) = **10 green on CPU**.

**Verified on CPU (no GPU/download, tiny GPT-2 stand-in):** KV handoff identity (split==whole), clone independence, suffix/concat reconstruction, the full four-phase wiring incl. the orchestrator decode→parse→workers and the concatenated-cache decode, brief parsing, F1. Covers the real bug-risk.

**NOT verified (GPU-only, held for you):** real accuracy on Qwen3-4B. `latent_mas` pulls in vLLM (CUDA), so this CPU venv can't run the real end-to-end.

**Run on the box (Qwen3-4B; keeps their defaults latent_steps=10/temp 0.6):**
```bash
pip install -r requirements.txt        # or: pip install -e . ; pip install vllm

# --- THE TEST: HotpotQA, routed vs the chain, 3 seeds each (no published baseline) ---
for s in 42 43 44; do
  python run.py --method routed_mas --task hotpotqa --model_name Qwen/Qwen3-4B \
    --num_workers 3 --routing orchestrated --max_samples 100 --seed $s \
    --do_not_enforce_qwen --log_file results/routed_hotpot_$s.jsonl
  python run.py --method latent_mas --task hotpotqa --model_name Qwen/Qwen3-4B \
    --prompt hierarchical --max_samples 100 --seed $s \
    --log_file results/chain_hotpot_$s.jsonl
done

# --- THE CONTROL: their tasks, reuse their published numbers, run only your arm ---
python run.py --method routed_mas --task gsm8k --model_name Qwen/Qwen3-4B \
  --num_workers 3 --max_samples 100 --seed 42 --do_not_enforce_qwen \
  --log_file results/routed_gsm8k_42.jsonl   # expect ~their number + divergence≈0
```

**Judge by the triple rule:** routed "wins" only if Δacc(routed−chain) on HotpotQA **> seed spread** AND **mean_worker_divergence is high** (workers actually differentiated) AND **no effect on the GSM8K control**. First number to read is `mean_worker_divergence` — if ≈0, the orchestrator isn't producing distinct briefs and nothing downstream matters. Their benchmarks don't decompose, so they can only show "no harm," never a win.

**Known limits:** bs=1 (slow, fine for a first read); static doc→worker mapping even in orchestrated mode (the lead writes sub-tasks but docs are still split contiguously — matching sub-task↔doc is a later knob); F1 is token-overlap, not the official HotpotQA supporting-fact metric.
