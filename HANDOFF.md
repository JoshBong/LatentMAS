# HANDOFF — routed_mas (LatentMAS routing fork)

*Last updated: 2026-07-30. Read this first when resuming.*

## What this is
A standalone, for-fun exploration (separate from the DeltaMAS Fellows artifact):
**does a real routed fan-out beat LatentMAS's linear chain on decomposable tasks?**
Lead decomposes → workers run independently on disjoint context → judge synthesizes.
Repo: `github.com/JoshBong/LatentMAS`, branch **`routed-mas`**. Local: `~/LatentMAS`.

## The baton — do this next
1. **Run the router probe** (cheap, lead-decode only, no workers/judge):
   ```bash
   python experiments/probe_router.py --model Qwen/Qwen3-1.7B --device cuda --n 20 --type comparison
   ```
   Read: count-match / precision / recall of the lead's units vs `supporting_facts`.
2. **If recall is high** → wire the router into `routed_mas`: `n_workers` and the
   per-worker briefs come from `methods.router.route()` instead of the `--num_workers`
   flag + `build_orchestrator_prompt`. (Router is built + tested but NOT yet wired in.)
3. **If precision is low** (0.5B added a spurious "Nationality" unit) → tweak the
   router prompt only; nothing downstream matters until the lead decomposes right.
4. Then the real experiment: `routed` vs `latent_mas` on **comparison-only** HotpotQA,
   1.7B, both scored with the fixed extractor, `--num_workers 2`, 3 seeds.

## The core idea (why divergence was low)
Split by **CONTEXT** (disjoint info each worker needs), not by **ROLE**. The first runs
gave workers different *jobs* on the *same* info → identical briefs → divergence ~0.
Fix = partition by entity ("everything about X" / "everything about Y"). Agent count =
number of entities (comparison Q = 2). Bridge/sequential Qs should NOT fan out → the
experiment is comparison-only. The router is just the lead LLM call; its output is TEXT
(control flow: how many workers, who); the brief payload can later be latent.

## What's built (all on `routed-mas`, tests green on transformers 4.46 + 5.14)
- `methods/router.py` — decompose → units + 4-field specs; `score_units` vs gold titles; `route()`. **Standalone.**
- `methods/routed_mas.py` — lead → orchestrator briefs → clone-per-worker → cache concat + RoPE reindex → judge.
- `methods/cache_ops.py` — clone/suffix/concat/`cache_reindex`/`rope_inv_freq` (KV surgery; tf4+tf5).
- `prompts_routed.py` — orchestrator/worker/judger prompts + `parse_briefs`.
- `utils.py` — `extract_answer`/`answer_hit`/`token_f1`/`squad_norm` (free-form scoring).
- `experiments/` — `run_suite.py` (matrix), `analyze.py` (table + triple-rule), `probe_router.py`, `inspect_orchestrator.py`.
- `tests/` — 26 tests (cache ops, RoPE reindex ground-truth, router parse+score, real-judge-output extraction). Run `pytest -q`.

## Bugs conquered this session (all fail-loud now — don't re-hit them)
1. **Dataset IDs** → namespaced: `openai/gsm8k`, `hotpotqa/hotpot_qa` (no trust_remote_code).
2. **transformers 5 removed the legacy KV cache API** → shim in `cache_ops._to_legacy/_from_legacy`.
3. **`rope_theta` moved to `config.rope_parameters` in tf5** → now read `inv_freq` off the model (`rope_inv_freq`), correct under rope_scaling.
4. **Thinking budget** → `enable_thinking=False` for the orchestrator (Qwen3 spent the budget in `<think>` and never emitted briefs).
5. **OOM** → `low_cpu_mem_usage=True`; and 4B is too big for a T4 chain on long hotpot context → **use Qwen3-1.7B**.
6. **THE big one — free-form scoring**: the judge answers correctly in prose ("Yes, both American") but the old extractor grabbed a random number → HotpotQA accuracy was a *measurement artifact*, not the method. Fixed; a live item now scores `OK=True`.

## First real GPU-run finding
The model **reasons and answers correctly** (~7/7 on the sample); the 0.45 HotpotQA
number was the broken scorer. So the method works at the reasoning level — the open
question is whether the *latent* channel carries it (the actual experiment).

## Queued (not built — build after a baseline number exists)
- **`routed_latentonly`** arm: suffix from `base_len + prompt_len` (latents only) → tests the O(1)/compression claim. Currently the judge re-reads all docs (channel not compressed yet).
- **`--num_workers` 2 vs 3** sweep (run ids already namespaced so they don't collide).
- DeltaMAS side: **N+1 handoff repair pass** (converge each worker's state to the pseudoinverse before handoff) — see `ark/projects/latent-comm-research-log.md` §5.

## Running on Colab (ephemeral `/content` — re-clone every fresh runtime)
```python
%cd /content && !rm -rf /content/LatentMAS
!git clone -b routed-mas https://github.com/JoshBong/LatentMAS.git
%cd /content/LatentMAS && !pip install -q transformers datasets accelerate
```
Then the probe/suite commands. Point `--out` at `/content/drive/MyDrive/...` so results
survive restarts. First `[ModelWrapper] … cuda=True` line confirms the GPU is attached.
Compute: Colab Pro (1-month) now; Cornell Red Cloud (free) / G2 (via a lab) once enrolled.

## Sibling project
The **DeltaMAS Fellows artifact** (delta-rule recurrent state as an O(1) inter-agent
channel) is separate: `~/deltamas-workspace/` + `ark/projects/latent-comm-research-log.md`.
This routing fork is plain-transformer KV — the repair-pass / delta-state ideas belong there.
