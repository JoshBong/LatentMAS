# Routed LatentMAS

**Research log: can a truly parallel multi-agent system divide *information*, not just *thinking*?**

Fork of [LatentMAS](https://github.com/Gen-Verse/LatentMAS) (Zou et al., [arXiv:2511.20639](https://arxiv.org/abs/2511.20639)).

LatentMAS chains agents through one shared cache. I tried to run the agents in parallel and merge their caches. The merged caches carry real information but lose to one model reading everything (50% vs. 64% exact match on HotpotQA), mostly because the orchestrator, routing by title alone, often never handed the second-hop paragraph to any worker. Sharing the *input* cache and having workers report back in text keeps the accuracy (62%) at 40% fewer prompt tokens than handing every worker the documents.

![Routed LatentMAS architecture](assets/routed-latentmas.png)

---

## Dividing the thinking vs. dividing the information

LatentMAS has agents pass their KV cache down a chain: a planner, a critic, a refiner and a judge, each waiting for the one before it, all working on the same question. That's dividing the *thinking*: different roles reasoning over shared information, one after another.

```
same question ─→ planner ─→ critic ─→ refiner ─→ judge → answer
                 (one cache, handed down the chain)
```

I wanted a truly parallel system that divides the *information* itself, map-reduce over the context:

```
documents ─→ split ─→ worker 1 reads docs A ─┐
                   ─→ worker 2 reads docs B ─┴─→ judge → answer
```

That's what real systems need when:

- **the context is too big for one agent:** thousands of pages, a whole codebase, long logs;
- **speed matters:** read N pieces in parallel instead of one long sequential read;
- **the information is private by nature:** one agent per database, tool, or user, each seeing only its own.

It's also where latent communication gets hard. When everyone sees the same input, it doesn't matter that agents never look at each other's work. When the evidence is split, worker 2 never sees what worker 1 found, and a question that chains across both can break. This log is about that gap.

## 1. Initial idea

LatentMAS lets agents skip writing to each other. A transformer builds a KV cache as it reads, which is its working memory of the context, and LatentMAS hands that cache straight to the next agent instead of a text summary. Nothing gets flattened into natural language between agents.

If the cache can be handed off, maybe it can also be **split**. A lead reads the context once, workers each take a different slice in parallel, and a judge reads all their caches together at once. It's map-reduce, with the KV cache as the channel.

Built it as `routed_mas` ([`64d2d36`](https://github.com/JoshBong/LatentMAS/commit/64d2d36)):

1. The lead reads the shared context → base cache `S0`.
2. Each worker gets a copy of `S0` plus its own sub-task, and runs without seeing the others.
3. Each worker's new tokens are sliced off and concatenated onto `S0`.
4. The judge answers over `[S0 | worker 1 | worker 2 | ...]`.

## 2. RoPE reindexing

Every worker started right after `S0`, so all their tokens claimed the same positions. Stitched together naively, the judge read garbage.

RoPE stores a key computed at position `p` as `R(p)·k`, and rotations compose (`R(Δ)·R(p) = R(p+Δ)`), so a cached key can be moved without re-running the model:

```
k' = k ⊙ cos(Δf) + rotate_half(k) ⊙ sin(Δf)        f = the model's rotary frequencies
```

Each worker shifts by the total length of the workers before it ([`64d2d36`](https://github.com/JoshBong/LatentMAS/commit/64d2d36)). Surprisingly it worked and what I originally thought was going to be the main obstacle was solved relatively easily.

## 3. Does stitching work

With positions fixed, there was still a deeper question. Each worker's cache was built blind to the others. How much does that cost, and could that be fixed without training or recomputing?

Where it landed ([full write-up](docs/why-stitching-fails.md)): the loss has two parts.

- **Calibration.** Blocks encoded in parallel give the judge a miscalibrated attention distribution. That part has a training-free fix: [APE](https://arxiv.org/abs/2502.05431) rescales attention at read time and recovers 93–98% of sequential performance on retrieval and in-context learning.
- **Content.** If worker B would only have looked something up *because of* what worker A found, B never looked it up. "What was Y's revenue in the year X filed its first patent?" A finds the year; B, running blind, never pulls that year's revenue. No transform of B's cache can produce a number that was never computed.

So in principle, stitching is only sound when **no worker's job depends on another worker's findings**. Comparisons ("which city is bigger?") qualify; chains don't. Whether that's what actually costs accuracy here is still open (see section 6): in practice, the bigger loss came one step earlier, in routing. I tried having a 3B router check this before splitting the work, and it was at chance (3/6), wrong in both directions ([`c52acf4`](https://github.com/JoshBong/LatentMAS/commit/c52acf4)).

## 4. No more stitching

If merging caches back together is the part that breaks, keep the part that works and drop the rest. `routed_nl` ([`0be4da0`](https://github.com/JoshBong/LatentMAS/commit/0be4da0)):

- An orchestrator writes one sub-task per worker.
- The documents are encoded **once** into a shared KV prefix that every worker reads from.
- All workers decode their findings in **one batched forward pass**, as short text.
- The judge reads the findings, not every document.

The orchestrator only ever sees the document *titles*, never their text. It needs them to write sub-tasks and assign sources; with no titles it has nothing to route with. So every arm that routes gets the titles, and only the `nodocs` control runs it without them.

Every worker branches off the same read-only prefix, so nothing is ever merged. No cache surgery, no re-indexing, no stitching. This isn't a new idea: it's prefix caching, the same trick vLLM and SGLang use, studied for multi-agent workflows in [KVFlow](https://arxiv.org/abs/2507.07400) and for batched decoding in [Hydragen](https://arxiv.org/abs/2402.05099). What it changes relative to LatentMAS is the direction of sharing: the cache is shared on the way *in* (the documents), and results come back as text. The claim it can make is small and checkable: same accuracy as giving every worker the documents as text, with the documents paid for once instead of once per worker. `--channel` tests it with everything else fixed: `kv` (the mechanism), `text` (the cost control), `nodocs` (the negative control: if `kv ≈ nodocs`, the shared prefix carries nothing).

## 5. Similar work

[KVShareArena](https://arxiv.org/abs/2609.10266) (Shi, Zheng, Lou) came out a week after `routed_nl`. It benchmarks KV-cache reuse across contexts, multi-agent workflows included, and finds that reusing a cache computed under different preceding text reduces answer quality, and that which repairs help depends on the model. It's a careful, large-scale version of what I ran into here, and it's the reference to read for this problem.

## 6. The results

I ran the full comparison with a scorer for every arm: HotpotQA (distractor), first 50 validation questions, seed 42, Qwen3-4B, 2 workers. Each question needs facts from two paragraphs hidden among eight distractors.

| arm                   | what it is                                                      | EM       | F1   | prompt tokens / question |
| --------------------- | --------------------------------------------------------------- | -------- | ---- | ------------------------ |
| `single`              | one model reads all 10 paragraphs                               | **0.64** | 0.73 | –                        |
| `routed`              | parallel workers, caches stitched for the judge                 | 0.50     | 0.58 | –                        |
| `routed`, text blocks | same, workers skip the latent thinking steps                    | 0.52     | 0.61 | –                        |
| `empty_cache`         | control: judge gets no worker blocks                        | 0.16     | 0.25 | –                        |
| `nl_kv` (`routed_nl`) | documents shared once through the cache, workers report in text | **0.62** | 0.69 | **2,420**                |
| `nl_text`             | same, documents pasted into every worker's prompt               | 0.62     | 0.70 | 4,022                    |
| `nl_nodocs`           | control: workers get no documents                           | 0.14     | 0.21 | –                        |

What it says:

- **Stitched caches lose to one model reading everything** (0.50 vs. 0.64). Paired by question, `single` got 8 right that `routed` missed, and `routed` got 1 that `single` missed.
- **Most of that gap is routing, not merging.** On 20 of the 50 questions, all of them chain ("bridge") questions, the orchestrator never assigned one of the two gold paragraphs to any worker. That's expected when routing by title: you can't tell which paragraph the second hop needs until the first hop is solved. On those questions `routed` got 7 vs. 12 for `single`. When both gold paragraphs did reach different workers, it was 13 vs. 15 out of 24, too few to say whether workers not seeing each other costs anything.
- **The latent thinking steps don't matter on Qwen3** (0.50 vs. 0.52). On Qwen2.5-3B they wreck the judge's output entirely (word soup), so an earlier run of this table on that model is void.
- `routed_nl` **matched text delivery on these 50 questions** (31 correct each, F1 0.69 vs. 0.70) with 40% fewer prompt tokens. That's the same observed score, not proven equivalence.
- **The negative controls remove two things at once:** the orchestrator's titles and the workers' documents, so 0.50 vs. 0.16 and 0.62 vs. 0.14 combine both effects. The empty-cache 0.16 is also all guessing on comparison questions (8 of 14 right, 0 of 36 on chains). I checked whether the sub-tasks leaked answers: 3–4 of 50 contained the gold answer, and all but one just repeated an entity already named in the question.

n=50 with one seed is directional, not a benchmark result; the gaps to the controls are large, the closer gaps aren't. Upstream `latent_mas` is left out of the table: its answers were mostly right but came as full sentences ("Yes, Scott Derrickson and Ed Wood were of the same nationality"), which strict exact match scores as wrong, so its number measures answer format, not the method.

---

## Running it

```bash
git clone https://github.com/JoshBong/LatentMAS.git && cd LatentMAS
pip install -r requirements.txt "transformers>=5" matplotlib   # results in section 6 used transformers 5

# routed fan-out (2 entities -> 2 workers)
python run.py --method routed_mas --model_name Qwen/Qwen3-4B --task hotpotqa --num_workers 2 --do_not_enforce_qwen

# shared prefix + batched workers + text findings, and its two controls
python run.py --method routed_nl --channel kv     --model_name Qwen/Qwen3-4B --task hotpotqa --do_not_enforce_qwen
python run.py --method routed_nl --channel text   --model_name Qwen/Qwen3-4B --task hotpotqa --do_not_enforce_qwen
python run.py --method routed_nl --channel nodocs --model_name Qwen/Qwen3-4B --task hotpotqa --do_not_enforce_qwen

# the comparison in section 6 (resumable, one folder per arm)
python experiments/run_suite.py --model Qwen/Qwen3-4B --tasks hotpotqa \
    --arms single empty_cache routed nl_kv nl_text nl_nodocs --seeds 42 --max_samples 50 --num_workers 2 --out results/gate
python experiments/analyze.py --out results/gate

python -m pytest -q tests   # 49 tests: cache ops, RoPE re-index, batched prefix decode, scoring parity
```

`--method baseline` is one agent with full context; `--method latent_mas` is the upstream chain; `--no_reindex` turns off the position fix.

## Credit

Fork of [LatentMAS](https://github.com/Gen-Verse/LatentMAS) — Zou et al., *Latent Collaboration in Multi-Agent Systems*, [arXiv:2511.20639](https://arxiv.org/abs/2511.20639). The routed methods, KV-cache operations, router, and experiment harness are this fork's additions.

---

Research log, Jul–Oct 2026. Full dated commit history available on request.