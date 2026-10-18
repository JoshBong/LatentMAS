# Why stitching worker caches fails

Notes from working out what `routed_mas` loses when it concatenates independently computed worker caches (Jul 30 – Aug 2026). This page keeps the conclusions and sources.

## The setup

```
S0 = lead reads the shared context
worker w:  S0_w = clone_cache(S0) → runs its own sub-task, blind to the other workers → suffix_w
judge:     decodes over  cache_concat([S0, reindex(suffix_1), reindex(suffix_2), ...])
```

(`methods/routed_mas.py`, `methods/cache_ops.py`)

Compared to one model reading `[S0, A, B]` in sequence, causality already blocks A from attending to B. The **only** edge parallelism removes is B → A: worker B's tokens never attended to worker A's. A's block is identical to the sequential pass, and the judge's own tokens attend to everything. So the loss is narrower than "the blocks don't know about each other". It is exactly one direction of attention between workers.

## Two separate problems, often lumped together

**1. Positions — solved.** Every worker's block starts right after `S0`, so the blocks collide on position IDs. A closed-form RoPE rotation moves each block to its real offset (`cache_reindex`, derivation in the README). This is necessary but not sufficient: KV Packet ([2604.13226](https://arxiv.org/abs/2604.13226)) notes that reusing a cached document in a new context means accounting for shifts in attention distribution, not just positions, and Cartridges at Scale ([2606.04557](https://arxiv.org/abs/2606.04557)) reports that naively mixing caches trained in isolation collapses to near chance.

**2. Missing attention between workers — the real problem.** It splits into two parts.

| | what's lost | recoverable without training or recompute? |
|---|---|---|
| **Distributional** | parallel-encoded blocks give the judge a flatter, miscalibrated attention distribution | **Yes.** [APE](https://arxiv.org/abs/2502.05431) (ICLR 2025) aligns it with a shared prefix, an attention temperature and a scaling factor, all applied at read time: 98% of sequential performance on RAG, 93% on ICL |
| **Content** | anything B would only have computed because of what A found | **No.** Temperature and scaling reweight what's in the cache; they can't create a value that was never computed |

The content half is the fundamental one. A counterexample makes it concrete:

> "What was company Y's revenue in the year company X filed its first patent?"

Worker A finds the patent year. Worker B, running blind, has no reason to pull that particular year's revenue. Unless B dumps its whole history, the number the judge needs is never in B's cache, and no read-time correction can surface it.

APE's strong numbers fit this picture rather than contradicting it. Retrieved documents and in-context examples are already independent of each other, so APE is working in the regime where only the distributional half exists.

## The condition that falls out

Stitching is sound only when the question is **symmetrically independent**: what each worker needs to extract doesn't depend on what any other worker finds.

- "Which city is more populous, Tokyo or New York?" Independent. Each worker looks up one number and the judge compares them.
- "Highest-grossing movie by Inception's director?" Dependent. You need the director before you know which filmography to search.

In the independent case the judge does a shallow join at the end, and APE-style calibration should close most of the gap. Outside it, no post-hoc operator can help: the fix is to recompute (run the dependent worker after the one it depends on) or not to fan out.

## Can a router tell the two cases apart before fanning out?

Not reliably, in the one test I ran. Zero-shot Qwen2.5-3B as a pre-routing dependency check (`experiments/detect_asymmetric_dependencies.py`, n=6):

| question | truth | model |
|---|---|---|
| capitals of France and Germany | independent | independent ✓ |
| 2022 revenue, Apple vs Microsoft | independent | dependent ✗ |
| more populous, Tokyo vs New York | independent | dependent ✗ |
| Microsoft's revenue in the year Apple filed its first patent | dependent | dependent ✓ |
| US president when Microsoft's founder was born | dependent | dependent ✓ |
| highest-grossing movie by Inception's director | dependent | independent ✗ |

3/6, wrong in both directions. It treats "needs a final comparison" as "workers must talk", but a join at the judge is exactly the safe case. It also misses a real conditional chain. Both errors are expensive: a false positive serializes a question that could have run in parallel, and a false negative runs a dependent chain in parallel and triggers the content loss. With six items and a 3B model this is directional only. A clearer prompt, few-shot examples, or a bigger model would probably help.

## Open

- **Does APE's correction carry over from documents to reasoning traces?** Worker suffixes are generated reasoning with peaked attention, not flat retrieved text. One constant might work, or each block might need its own calibration. Nobody has measured this. It's the experiment that would settle the distributional half for this setting.
- **How does the loss scale with dependence?** I haven't found a measured curve of stitching loss against how much the workers depend on each other.

## Sources

- APE, Yang, Chen, Chen: [arXiv:2502.05431](https://arxiv.org/abs/2502.05431)
- CacheBlend, Yao et al. (training-free, recomputes a subset of tokens): [arXiv:2405.16444](https://arxiv.org/abs/2405.16444)
- KV Packet, Chen et al.: [arXiv:2604.13226](https://arxiv.org/abs/2604.13226)
- Cartridges at Scale, Hardalov et al.: [arXiv:2606.04557](https://arxiv.org/abs/2606.04557)
- KVShareArena, Shi, Zheng, Lou: [arXiv:2609.10266](https://arxiv.org/abs/2609.10266)
- Prompt Cache: [arXiv:2311.04934](https://arxiv.org/abs/2311.04934) · KVLink: [arXiv:2502.16002](https://arxiv.org/abs/2502.16002) · Block-Attention: [arXiv:2409.15355](https://arxiv.org/abs/2409.15355)
- LatentMAS, Zou et al.: [arXiv:2511.20639](https://arxiv.org/abs/2511.20639)
