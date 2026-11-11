# Parity with the research harness

Check: the library's `SPLens.read` against the published demo export (`demo/data/smollm3_3b.json`, "JR+prev" lists), using the same fitted V4 SmolLM3-3B lenses and the same calibration recipe (25 pile-10k prompts of 128 tokens). First 10 non-poetry confirmation items, 34 source layers each.

| Run | Token overlap per layer list | Identical lists |
|---|---|---|
| Mac MPS, bfloat16 | 88.8% | 22 / 340 |
| Mac MPS, float32 | 91.1% | 49 / 340 |

The demo export ran on CUDA in bfloat16. A line-by-line diff of `scripts/demo_keywords.py` against `src/splens/lens.py` finds no logic difference (PMI 0.5, pool 200, surface dedupe, prompt tokens removed, weights R 1.0 / J 0.25 / logit 0.25 / prev 0.5, novelty 0.35 capped at 3, same tie-break). The differences we saw are lower-rank swaps between near-tied tokens, for example at layer 20 of the last item, where 9 of 10 tokens agree.

Not yet done: the same check on CUDA bfloat16 on the 5090 (the GPU was fully in use), and a rerun of the 421-item hidden-concept rate through the library. Until then, parity is "same algorithm, close numerically", not "bit-identical".
