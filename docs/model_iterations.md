# Model Iterations

This document records the main evaluation-driven model iterations in the AI Interview Coach project, including what was tried, what was measured, what changed, and how performance improved.

## Iteration 1: v3 to v4

### What we tried

The `v3` scorer used a hybrid of sentence-embedding similarity and keyword coverage. It worked reasonably well for broad relevance matching, but benchmark cases showed that it could still reward answers that were blank, non-answers, or close restatements of the prompt.

### What we measured

The controlled benchmark in `data/results/version_benchmark_results.csv` showed that `v3` frequently over-scored weak responses such as:

- explicit non-answers
- topical but incorrect answers
- question restatements

Quantitatively:

- `v3` MAE: `1.150`
- `v3` exact accuracy: `0.200`
- `v3` within ±1 accuracy: `0.650`

### What we changed

In `v4`, the scoring pipeline added stronger non-answer guardrails on top of the hybrid semantic-plus-keyword baseline. This version explicitly detects common uncertain responses and prevents them from receiving semantic credit just because they are topically related.

### How performance improved

Compared with `v3`, the `v4` scorer improved benchmark alignment:

- `v4` MAE: `1.000`
- `v4` exact accuracy: `0.300`
- `v4` within ±1 accuracy: `0.700`

This iteration improved handling of weak or empty answers without changing the overall hybrid design.

## Iteration 2: v5 to v6

### What we tried

The `v5` scorer replaced the rule-based hybrid approach with an LLM rubric evaluator. This improved correctness checking substantially, but evaluation results showed that a single overall score was not always easy to interpret or calibrate.

### What we measured

The benchmark showed that `v5` was much stronger than `v3` and `v4`, especially on incorrect or meaningless responses.

Quantitatively:

- `v5` MAE: `0.500`
- `v5` exact accuracy: `0.650`
- `v5` within ±1 accuracy: `0.850`

At the same time, case-level analysis showed that `v5` could be conservative on concise strong answers and did not provide deterministic subscores.

### What we changed

In `v6`, the LLM evaluator was reworked into a structured rubric scorer with deterministic subscores:

- correctness: `0–2`
- completeness: `0–2`
- clarity: `0–2`

The final rating is then computed in Python from those subscores. The prompt was also made stricter about:

- question restatements
- topical but incorrect answers
- missing core concepts

### How performance improved

Compared with `v5`, the `v6` scorer improved overall calibration:

- `v6` MAE: `0.450`
- `v6` exact accuracy: `0.550`
- `v6` within ±1 accuracy: `1.000`
- `v6` Pearson correlation: `0.938`

This iteration improved calibration and produced deterministic rubric subscores, making the evaluator more consistent and easier to interpret.

## Summary

The two main evaluation-driven improvement cycles were:

1. `v3 -> v4`
   Hybrid scorer plus stronger non-answer guardrails.
2. `v5 -> v6`
   LLM rubric scorer plus structured subscores and stricter calibration.

Together, these iterations show a clear pattern of using benchmark results and failure analysis to guide concrete model improvements rather than changing the system arbitrarily.
