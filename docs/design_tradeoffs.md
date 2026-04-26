# Design Tradeoffs

This document records a distinct design decision where the project chose between ML approaches based on technical tradeoffs rather than simply taking the highest single metric.

## Decision: choosing v6 over v5 as the final model

The strongest late-stage candidates were:

- `v5_llm_rubric`
- `v6_llm_structured_rubric`

Both versions substantially outperformed the earlier hybrid embedding-based scorers, but they emphasized different strengths.

## Tradeoff considered

### v5 strengths

- better exact accuracy on the benchmark
- better Spearman correlation
- strong handling of clearly incorrect and meaningless answers

### v6 strengths

- lower MAE
- perfect within ±1 accuracy on the benchmark
- higher Pearson correlation
- deterministic correctness, completeness, and clarity subscores
- more interpretable feedback for users

## Why v6 was chosen

The project selected `v6` as the final deployed model because the goal of the system is not only to classify answers correctly in the strictest exact-match sense, but also to provide stable, understandable interview feedback.

That made the following tradeoffs important:

- calibration over exact-match precision alone
- interpretability over a single opaque holistic score
- consistency for user-facing coaching over narrow benchmark wins on one metric

In short, `v5` performed slightly better on some rank-style or exact-match metrics, but `v6` was the better overall deployment choice because it combined strong benchmark accuracy with clearer rubric-based explanations.

## Supporting Evidence

- benchmark summary: `docs/benchmark_metrics_table.md`
- failure analysis: `docs/error_analysis.md`
- iteration log: `docs/model_iterations.md`
