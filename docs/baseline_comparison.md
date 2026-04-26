# Baseline Comparison

This document records the simple keyword-only baseline used as a lower-complexity reference point for the interview coach benchmark.

## Keyword-Only Baseline Metrics

| Model | Rows | Avg Final Score | Avg Rating | MAE | Exact Accuracy | Within ±1 Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| keyword-only baseline | 20 | 0.297 | 1.95 | 0.850 | 0.450 | 0.750 |

The baseline is intentionally simple: it scores answers using only keyword coverage against the reference answer. Later versions outperform it by adding sentence embeddings, rubric-based LLM scoring, and stricter evaluation calibration.
