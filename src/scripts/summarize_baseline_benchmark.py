"""Summarize the keyword-only baseline benchmark for report-ready use."""

from __future__ import annotations

import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_RESULTS_PATH = PROJECT_ROOT / "data" / "results" / "baseline_benchmark_results.csv"
VERSION_SUMMARY_PATH = PROJECT_ROOT / "data" / "results" / "benchmark_metrics_summary.csv"
OUTPUT_PATH = PROJECT_ROOT / "docs" / "baseline_comparison.md"


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def main() -> None:
    """Generate a short comparison between the simple baseline and later models."""
    baseline_rows = load_csv_rows(BASELINE_RESULTS_PATH)
    if not baseline_rows:
        raise ValueError("Baseline benchmark results are empty.")

    expected = [int(row["expected_rating"]) for row in baseline_rows]
    predicted = [int(row["rating"]) for row in baseline_rows]
    final_scores = [float(row["final_score"]) for row in baseline_rows]

    mae = mean([abs(pred - gold) for pred, gold in zip(predicted, expected)])
    exact_accuracy = mean([1.0 if pred == gold else 0.0 for pred, gold in zip(predicted, expected)])
    within_one_accuracy = mean(
        [1.0 if abs(pred - gold) <= 1 else 0.0 for pred, gold in zip(predicted, expected)]
    )
    average_rating = mean([float(value) for value in predicted])
    average_final_score = mean(final_scores)

    version_rows = load_csv_rows(VERSION_SUMMARY_PATH) if VERSION_SUMMARY_PATH.exists() else []

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        output_file.write("# Baseline Comparison\n\n")
        output_file.write(
            "This document records the simple keyword-only baseline used as a lower-complexity reference point for the interview coach benchmark.\n\n"
        )
        output_file.write("## Keyword-Only Baseline Metrics\n\n")
        output_file.write("| Model | Rows | Avg Final Score | Avg Rating | MAE | Exact Accuracy | Within ±1 Accuracy |\n")
        output_file.write("|---|---:|---:|---:|---:|---:|---:|\n")
        output_file.write(
            f"| keyword-only baseline | {len(baseline_rows)} | {average_final_score:.3f} | {average_rating:.2f} | {mae:.3f} | {exact_accuracy:.3f} | {within_one_accuracy:.3f} |\n"
        )

        if version_rows:
            output_file.write("\n## Comparison to Later Versions\n\n")
            output_file.write("| Model | MAE | Exact Accuracy | Within ±1 Accuracy |\n")
            output_file.write("|---|---:|---:|---:|\n")
            output_file.write(
                f"| keyword-only baseline | {mae:.3f} | {exact_accuracy:.3f} | {within_one_accuracy:.3f} |\n"
            )
            for row in version_rows:
                output_file.write(
                    f"| {row['version']} | {float(row['mae']):.3f} | {float(row['exact_accuracy']):.3f} | {float(row['within_one_accuracy']):.3f} |\n"
                )

        output_file.write(
            "\nThe baseline is intentionally simple: it scores answers using only keyword coverage against the reference answer. "
            "Later versions outperform it by adding sentence embeddings, rubric-based LLM scoring, and stricter evaluation calibration.\n"
        )

    print(f"Baseline summary saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
