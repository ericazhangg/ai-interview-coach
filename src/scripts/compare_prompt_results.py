"""Summarize prompt-comparison benchmark results into a small table."""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = PROJECT_ROOT / "data" / "results" / "prompt_comparison_results.csv"
SUMMARY_CSV_PATH = PROJECT_ROOT / "data" / "results" / "prompt_comparison_summary.csv"
SUMMARY_MD_PATH = PROJECT_ROOT / "docs" / "prompt_comparison_table.md"


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    """Load prompt comparison rows from CSV."""
    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def compute_metrics(rows: list[dict[str, str]]) -> list[dict[str, str | float | int]]:
    """Compute simple rubric-facing metrics for each prompt variant."""
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["prompt_variant"], []).append(row)

    summary_rows: list[dict[str, str | float | int]] = []
    for prompt_variant, variant_rows in grouped.items():
        expected = [float(row["expected_rating"]) for row in variant_rows]
        predicted = [float(row["rating"]) for row in variant_rows]
        final_scores = [float(row["final_score"]) for row in variant_rows]
        absolute_errors = [abs(p - e) for e, p in zip(expected, predicted)]
        exact_accuracy = sum(1 for e, p in zip(expected, predicted) if e == p) / len(expected)
        within_one_accuracy = sum(
            1 for e, p in zip(expected, predicted) if abs(e - p) <= 1
        ) / len(expected)

        summary_rows.append(
            {
                "prompt_variant": prompt_variant,
                "rows": len(variant_rows),
                "avg_final_score": mean(final_scores),
                "avg_rating": mean(predicted),
                "mae": mean(absolute_errors),
                "exact_accuracy": exact_accuracy,
                "within_one_accuracy": within_one_accuracy,
            }
        )

    return sorted(summary_rows, key=lambda row: row["prompt_variant"])


def save_summary_csv(rows: list[dict[str, str | float | int]], output_path: Path) -> None:
    """Save summary metrics as CSV."""
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_summary_markdown(rows: list[dict[str, str | float | int]], output_path: Path) -> None:
    """Save a short report-ready markdown table."""
    lines = [
        "| Prompt Variant | Rows | Avg Final Score | Avg Rating | MAE | Exact Accuracy | Within ±1 Accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {prompt_variant} | {rows} | {avg_final_score:.3f} | {avg_rating:.2f} | {mae:.3f} | {exact_accuracy:.3f} | {within_one_accuracy:.3f} |".format(
                **row
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Generate summary artifacts for prompt-comparison results."""
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"No prompt comparison results found at {RESULTS_PATH}. Run run_prompt_benchmark.py first."
        )

    rows = load_rows(RESULTS_PATH)
    summary_rows = compute_metrics(rows)
    save_summary_csv(summary_rows, SUMMARY_CSV_PATH)
    save_summary_markdown(summary_rows, SUMMARY_MD_PATH)

    print("Prompt comparison summary")
    for row in summary_rows:
        print(
            f"{row['prompt_variant']}: MAE={row['mae']:.3f}, exact={row['exact_accuracy']:.3f}, within±1={row['within_one_accuracy']:.3f}"
        )
    print(f"Summary CSV saved to: {SUMMARY_CSV_PATH}")
    print(f"Markdown table saved to: {SUMMARY_MD_PATH}")


if __name__ == "__main__":
    main()
