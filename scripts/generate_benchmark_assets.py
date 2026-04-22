"""Generate report-ready benchmark tables and SVG visualizations."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
BENCHMARK_PATH = RESULTS_DIR / "version_benchmark_results.csv"
METRICS_TABLE_PATH = RESULTS_DIR / "benchmark_metrics_table.md"
METRICS_CSV_PATH = RESULTS_DIR / "benchmark_metrics_summary.csv"
METRICS_SVG_PATH = RESULTS_DIR / "benchmark_metrics_chart.svg"

VERSION_ORDER = [
    "v3_hybrid_semantic_flexible_keyword",
    "v4_hybrid_keyword_guardrails",
    "v5_llm_rubric",
    "v6_llm_structured_rubric",
]
VERSION_LABELS = {
    "v3_hybrid_semantic_flexible_keyword": "v3",
    "v4_hybrid_keyword_guardrails": "v4",
    "v5_llm_rubric": "v5",
    "v6_llm_structured_rubric": "v6",
}


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    """Load rows from a CSV file."""
    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def safe_float(value: str) -> float | None:
    """Convert a value to float when possible."""
    if value in {"", None}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def rank_values(values: list[float]) -> list[float]:
    """Assign average ranks to values for Spearman correlation."""
    sorted_indices = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(sorted_indices):
        end = start
        while (
            end + 1 < len(sorted_indices)
            and values[sorted_indices[end + 1]] == values[sorted_indices[start]]
        ):
            end += 1
        average_rank = (start + end + 2) / 2
        for index in range(start, end + 1):
            ranks[sorted_indices[index]] = average_rank
        start = end + 1
    return ranks


def pearson_correlation(xs: list[float], ys: list[float]) -> float | None:
    """Compute Pearson correlation."""
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x = mean(xs)
    mean_y = mean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator_x = sum((x - mean_x) ** 2 for x in xs) ** 0.5
    denominator_y = sum((y - mean_y) ** 2 for y in ys) ** 0.5
    if denominator_x == 0 or denominator_y == 0:
        return None
    return numerator / (denominator_x * denominator_y)


def spearman_correlation(xs: list[float], ys: list[float]) -> float | None:
    """Compute Spearman correlation."""
    return pearson_correlation(rank_values(xs), rank_values(ys))


def compute_metrics(rows: list[dict[str, str]]) -> list[dict[str, str | float]]:
    """Compute aggregate metrics for each scoring version."""
    rows_by_version: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_version[row["scoring_version"]].append(row)

    metrics_rows: list[dict[str, str | float]] = []
    for version in VERSION_ORDER:
        version_rows = rows_by_version.get(version, [])
        expected_predicted = []
        final_scores = []
        ratings = []
        for row in version_rows:
            expected = safe_float(row.get("expected_rating", ""))
            predicted = safe_float(row.get("rating", ""))
            final_score = safe_float(row.get("final_score", ""))
            if expected is not None and predicted is not None:
                expected_predicted.append((expected, predicted))
                ratings.append(predicted)
            if final_score is not None:
                final_scores.append(final_score)

        absolute_errors = [abs(predicted - expected) for expected, predicted in expected_predicted]
        exact_matches = sum(1 for expected, predicted in expected_predicted if expected == predicted)
        within_one = sum(1 for expected, predicted in expected_predicted if abs(expected - predicted) <= 1)
        over_scored = sum(1 for expected, predicted in expected_predicted if predicted > expected)
        under_scored = sum(1 for expected, predicted in expected_predicted if predicted < expected)
        expected_values = [expected for expected, _ in expected_predicted]
        predicted_values = [predicted for _, predicted in expected_predicted]

        metrics_rows.append(
            {
                "version": VERSION_LABELS[version],
                "rows": len(version_rows),
                "avg_final_score": mean(final_scores) if final_scores else 0.0,
                "avg_rating": mean(ratings) if ratings else 0.0,
                "mae": mean(absolute_errors) if absolute_errors else 0.0,
                "exact_accuracy": exact_matches / len(expected_predicted) if expected_predicted else 0.0,
                "within_one_accuracy": within_one / len(expected_predicted) if expected_predicted else 0.0,
                "over_scored": over_scored,
                "under_scored": under_scored,
                "pearson": pearson_correlation(expected_values, predicted_values) or 0.0,
                "spearman": spearman_correlation(expected_values, predicted_values) or 0.0,
            }
        )

    return metrics_rows


def write_metrics_csv(metrics_rows: list[dict[str, str | float]]) -> None:
    """Write metrics summary as CSV."""
    fieldnames = list(metrics_rows[0].keys())
    with METRICS_CSV_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_rows)


def write_metrics_markdown(metrics_rows: list[dict[str, str | float]]) -> None:
    """Write a report-ready markdown table."""
    lines = [
        "| Version | Rows | Avg Final Score | Avg Rating | MAE | Exact Accuracy | Within ±1 Accuracy | Over-scored | Under-scored | Pearson | Spearman |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics_rows:
        lines.append(
            "| {version} | {rows} | {avg_final_score:.3f} | {avg_rating:.2f} | {mae:.3f} | {exact_accuracy:.3f} | {within_one_accuracy:.3f} | {over_scored} | {under_scored} | {pearson:.3f} | {spearman:.3f} |".format(
                **row
            )
        )
    METRICS_TABLE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_svg_chart(metrics_rows: list[dict[str, str | float]]) -> None:
    """Write a simple SVG bar chart for key benchmark metrics."""
    width = 900
    height = 420
    margin_left = 80
    margin_bottom = 70
    chart_height = 260
    chart_width = 760
    baseline_y = 320
    categories = ["MAE", "Exact", "Within1", "Pearson", "Spearman"]
    colors = {
        "v3": "#D95F02",
        "v4": "#7570B3",
        "v5": "#1B9E77",
        "v6": "#E7298A",
    }

    category_positions = [margin_left + i * (chart_width / len(categories)) for i in range(len(categories))]
    bar_width = 24
    offsets = [-42, -14, 14, 42]

    def metric_value(row: dict[str, str | float], category: str) -> float:
        if category == "MAE":
            # Lower is better, so invert for the chart.
            return max(0.0, 1.5 - float(row["mae"])) / 1.5
        if category == "Exact":
            return float(row["exact_accuracy"])
        if category == "Within1":
            return float(row["within_one_accuracy"])
        if category == "Pearson":
            return float(row["pearson"])
        return float(row["spearman"])

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text { font-family: Arial, sans-serif; fill: #222; } .title { font-size: 22px; font-weight: 700; } .label { font-size: 13px; } .small { font-size: 12px; }</style>',
        f'<text x="{width/2}" y="34" text-anchor="middle" class="title">Benchmark Metrics by Model Version</text>',
        f'<line x1="{margin_left}" y1="{baseline_y}" x2="{margin_left + chart_width}" y2="{baseline_y}" stroke="#333" stroke-width="2"/>',
    ]

    for tick in range(6):
        y = baseline_y - (tick * chart_height / 5)
        value = tick / 5
        svg_lines.append(f'<line x1="{margin_left}" y1="{y}" x2="{margin_left + chart_width}" y2="{y}" stroke="#ddd" stroke-width="1"/>')
        svg_lines.append(f'<text x="{margin_left - 12}" y="{y + 4}" text-anchor="end" class="small">{value:.1f}</text>')

    for category, x in zip(categories, category_positions):
        svg_lines.append(f'<text x="{x}" y="{baseline_y + 28}" text-anchor="middle" class="label">{category}</text>')

    for row_index, row in enumerate(metrics_rows):
        version = str(row["version"])
        for category, x in zip(categories, category_positions):
            normalized_value = metric_value(row, category)
            bar_height = normalized_value * chart_height
            bar_x = x + offsets[row_index] - bar_width / 2
            bar_y = baseline_y - bar_height
            svg_lines.append(
                f'<rect x="{bar_x:.1f}" y="{bar_y:.1f}" width="{bar_width}" height="{bar_height:.1f}" fill="{colors[version]}"/>'
            )

    legend_x = 620
    legend_y = 70
    for index, row in enumerate(metrics_rows):
        version = str(row["version"])
        y = legend_y + index * 24
        svg_lines.append(f'<rect x="{legend_x}" y="{y - 10}" width="14" height="14" fill="{colors[version]}"/>')
        svg_lines.append(f'<text x="{legend_x + 24}" y="{y + 2}" class="label">{version}</text>')

    svg_lines.append(
        '<text x="80" y="380" class="small">MAE is inverted in this chart so taller bars indicate better performance, consistent with the other metrics.</text>'
    )
    svg_lines.append('</svg>')

    METRICS_SVG_PATH.write_text("\n".join(svg_lines), encoding="utf-8")


def main() -> None:
    """Generate benchmark report assets."""
    rows = load_rows(BENCHMARK_PATH)
    metrics_rows = compute_metrics(rows)
    write_metrics_csv(metrics_rows)
    write_metrics_markdown(metrics_rows)
    write_svg_chart(metrics_rows)
    print("Generated benchmark assets")
    print(f"Markdown table: {METRICS_TABLE_PATH}")
    print(f"Metrics CSV: {METRICS_CSV_PATH}")
    print(f"SVG chart: {METRICS_SVG_PATH}")


if __name__ == "__main__":
    main()
