"""Compare benchmark and historical results across scoring versions."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
BENCHMARK_PATH = RESULTS_DIR / "version_benchmark_results.csv"
HISTORICAL_RESULT_FILES = {
    "v3": RESULTS_DIR / "session_results_v3.csv",
    "v4": RESULTS_DIR / "session_results_v4_hybrid_keyword_guardrails.csv",
    "v5": RESULTS_DIR / "session_results_v5_llm_rubric.csv",
    "v6": RESULTS_DIR / "session_results_v6_llm_structured_rubric.csv",
}
VERSION_LABELS = {
    "v3_hybrid_semantic_flexible_keyword": "v3",
    "v4_hybrid_keyword_guardrails": "v4",
    "v5_llm_rubric": "v5",
    "v6_llm_structured_rubric": "v6",
}


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    """Load rows from a CSV file if it exists."""
    if not csv_path.exists():
        return []

    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def safe_float(value: str) -> float | None:
    """Convert a string to float when possible."""
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

    position = 0
    while position < len(sorted_indices):
        next_position = position
        while (
            next_position + 1 < len(sorted_indices)
            and values[sorted_indices[next_position + 1]] == values[sorted_indices[position]]
        ):
            next_position += 1

        average_rank = (position + next_position + 2) / 2
        for current_position in range(position, next_position + 1):
            ranks[sorted_indices[current_position]] = average_rank

        position = next_position + 1

    return ranks


def pearson_correlation(xs: list[float], ys: list[float]) -> float | None:
    """Compute Pearson correlation without extra dependencies."""
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
    """Compute Spearman correlation by ranking then using Pearson."""
    if len(xs) < 2 or len(xs) != len(ys):
        return None

    return pearson_correlation(rank_values(xs), rank_values(ys))


def summarize_version(version: str, rows: list[dict[str, str]]) -> None:
    """Print a short summary for one scoring version."""
    print(f"{version.upper()} summary")
    print(f"Rows: {len(rows)}")

    if not rows:
        print("No data.\n")
        return

    final_scores = [
        score for row in rows if (score := safe_float(row.get("final_score", ""))) is not None
    ]
    ratings = [
        rating for row in rows if (rating := safe_float(row.get("rating", ""))) is not None
    ]

    if final_scores:
        print(f"Average final score: {mean(final_scores):.3f}")
    if ratings:
        print(f"Average rating: {mean(ratings):.2f}/5")
    print()


def print_quantitative_metrics(rows_by_version: dict[str, list[dict[str, str]]]) -> None:
    """Print benchmark metrics against expected ratings."""
    print("Quantitative metrics\n")

    for version in ("v3", "v4", "v5", "v6"):
        rows = rows_by_version.get(version, [])
        paired_rows = []
        for row in rows:
            expected_rating = safe_float(row.get("expected_rating", ""))
            predicted_rating = safe_float(row.get("rating", ""))
            if expected_rating is None or predicted_rating is None:
                continue
            paired_rows.append((expected_rating, predicted_rating))

        print(f"{version.upper()} metrics")
        if not paired_rows:
            print("No expected-rating data.\n")
            continue

        absolute_errors = [abs(predicted - expected) for expected, predicted in paired_rows]
        exact_matches = sum(1 for expected, predicted in paired_rows if predicted == expected)
        within_one = sum(1 for expected, predicted in paired_rows if abs(predicted - expected) <= 1)
        over_scored = sum(1 for expected, predicted in paired_rows if predicted > expected)
        under_scored = sum(1 for expected, predicted in paired_rows if predicted < expected)
        expected_values = [expected for expected, _ in paired_rows]
        predicted_values = [predicted for _, predicted in paired_rows]

        pearson_value = pearson_correlation(expected_values, predicted_values)
        spearman_value = spearman_correlation(expected_values, predicted_values)

        print(f"MAE: {mean(absolute_errors):.3f}")
        print(f"Exact accuracy: {exact_matches / len(paired_rows):.3f}")
        print(f"Accuracy within ±1: {within_one / len(paired_rows):.3f}")
        print(f"Over-scored cases: {over_scored}")
        print(f"Under-scored cases: {under_scored}")
        if pearson_value is not None:
            print(f"Pearson correlation: {pearson_value:.3f}")
        if spearman_value is not None:
            print(f"Spearman correlation: {spearman_value:.3f}")
        print()


def print_historical_summary() -> None:
    """Print summary stats for the historical versioned result files."""
    print("Historical summary\n")
    for version in ("v3", "v4", "v5", "v6"):
        summarize_version(version, load_rows(HISTORICAL_RESULT_FILES[version]))


def build_case_index(
    benchmark_rows: list[dict[str, str]],
) -> dict[str, dict[str, dict[str, str]]]:
    """Group benchmark rows by case id and version."""
    case_index: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)

    for row in benchmark_rows:
        version = VERSION_LABELS.get(row.get("scoring_version", ""), row.get("scoring_version", ""))
        case_index[row["case_id"]][version] = row

    return case_index


def print_case_table(case_id: str, version_map: dict[str, dict[str, str]]) -> None:
    """Print one benchmark case side by side."""
    sample_row = next(iter(version_map.values()))
    print(f"Case {case_id}: {sample_row.get('notes', '')}")
    print(f"Question #{sample_row.get('question_number', '')}: {sample_row.get('question', '')}")
    print(f"User answer: {sample_row.get('user_answer', '')}")
    if sample_row.get("expected_rating", ""):
        print(f"Expected rating: {sample_row.get('expected_rating', '')}/5")

    for version in ("v3", "v4", "v5", "v6"):
        row = version_map.get(version)
        if row is None:
            print(f"  {version.upper()}: missing")
            continue

        rating = row.get("rating", "")
        final_score = row.get("final_score", "")
        subscore_parts = []
        if row.get("correctness_subscore", ""):
            subscore_parts.append(f"C={row['correctness_subscore']}")
        if row.get("completeness_subscore", ""):
            subscore_parts.append(f"M={row['completeness_subscore']}")
        if row.get("clarity_subscore", ""):
            subscore_parts.append(f"L={row['clarity_subscore']}")
        subscore_text = f" ({', '.join(subscore_parts)})" if subscore_parts else ""

        print(f"  {version.upper()}: rating={rating}, final_score={final_score}{subscore_text}")

    print()


def print_benchmark_summary(benchmark_rows: list[dict[str, str]]) -> None:
    """Print summary stats from the benchmark file."""
    print("Benchmark summary\n")

    rows_by_version: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in benchmark_rows:
        version = VERSION_LABELS.get(row.get("scoring_version", ""), row.get("scoring_version", ""))
        rows_by_version[version].append(row)

    for version in ("v3", "v4", "v5", "v6"):
        summarize_version(version, rows_by_version[version])

    print_quantitative_metrics(rows_by_version)


def print_benchmark_case_comparison(benchmark_rows: list[dict[str, str]]) -> None:
    """Print side-by-side comparison for each benchmark case."""
    case_index = build_case_index(benchmark_rows)

    print("Benchmark case comparison\n")
    for case_id in sorted(case_index, key=lambda value: int(value)):
        print_case_table(case_id, case_index[case_id])


def print_benchmark_disagreements(benchmark_rows: list[dict[str, str]]) -> None:
    """Print the largest rating spreads from benchmark cases."""
    case_index = build_case_index(benchmark_rows)
    disagreements = []

    for case_id, version_map in case_index.items():
        ratings = [
            safe_float(row.get("rating", ""))
            for row in version_map.values()
            if safe_float(row.get("rating", "")) is not None
        ]
        if len(ratings) < 2:
            continue

        disagreements.append((max(ratings) - min(ratings), case_id, version_map))

    disagreements.sort(reverse=True, key=lambda item: item[0])

    print("Largest benchmark disagreements\n")
    if not disagreements:
        print("No disagreements available.\n")
        return

    for spread, case_id, version_map in disagreements[:10]:
        sample_row = next(iter(version_map.values()))
        print(f"Case {case_id}: spread={spread:.1f}")
        print(f"Question: {sample_row.get('question', '')}")
        print(f"User answer: {sample_row.get('user_answer', '')}")
        for version in ("v3", "v4", "v5", "v6"):
            row = version_map.get(version)
            if row is None:
                continue
            print(f"  {version.upper()}: rating={row.get('rating', '')}, final_score={row.get('final_score', '')}")
        print()


def main() -> None:
    """Run benchmark-first comparison, with historical fallback."""
    if BENCHMARK_PATH.exists():
        benchmark_rows = load_rows(BENCHMARK_PATH)
        print("Interview Coach Benchmark Comparison Report\n")
        print_benchmark_summary(benchmark_rows)
        print_benchmark_case_comparison(benchmark_rows)
        print_benchmark_disagreements(benchmark_rows)
    else:
        print("Interview Coach Historical Comparison Report\n")
        print_historical_summary()
        print("No benchmark file found. Run `python scripts/run_version_benchmark.py` first.")


if __name__ == "__main__":
    main()
