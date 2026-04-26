"""Run a simple keyword-only baseline on the shared benchmark cases."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from interview_coach.coach import (  # noqa: E402
    KEYWORD_BASELINE_SCORING_VERSION,
    evaluate_answer_keyword_baseline,
)
from interview_coach.data import load_and_clean_dataset  # noqa: E402

DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "Software_Questions.csv"
TEST_CASES_PATH = PROJECT_ROOT / "data" / "evaluation" / "test_cases.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "results" / "baseline_benchmark_results.csv"


def load_test_cases(test_cases_path: Path) -> list[dict[str, str]]:
    """Load benchmark cases from CSV."""
    with test_cases_path.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def build_dataset_index(dataset_rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    """Build a lookup by question number."""
    return {
        str(int(row["Question Number"])) if float(row["Question Number"]).is_integer() else str(row["Question Number"]): row
        for row in dataset_rows
    }


def save_rows(rows: list[dict[str, object]], output_path: Path) -> None:
    """Write baseline benchmark results."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "question_number",
        "question",
        "user_answer",
        "expected_rating",
        "notes",
        "scoring_version",
        "semantic_score",
        "coverage_score",
        "final_score",
        "rating",
        "strengths",
        "improvement",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run the keyword-only baseline on the controlled benchmark set."""
    dataset = load_and_clean_dataset(DATASET_PATH)
    dataset_index = build_dataset_index(dataset.to_dict("records"))
    test_cases = load_test_cases(TEST_CASES_PATH)

    rows: list[dict[str, object]] = []
    for case in test_cases:
        question_number = case["question_number"].strip()
        dataset_row = dataset_index.get(question_number)
        if dataset_row is None:
            raise ValueError(f"Question number {question_number} was not found in the dataset.")

        evaluation = evaluate_answer_keyword_baseline(
            user_answer=case["user_answer"],
            reference_answer=str(dataset_row["Answer"]),
        )
        rows.append(
            {
                "case_id": case["case_id"],
                "question_number": question_number,
                "question": dataset_row["Question"],
                "user_answer": case["user_answer"],
                "expected_rating": case.get("expected_rating", ""),
                "notes": case.get("notes", ""),
                "scoring_version": KEYWORD_BASELINE_SCORING_VERSION,
                "semantic_score": evaluation["semantic_score"],
                "coverage_score": evaluation["coverage_score"],
                "final_score": evaluation["similarity_score"],
                "rating": evaluation["rating"],
                "strengths": evaluation["strengths"],
                "improvement": evaluation["improvement"],
            }
        )

    save_rows(rows, OUTPUT_PATH)
    print("Baseline benchmark complete")
    print(f"Cases evaluated: {len(test_cases)}")
    print(f"Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
