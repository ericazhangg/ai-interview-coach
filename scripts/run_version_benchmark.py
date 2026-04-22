"""Run the same test cases through v3, v4, v5, and v6 scorers."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from interview_coach.coach import (  # noqa: E402
    FLEXIBLE_HYBRID_SCORING_VERSION,
    HYBRID_SCORING_VERSION,
    evaluate_answer,
    evaluate_answer_v3,
    load_embedding_model,
)
from interview_coach.data import load_and_clean_dataset  # noqa: E402
from interview_coach.llm_evaluator import (  # noqa: E402
    LLM_SCORING_VERSION,
    STRUCTURED_LLM_SCORING_VERSION,
    evaluate_answer_with_llm,
    evaluate_answer_with_structured_llm,
    is_llm_configured,
)

DATASET_PATH = PROJECT_ROOT / "Software Questions.csv"
TEST_CASES_PATH = PROJECT_ROOT / "evaluation" / "test_cases.csv"
OUTPUT_PATH = PROJECT_ROOT / "results" / "version_benchmark_results.csv"


def load_test_cases(test_cases_path: Path) -> list[dict[str, str]]:
    """Load benchmark cases from CSV."""
    with test_cases_path.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def build_dataset_index(dataset_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Build a quick lookup by question number."""
    return {
        str(int(row["Question Number"])) if float(row["Question Number"]).is_integer() else str(row["Question Number"]): row
        for row in dataset_rows
    }


def evaluate_case(
    case: dict[str, str],
    dataset_index: dict[str, dict[str, Any]],
    model: Any,
    llm_enabled: bool,
) -> list[dict[str, Any]]:
    """Score one test case with every available version."""
    question_number = case["question_number"].strip()
    dataset_row = dataset_index.get(question_number)
    if dataset_row is None:
        raise ValueError(f"Question number {question_number} was not found in the dataset.")

    question = dataset_row["Question"]
    reference_answer = dataset_row["Answer"]
    user_answer = case["user_answer"]

    version_outputs = [
        (FLEXIBLE_HYBRID_SCORING_VERSION, evaluate_answer_v3(user_answer, reference_answer, model)),
        (HYBRID_SCORING_VERSION, evaluate_answer(user_answer, reference_answer, model)),
    ]

    if llm_enabled:
        version_outputs.extend(
            [
                (
                    LLM_SCORING_VERSION,
                    evaluate_answer_with_llm(
                        question=question,
                        reference_answer=reference_answer,
                        user_answer=user_answer,
                    ),
                ),
                (
                    STRUCTURED_LLM_SCORING_VERSION,
                    evaluate_answer_with_structured_llm(
                        question=question,
                        reference_answer=reference_answer,
                        user_answer=user_answer,
                    ),
                ),
            ]
        )

    result_rows = []
    for scoring_version, evaluation in version_outputs:
        result_rows.append(
            {
                "case_id": case["case_id"],
                "question_number": question_number,
                "question": question,
                "user_answer": user_answer,
                "expected_rating": case.get("expected_rating", ""),
                "notes": case.get("notes", ""),
                "scoring_version": scoring_version,
                "llm_model": evaluation.get("llm_model", ""),
                "semantic_score": evaluation.get("semantic_score", ""),
                "coverage_score": evaluation.get("coverage_score", ""),
                "final_score": evaluation["similarity_score"],
                "rating": evaluation["rating"],
                "correctness_subscore": evaluation.get("correctness_subscore", ""),
                "completeness_subscore": evaluation.get("completeness_subscore", ""),
                "clarity_subscore": evaluation.get("clarity_subscore", ""),
                "correctness": evaluation.get("correctness", ""),
                "completeness": evaluation.get("completeness", ""),
                "clarity": evaluation.get("clarity", ""),
                "strengths": evaluation["strengths"],
                "improvement": evaluation["improvement"],
            }
        )

    return result_rows


def save_benchmark_results(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Save the benchmark comparison CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "question_number",
        "question",
        "user_answer",
        "expected_rating",
        "notes",
        "scoring_version",
        "llm_model",
        "semantic_score",
        "coverage_score",
        "final_score",
        "rating",
        "correctness_subscore",
        "completeness_subscore",
        "clarity_subscore",
        "correctness",
        "completeness",
        "clarity",
        "strengths",
        "improvement",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run the benchmark on a fixed set of shared test cases."""
    dataset = load_and_clean_dataset(DATASET_PATH)
    dataset_index = build_dataset_index(dataset.to_dict("records"))
    test_cases = load_test_cases(TEST_CASES_PATH)
    model = load_embedding_model()
    llm_enabled = is_llm_configured()

    benchmark_rows: list[dict[str, Any]] = []
    for case in test_cases:
        benchmark_rows.extend(
            evaluate_case(
                case=case,
                dataset_index=dataset_index,
                model=model,
                llm_enabled=llm_enabled,
            )
        )

    save_benchmark_results(benchmark_rows, OUTPUT_PATH)

    print("Benchmark complete")
    print(f"Cases evaluated: {len(test_cases)}")
    print(f"Versions run per case: {'4' if llm_enabled else '2'}")
    print(f"Results saved to: {OUTPUT_PATH}")
    if not llm_enabled:
        print("LLM versions were skipped because no API key is configured.")


if __name__ == "__main__":
    main()
