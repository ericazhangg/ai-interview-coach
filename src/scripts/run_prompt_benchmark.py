"""Run the same benchmark cases through multiple structured prompt variants."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from interview_coach.data import load_and_clean_dataset  # noqa: E402
from interview_coach.llm_evaluator import (  # noqa: E402
    PROMPT_VARIANTS,
    evaluate_answer_with_structured_prompt_variant,
    get_llm_model_name,
    is_llm_configured,
)

DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "Software_Questions.csv"
TEST_CASES_PATH = PROJECT_ROOT / "data" / "evaluation" / "test_cases.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "results" / "prompt_comparison_results.csv"


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
) -> list[dict[str, Any]]:
    """Score one test case with every prompt variant."""
    question_number = case["question_number"].strip()
    dataset_row = dataset_index.get(question_number)
    if dataset_row is None:
        raise ValueError(f"Question number {question_number} was not found in the dataset.")

    question = dataset_row["Question"]
    reference_answer = dataset_row["Answer"]
    user_answer = case["user_answer"]

    rows = []
    for prompt_variant in PROMPT_VARIANTS:
        evaluation = evaluate_answer_with_structured_prompt_variant(
            question=question,
            reference_answer=reference_answer,
            user_answer=user_answer,
            prompt_variant=prompt_variant,
        )
        rows.append(
            {
                "case_id": case["case_id"],
                "question_number": question_number,
                "question": question,
                "user_answer": user_answer,
                "expected_rating": case.get("expected_rating", ""),
                "notes": case.get("notes", ""),
                "prompt_variant": prompt_variant,
                "llm_model": evaluation.get("llm_model", ""),
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
    return rows


def save_results(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Save prompt comparison rows to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "question_number",
        "question",
        "user_answer",
        "expected_rating",
        "notes",
        "prompt_variant",
        "llm_model",
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
    """Run prompt-comparison benchmark using the structured rubric variants."""
    if not is_llm_configured():
        raise ValueError("No API key found. Configure OPENAI_API_KEY or DUKE_AI_API_KEY first.")

    dataset = load_and_clean_dataset(DATASET_PATH)
    dataset_index = build_dataset_index(dataset.to_dict("records"))
    test_cases = load_test_cases(TEST_CASES_PATH)

    all_rows: list[dict[str, Any]] = []
    for case in test_cases:
        all_rows.extend(evaluate_case(case=case, dataset_index=dataset_index))

    save_results(all_rows, OUTPUT_PATH)
    print("Prompt benchmark complete")
    print(f"Cases evaluated: {len(test_cases)}")
    print(f"Prompt variants run per case: {len(PROMPT_VARIANTS)}")
    print(f"Model used: {get_llm_model_name()}")
    print(f"Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
