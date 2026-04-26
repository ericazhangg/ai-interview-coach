"""Utilities for saving baseline evaluation results.

This file was created with AI help and then reviewed, edited, and tested by me.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "data" / "results"

RESULT_COLUMNS = [
    "session_id",
    "scoring_version",
    "llm_model",
    "question_number",
    "category",
    "difficulty",
    "question",
    "input_mode",
    "transcription_text",
    "tts_used",
    "user_answer",
    "reference_answer",
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


def create_session_id() -> str:
    """Create a simple timestamp-based session id."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_versioned_results_path(scoring_version: str) -> Path:
    """Return a per-version results path so experiments stay separated."""
    safe_version = scoring_version.replace("/", "_").replace(" ", "_")
    return RESULTS_DIR / f"session_results_{safe_version}.csv"


def save_session_results(
    results: list[dict[str, Any]],
    output_path: str | Path = RESULTS_DIR / "session_results.csv",
) -> Path:
    """Append session results to a CSV file for later evaluation."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if output_file.exists():
        with output_file.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.reader(csv_file)
            existing_header = next(reader, [])

        if existing_header != RESULT_COLUMNS:
            existing_rows = list(csv.DictReader(output_file.open(encoding="utf-8")))
            with output_file.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=RESULT_COLUMNS)
                writer.writeheader()
                for row in existing_rows:
                    normalized_row = {column: row.get(column, "") for column in RESULT_COLUMNS}
                    writer.writerow(normalized_row)

    file_exists = output_file.exists()
    with output_file.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=RESULT_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)

    return output_file


def migrate_results_file(
    source_path: str | Path,
    scoring_version: str,
) -> Path | None:
    """Copy an existing results file into a version-specific file if needed."""
    source_file = Path(source_path)
    if not source_file.exists():
        return None

    target_file = get_versioned_results_path(scoring_version)
    if target_file.exists():
        return target_file

    rows = list(csv.DictReader(source_file.open(encoding="utf-8")))
    if not rows:
        return None

    save_session_results(rows, output_path=target_file)
    return target_file


def normalize_results_csv(
    input_path: str | Path = RESULTS_DIR / "session_results.csv",
) -> Path | None:
    """Rewrite a mixed legacy results file into one consistent schema."""
    input_file = Path(input_path)
    if not input_file.exists():
        return None

    with input_file.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        rows = list(reader)

    if not rows:
        return input_file

    normalized_rows: list[dict[str, Any]] = []
    for raw_row in rows[1:]:
        if not raw_row:
            continue

        if len(raw_row) == 11:
            normalized_rows.append(
                {
                    "session_id": raw_row[0],
                    "scoring_version": "v1_semantic_only",
                    "llm_model": "",
                    "question_number": raw_row[1],
                    "category": raw_row[2],
                    "difficulty": raw_row[3],
                    "question": raw_row[4],
                    "input_mode": "",
                    "transcription_text": "",
                    "tts_used": "",
                    "user_answer": raw_row[5],
                    "reference_answer": raw_row[6],
                    "semantic_score": "",
                    "coverage_score": "",
                    "final_score": raw_row[7],
                    "rating": raw_row[8],
                    "correctness_subscore": "",
                    "completeness_subscore": "",
                    "clarity_subscore": "",
                    "correctness": "",
                    "completeness": "",
                    "clarity": "",
                    "strengths": raw_row[9],
                    "improvement": raw_row[10],
                }
            )
        elif len(raw_row) == 13:
            normalized_rows.append(
                {
                    "session_id": raw_row[0],
                    "scoring_version": "v2_hybrid_semantic_keyword",
                    "llm_model": "",
                    "question_number": raw_row[1],
                    "category": raw_row[2],
                    "difficulty": raw_row[3],
                    "question": raw_row[4],
                    "input_mode": "",
                    "transcription_text": "",
                    "tts_used": "",
                    "user_answer": raw_row[5],
                    "reference_answer": raw_row[6],
                    "semantic_score": raw_row[7],
                    "coverage_score": raw_row[8],
                    "final_score": raw_row[9],
                    "rating": raw_row[10],
                    "correctness_subscore": "",
                    "completeness_subscore": "",
                    "clarity_subscore": "",
                    "correctness": "",
                    "completeness": "",
                    "clarity": "",
                    "strengths": raw_row[11],
                    "improvement": raw_row[12],
                }
            )

    backup_file = input_file.with_name("session_results_legacy_mixed.csv")
    backup_file.write_text(input_file.read_text(encoding="utf-8"), encoding="utf-8")

    with input_file.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(normalized_rows)

    return input_file
