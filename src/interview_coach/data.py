"""Utilities for loading and cleaning the interview dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "Question Number",
    "Question",
    "Answer",
    "Category",
    "Difficulty",
]


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load the raw CSV file and validate the expected columns."""
    encodings_to_try = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]

    dataset = None
    last_error = None
    for encoding in encodings_to_try:
        try:
            # Try a few common encodings because downloaded CSVs are often not UTF-8.
            dataset = pd.read_csv(csv_path, encoding=encoding)
            break
        except UnicodeDecodeError as error:
            last_error = error

    if dataset is None:
        raise ValueError(
            f"Could not decode dataset file {csv_path} with supported encodings."
        ) from last_error

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in dataset.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    return dataset


def clean_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the dataset for downstream retrieval."""
    cleaned = dataset.copy()

    # Normalize text fields so matching behaves more consistently later.
    text_columns = ["Question", "Answer", "Category", "Difficulty"]
    for column in text_columns:
        cleaned[column] = cleaned[column].fillna("").astype(str).str.strip()

    # Drop rows that do not contain the minimum information we need.
    cleaned = cleaned[(cleaned["Question"] != "") & (cleaned["Answer"] != "")]

    # Keep question numbers numeric when possible for cleaner reporting.
    cleaned["Question Number"] = pd.to_numeric(
        cleaned["Question Number"], errors="coerce"
    )

    # Reset the index so later retrieval code can rely on stable row positions.
    cleaned = cleaned.reset_index(drop=True)
    return cleaned


def load_and_clean_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Convenience wrapper used by the baseline pipeline."""
    raw_dataset = load_dataset(csv_path)
    return clean_dataset(raw_dataset)
