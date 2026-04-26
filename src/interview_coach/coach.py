"""Baseline interview flow utilities."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
FLEXIBLE_HYBRID_SCORING_VERSION = "v3_hybrid_semantic_flexible_keyword"
HYBRID_SCORING_VERSION = "v4_hybrid_keyword_guardrails"
SEMANTIC_WEIGHT = 0.6
COVERAGE_WEIGHT = 0.4
NON_ANSWER_PATTERNS = [
    r"\bi\s+don'?t\s+know\b",
    r"\bdo\s+not\s+know\b",
    r"\bnot\s+sure\b",
    r"\bno\s+idea\b",
    r"\bcan'?t\s+remember\b",
    r"\bcannot\s+remember\b",
    r"\bdon'?t\s+remember\b",
    r"\bcan'?t\s+recall\b",
    r"\bcannot\s+recall\b",
]
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "with",
}


def load_embedding_model(model_name: str = DEFAULT_MODEL_NAME) -> SentenceTransformer:
    """Load the embedding model used for answer comparison."""
    return SentenceTransformer(model_name)


def select_random_question(
    dataset: pd.DataFrame,
    used_question_numbers: set[int | float] | None = None,
    random_state: int | None = None,
) -> pd.Series:
    """Pick one random interview question from the cleaned dataset."""
    available_questions = dataset

    if used_question_numbers:
        available_questions = dataset[
            ~dataset["Question Number"].isin(used_question_numbers)
        ]

    if available_questions.empty:
        raise ValueError("No unused questions are left in the dataset.")

    return available_questions.sample(n=1, random_state=random_state).iloc[0]


def compute_answer_similarity(
    user_answer: str,
    reference_answer: str,
    model: SentenceTransformer,
) -> float:
    """Compare the user answer to the reference answer using cosine similarity."""
    cleaned_user_answer = user_answer.strip()
    cleaned_reference_answer = reference_answer.strip()

    if not cleaned_user_answer:
        return 0.0
    if not cleaned_reference_answer:
        raise ValueError("Reference answer cannot be empty.")

    embeddings = model.encode(
        [cleaned_user_answer, cleaned_reference_answer],
        convert_to_numpy=True,
    )
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

    # Clamp the result to the expected range for easier downstream handling.
    return float(np.clip(similarity, 0.0, 1.0))


def tokenize_text(text: str) -> list[str]:
    """Split text into simple lowercase tokens for keyword matching."""
    return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())


def is_non_answer(text: str) -> bool:
    """Detect common uncertainty phrases that should not receive semantic credit."""
    cleaned_text = text.strip().lower()
    if not cleaned_text:
        return True

    return any(re.search(pattern, cleaned_text) for pattern in NON_ANSWER_PATTERNS)


def expand_special_tokens(tokens: list[str]) -> list[str]:
    """Expand mixed tokens like 2fa into simpler pieces for looser matching."""
    expanded_tokens: list[str] = []
    for token in tokens:
        expanded_tokens.append(token)

        letter_groups = re.findall(r"[a-zA-Z]+|\d+", token)
        if len(letter_groups) > 1:
            expanded_tokens.extend(letter_groups)

    return expanded_tokens


def normalize_token(token: str) -> str:
    """Apply lightweight normalization so close variants can still match."""
    normalized = token.lower()

    # Strip common English suffixes to reduce simple wording differences.
    suffixes = ["ization", "ation", "ments", "ment", "ing", "ers", "ies", "ied", "ed", "es", "s"]
    for suffix in suffixes:
        if len(normalized) > len(suffix) + 2 and normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break

    if normalized.endswith("i") and len(normalized) > 3:
        normalized = normalized[:-1] + "y"

    return normalized


def build_normalized_token_set(text: str) -> set[str]:
    """Create a normalized token set for flexible keyword matching."""
    raw_tokens = tokenize_text(text)
    expanded_tokens = expand_special_tokens(raw_tokens)
    normalized_tokens = {
        normalize_token(token)
        for token in expanded_tokens
        if token not in STOPWORDS and (len(token) >= 3 or token.isdigit())
    }
    return {token for token in normalized_tokens if token}


def extract_reference_keywords(reference_answer: str) -> set[str]:
    """Keep simple content words from the reference answer."""
    return build_normalized_token_set(reference_answer)


def tokens_match(reference_token: str, user_token: str) -> bool:
    """Allow exact and near matches for short technical word variations."""
    if reference_token == user_token:
        return True

    if reference_token.isdigit() or user_token.isdigit():
        return False

    shorter_length = min(len(reference_token), len(user_token))
    if shorter_length < 4:
        return False

    # Accept close prefix matches for paraphrases like initialize/initialization.
    common_prefix_length = 0
    for ref_char, user_char in zip(reference_token, user_token):
        if ref_char != user_char:
            break
        common_prefix_length += 1

    return common_prefix_length >= max(4, shorter_length - 2)


def compute_keyword_coverage(user_answer: str, reference_answer: str) -> float:
    """Estimate how much of the reference content appears in the user answer."""
    cleaned_user_answer = user_answer.strip()
    cleaned_reference_answer = reference_answer.strip()

    if not cleaned_user_answer:
        return 0.0
    if not cleaned_reference_answer:
        raise ValueError("Reference answer cannot be empty.")

    reference_keywords = extract_reference_keywords(cleaned_reference_answer)
    if not reference_keywords:
        return 0.0

    user_tokens = build_normalized_token_set(cleaned_user_answer)
    matched_keywords = {
        reference_keyword
        for reference_keyword in reference_keywords
        if any(tokens_match(reference_keyword, user_token) for user_token in user_tokens)
    }
    return len(matched_keywords) / len(reference_keywords)


def combine_scores(semantic_score: float, coverage_score: float) -> float:
    """Blend semantic similarity with keyword coverage into one baseline score."""
    combined_score = (SEMANTIC_WEIGHT * semantic_score) + (COVERAGE_WEIGHT * coverage_score)
    return float(np.clip(combined_score, 0.0, 1.0))


def similarity_to_rating(similarity_score: float) -> int:
    """Map a similarity score in [0, 1] to a simple 1-5 rating."""
    if similarity_score >= 0.85:
        return 5
    if similarity_score >= 0.70:
        return 4
    if similarity_score >= 0.50:
        return 3
    if similarity_score >= 0.30:
        return 2
    return 1


def generate_feedback(
    final_score: float,
    semantic_score: float,
    coverage_score: float,
) -> dict[str, str]:
    """Return simple rule-based feedback for the baseline."""
    if final_score >= 0.85:
        return {
            "strengths": "Your answer matches the reference answer very closely and covers the main idea well.",
            "improvement": "To make it even stronger, add a short example or a clearer real-world explanation.",
        }
    if final_score >= 0.70:
        return {
            "strengths": "Your answer captures most of the important points from the reference answer.",
            "improvement": "You could improve by adding one or two missing details and making the explanation a bit more precise.",
        }
    if semantic_score >= 0.70 and coverage_score < 0.35:
        return {
            "strengths": "Your answer is related to the reference answer and points in the right direction.",
            "improvement": "Add more specific technical details or key terms from the concept to make the answer complete.",
        }
    if final_score >= 0.50:
        return {
            "strengths": "Your answer shows partial understanding of the concept.",
            "improvement": "Try to explain the core idea more directly and include the key technical details from the reference answer.",
        }
    if final_score >= 0.30:
        return {
            "strengths": "Your answer seems related to the topic.",
            "improvement": "Focus more on the exact concept being asked and include a clearer definition or explanation.",
        }
    return {
        "strengths": "You made an attempt to answer the question.",
        "improvement": "Review the core concept and try again with a more complete and accurate explanation.",
    }


def evaluate_answer(
    user_answer: str,
    reference_answer: str,
    model: SentenceTransformer,
) -> dict[str, Any]:
    """Score a user answer and attach simple baseline feedback."""
    if is_non_answer(user_answer):
        return {
            "semantic_score": 0.0,
            "coverage_score": 0.0,
            "similarity_score": 0.0,
            "rating": 1,
            "strengths": "You made an attempt to respond.",
            "improvement": "Try giving a short definition or key idea, even if you are unsure.",
        }

    semantic_score = compute_answer_similarity(user_answer, reference_answer, model)
    coverage_score = compute_keyword_coverage(user_answer, reference_answer)
    similarity_score = combine_scores(semantic_score, coverage_score)
    rating = similarity_to_rating(similarity_score)
    feedback = generate_feedback(similarity_score, semantic_score, coverage_score)

    return {
        "semantic_score": semantic_score,
        "coverage_score": coverage_score,
        "similarity_score": similarity_score,
        "rating": rating,
        "strengths": feedback["strengths"],
        "improvement": feedback["improvement"],
    }


def evaluate_answer_v3(
    user_answer: str,
    reference_answer: str,
    model: SentenceTransformer,
) -> dict[str, Any]:
    """Reproduce the v3 scorer without the v4 non-answer guardrails."""
    semantic_score = compute_answer_similarity(user_answer, reference_answer, model)
    coverage_score = compute_keyword_coverage(user_answer, reference_answer)
    similarity_score = combine_scores(semantic_score, coverage_score)
    rating = similarity_to_rating(similarity_score)
    feedback = generate_feedback(similarity_score, semantic_score, coverage_score)

    return {
        "semantic_score": semantic_score,
        "coverage_score": coverage_score,
        "similarity_score": similarity_score,
        "rating": rating,
        "strengths": feedback["strengths"],
        "improvement": feedback["improvement"],
    }


def format_question_number(question_number: Any) -> str:
    """Format question numbers cleanly for display."""
    if pd.isna(question_number):
        return "Unknown"

    numeric_question_number = float(question_number)
    if numeric_question_number.is_integer():
        return str(int(numeric_question_number))

    return str(numeric_question_number)
