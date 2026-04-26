"""LLM-based rubric evaluator for the interview coach."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field


load_dotenv()

LLM_SCORING_VERSION = "v5_llm_rubric"
STRUCTURED_LLM_SCORING_VERSION = "v6_llm_structured_rubric"
DEFAULT_LLM_MODEL = "gpt-5-nano"
DEFAULT_DUKE_BASE_URL = "https://litellm.oit.duke.edu/v1"
PROMPT_VARIANTS = {
    "prompt_a_simple": "Simple structured rubric prompt.",
    "prompt_b_strict": "Stricter interviewer prompt with calibrated scoring.",
    "prompt_c_penalty": "Structured rubric prompt with explicit penalties for vague or misleading answers.",
}


class LlmRubricEvaluation(BaseModel):
    """Structured evaluation returned by the LLM."""

    score: int = Field(ge=1, le=5)
    correctness: str
    completeness: str
    clarity: str
    strengths: list[str]
    improvements: list[str]


class StructuredRubricEvaluation(BaseModel):
    """Structured rubric with deterministic subscores."""

    correctness_subscore: int = Field(ge=0, le=2)
    completeness_subscore: int = Field(ge=0, le=2)
    clarity_subscore: int = Field(ge=0, le=2)
    correctness_reason: str
    completeness_reason: str
    clarity_reason: str
    strengths: list[str]
    improvements: list[str]


def format_list_for_csv(items: list[str]) -> str:
    """Convert a list of feedback points into a clean CSV-friendly string."""
    cleaned_items = [item.strip().lstrip("-").strip() for item in items if item.strip()]
    return " | ".join(cleaned_items)


def get_config_value(name: str, default: str = "") -> str:
    """Read configuration from env vars first, then Streamlit secrets if available."""
    env_value = os.getenv(name)
    if env_value:
        return env_value

    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass

    return default


def is_llm_configured() -> bool:
    """Return True when the OpenAI API key is available."""
    return bool(get_api_key())


def get_api_key() -> str:
    """Return the configured API key from supported environment variables."""
    return get_config_value("OPENAI_API_KEY") or get_config_value("DUKE_AI_API_KEY")


def get_base_url() -> str | None:
    """Return the configured base URL when using a proxy or gateway."""
    openai_base_url = get_config_value("OPENAI_BASE_URL")
    if openai_base_url:
        return openai_base_url

    if get_config_value("DUKE_AI_API_KEY"):
        return DEFAULT_DUKE_BASE_URL

    return None


def get_llm_model_name() -> str:
    """Return the configured model name for rubric evaluation."""
    return get_config_value("OPENAI_MODEL", DEFAULT_LLM_MODEL)


def build_rubric_prompt(
    question: str,
    reference_answer: str,
    user_answer: str,
) -> str:
    """Build the rubric prompt for the interview coach evaluator."""
    return f"""
You are an expert software engineering interviewer.

Evaluate the candidate answer using this rubric:
- Correctness: Is the answer technically accurate?
- Completeness: Does the answer cover the main ideas expected by the reference answer?
- Clarity: Is the explanation understandable and reasonably well expressed?

Scoring instructions:
- Use a 1 to 5 scale.
- A 1 means the answer is missing, mostly incorrect, or not useful.
- A 3 means the answer is partially correct but incomplete or unclear.
- A 5 means the answer is accurate, complete, and clearly explained.
- Do not give extra credit for style if the answer is incorrect.
- If the candidate says they do not know, score it low.

Interview Question:
{question}

Reference Answer:
{reference_answer}

Candidate Answer:
{user_answer}
""".strip()


def build_structured_rubric_prompt(
    question: str,
    reference_answer: str,
    user_answer: str,
) -> str:
    """Build the stricter v6 rubric prompt."""
    return f"""
You are an expert software engineering interviewer.

Evaluate the candidate answer using these three subscores only:
- correctness_subscore: 0 to 2
- completeness_subscore: 0 to 2
- clarity_subscore: 0 to 2

Definitions:
- correctness_subscore:
  0 = technically wrong or essentially missing
  1 = partially correct or mixed
  2 = technically correct
- completeness_subscore:
  0 = misses the main concept
  1 = covers part of the expected answer
  2 = covers the main ideas well
- clarity_subscore:
  0 = unclear, confusing, or only restates the prompt
  1 = understandable but rough or incomplete
  2 = clear and easy to follow

Important rules:
- Do not reward answers that only restate the question.
- Penalize answers that are topical but incorrect.
- Penalize answers missing the central concept from the reference answer.
- If the candidate says they do not know, use low subscores.
- Be strict and consistent.

Interview Question:
{question}

Reference Answer:
{reference_answer}

Candidate Answer:
{user_answer}
""".strip()


def build_structured_prompt_variant(
    question: str,
    reference_answer: str,
    user_answer: str,
    prompt_variant: str,
) -> str:
    """Build one of the prompt-engineering variants used for comparison."""
    if prompt_variant == "prompt_a_simple":
        return f"""
You are an expert software engineering interviewer.

Evaluate the candidate answer using these subscores:
- correctness_subscore: 0 to 2
- completeness_subscore: 0 to 2
- clarity_subscore: 0 to 2

Scoring guidance:
- 0 = poor or missing
- 1 = partial
- 2 = strong

Return short reasons plus strengths and improvements.

Interview Question:
{question}

Reference Answer:
{reference_answer}

Candidate Answer:
{user_answer}
""".strip()

    if prompt_variant == "prompt_b_strict":
        return build_structured_rubric_prompt(
            question=question,
            reference_answer=reference_answer,
            user_answer=user_answer,
        )

    if prompt_variant == "prompt_c_penalty":
        return f"""
You are an expert software engineering interviewer.

Evaluate the candidate answer using these three subscores only:
- correctness_subscore: 0 to 2
- completeness_subscore: 0 to 2
- clarity_subscore: 0 to 2

Definitions:
- correctness_subscore:
  0 = technically wrong, misleading, or missing
  1 = partially correct or mixed
  2 = technically correct
- completeness_subscore:
  0 = misses the central concept
  1 = covers some expected ideas
  2 = covers the main ideas well
- clarity_subscore:
  0 = vague, confusing, or just repeats the question
  1 = understandable but rough
  2 = clear and direct

Penalty rules:
- Do not reward answers that only restate the question.
- Do not give high scores to answers that sound relevant but are incorrect.
- Penalize vague answers that avoid the technical point.
- If the candidate says they do not know, keep scores low.
- Be conservative rather than generous.

Interview Question:
{question}

Reference Answer:
{reference_answer}

Candidate Answer:
{user_answer}
""".strip()

    raise ValueError(f"Unsupported prompt variant: {prompt_variant}")


def compute_structured_rating(
    correctness_subscore: int,
    completeness_subscore: int,
    clarity_subscore: int,
) -> tuple[int, float]:
    """Map deterministic subscores to a rating and normalized final score."""
    total_subscore = correctness_subscore + completeness_subscore + clarity_subscore

    if total_subscore <= 1:
        rating = 1
    elif total_subscore == 2:
        rating = 2
    elif total_subscore == 3:
        rating = 3
    elif total_subscore == 4:
        rating = 4
    else:
        rating = 5

    return rating, total_subscore / 6


def build_llm_client() -> OpenAI:
    """Create the OpenAI client using either OpenAI or the Duke gateway."""
    return OpenAI(
        api_key=get_api_key(),
        base_url=get_base_url(),
    )


def evaluate_answer_with_llm(
    question: str,
    reference_answer: str,
    user_answer: str,
) -> dict[str, Any]:
    """Evaluate an answer using an OpenAI model with structured output."""
    if not is_llm_configured():
        raise ValueError(
            "No API key found. Set OPENAI_API_KEY or DUKE_AI_API_KEY to use the v5 LLM evaluator."
        )

    client = build_llm_client()
    model_name = get_llm_model_name()

    response = client.responses.parse(
        model=model_name,
        input=[
            {
                "role": "system",
                "content": (
                    "You evaluate interview answers fairly and return structured rubric feedback."
                ),
            },
            {
                "role": "user",
                "content": build_rubric_prompt(
                    question=question,
                    reference_answer=reference_answer,
                    user_answer=user_answer,
                ),
            },
        ],
        text_format=LlmRubricEvaluation,
    )

    parsed_output = response.output_parsed
    if parsed_output is None:
        raise ValueError("The LLM response could not be parsed into the expected format.")

    final_score = parsed_output.score / 5

    return {
        "llm_model": model_name,
        "semantic_score": "",
        "coverage_score": "",
        "similarity_score": final_score,
        "rating": parsed_output.score,
        "correctness": parsed_output.correctness,
        "completeness": parsed_output.completeness,
        "clarity": parsed_output.clarity,
        "strengths": format_list_for_csv(parsed_output.strengths),
        "improvement": format_list_for_csv(parsed_output.improvements),
    }


def evaluate_answer_with_structured_llm(
    question: str,
    reference_answer: str,
    user_answer: str,
) -> dict[str, Any]:
    """Evaluate an answer using the stricter v6 structured rubric."""
    return evaluate_answer_with_structured_prompt_variant(
        question=question,
        reference_answer=reference_answer,
        user_answer=user_answer,
        prompt_variant="prompt_b_strict",
    )


def evaluate_answer_with_structured_prompt_variant(
    question: str,
    reference_answer: str,
    user_answer: str,
    prompt_variant: str,
) -> dict[str, Any]:
    """Evaluate an answer with one of the structured prompt variants."""
    if not is_llm_configured():
        raise ValueError(
            "No API key found. Set OPENAI_API_KEY or DUKE_AI_API_KEY to use the v6 LLM evaluator."
        )

    client = build_llm_client()
    model_name = get_llm_model_name()

    response = client.responses.parse(
        model=model_name,
        input=[
            {
                "role": "system",
                "content": (
                    "You evaluate interview answers with strict structured scoring and calibrated subscores."
                ),
            },
            {
                "role": "user",
                "content": build_structured_prompt_variant(
                    question=question,
                    reference_answer=reference_answer,
                    user_answer=user_answer,
                    prompt_variant=prompt_variant,
                ),
            },
        ],
        text_format=StructuredRubricEvaluation,
    )

    parsed_output = response.output_parsed
    if parsed_output is None:
        raise ValueError("The v6 LLM response could not be parsed into the expected format.")

    rating, final_score = compute_structured_rating(
        correctness_subscore=parsed_output.correctness_subscore,
        completeness_subscore=parsed_output.completeness_subscore,
        clarity_subscore=parsed_output.clarity_subscore,
    )

    return {
        "llm_model": model_name,
        "prompt_variant": prompt_variant,
        "semantic_score": "",
        "coverage_score": "",
        "similarity_score": final_score,
        "rating": rating,
        "correctness": parsed_output.correctness_reason,
        "completeness": parsed_output.completeness_reason,
        "clarity": parsed_output.clarity_reason,
        "correctness_subscore": parsed_output.correctness_subscore,
        "completeness_subscore": parsed_output.completeness_subscore,
        "clarity_subscore": parsed_output.clarity_subscore,
        "strengths": format_list_for_csv(parsed_output.strengths),
        "improvement": format_list_for_csv(parsed_output.improvements),
    }
