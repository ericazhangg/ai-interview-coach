"""Simple Streamlit UI for the AI Interview Coach."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from interview_coach.coach import (
    HYBRID_SCORING_VERSION,
    evaluate_answer,
    format_question_number,
    load_embedding_model,
    select_random_question,
)
from interview_coach.data import load_and_clean_dataset
from interview_coach.llm_evaluator import (
    LLM_SCORING_VERSION,
    STRUCTURED_LLM_SCORING_VERSION,
    evaluate_answer_with_llm,
    evaluate_answer_with_structured_llm,
    is_llm_configured,
)
from interview_coach.results import (
    create_session_id,
    get_versioned_results_path,
    save_session_results,
)


CSV_PATH = Path("Software Questions.csv")
SCORER_OPTIONS = {
    "v4 Hybrid Baseline": ("hybrid", HYBRID_SCORING_VERSION),
    "v5 LLM Rubric": ("llm", LLM_SCORING_VERSION),
    "v6 Structured LLM Rubric": ("llm_structured", STRUCTURED_LLM_SCORING_VERSION),
}


@st.cache_data
def get_dataset():
    """Load the cleaned dataset once for the UI session."""
    return load_and_clean_dataset(CSV_PATH)


@st.cache_resource
def get_hybrid_model():
    """Load the embedding model only when the hybrid scorer is used."""
    return load_embedding_model()


def ensure_session_state() -> None:
    """Initialize session state used by the Streamlit app."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = create_session_id()
    if "used_question_numbers" not in st.session_state:
        st.session_state.used_question_numbers = set()
    if "current_question" not in st.session_state:
        st.session_state.current_question = None
    if "last_result" not in st.session_state:
        st.session_state.last_result = None


def choose_new_question(dataset) -> None:
    """Pick and store a new random question for the current UI session."""
    selected_question = select_random_question(
        dataset,
        used_question_numbers=st.session_state.used_question_numbers,
    )
    st.session_state.used_question_numbers.add(selected_question["Question Number"])
    st.session_state.current_question = selected_question
    st.session_state.last_result = None


def evaluate_current_answer(user_answer: str, evaluator_mode: str):
    """Run the selected evaluator on the current question and answer."""
    question_row = st.session_state.current_question
    if question_row is None:
        raise ValueError("No question is currently selected.")

    if evaluator_mode == "llm":
        return evaluate_answer_with_llm(
            question=question_row["Question"],
            reference_answer=question_row["Answer"],
            user_answer=user_answer,
        )
    if evaluator_mode == "llm_structured":
        return evaluate_answer_with_structured_llm(
            question=question_row["Question"],
            reference_answer=question_row["Answer"],
            user_answer=user_answer,
        )

    hybrid_model = get_hybrid_model()
    return evaluate_answer(
        user_answer=user_answer,
        reference_answer=question_row["Answer"],
        model=hybrid_model,
    )


def save_ui_result(evaluation: dict, scoring_version: str, user_answer: str) -> Path:
    """Persist one UI evaluation row using the existing results format."""
    question_row = st.session_state.current_question
    result_row = {
        "session_id": st.session_state.session_id,
        "scoring_version": scoring_version,
        "llm_model": evaluation.get("llm_model", ""),
        "question_number": format_question_number(question_row["Question Number"]),
        "category": question_row["Category"],
        "difficulty": question_row["Difficulty"],
        "question": question_row["Question"],
        "user_answer": user_answer,
        "reference_answer": question_row["Answer"],
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
    output_path = get_versioned_results_path(scoring_version)
    save_session_results([result_row], output_path=output_path)
    return output_path


def render_result(evaluation: dict, scoring_version: str, saved_path: Path) -> None:
    """Display the evaluation output clearly in the UI."""
    question_row = st.session_state.current_question

    st.subheader("Results")
    metadata_columns = st.columns(4)
    metadata_columns[0].metric("Rating", f"{evaluation['rating']}/5")
    metadata_columns[1].metric("Final Score", f"{evaluation['similarity_score']:.3f}")
    metadata_columns[2].metric("Category", question_row["Category"])
    metadata_columns[3].metric("Difficulty", question_row["Difficulty"])

    if scoring_version == HYBRID_SCORING_VERSION:
        detail_columns = st.columns(2)
        detail_columns[0].metric("Semantic Score", f"{evaluation['semantic_score']:.3f}")
        detail_columns[1].metric("Keyword Coverage", f"{evaluation['coverage_score']:.3f}")
    elif scoring_version == STRUCTURED_LLM_SCORING_VERSION:
        subscore_columns = st.columns(3)
        subscore_columns[0].metric(
            "Correctness Subscore", f"{evaluation.get('correctness_subscore', '')}/2"
        )
        subscore_columns[1].metric(
            "Completeness Subscore", f"{evaluation.get('completeness_subscore', '')}/2"
        )
        subscore_columns[2].metric(
            "Clarity Subscore", f"{evaluation.get('clarity_subscore', '')}/2"
        )

    if evaluation.get("correctness"):
        st.markdown("**Correctness**")
        st.write(evaluation["correctness"])
    if evaluation.get("completeness"):
        st.markdown("**Completeness**")
        st.write(evaluation["completeness"])
    if evaluation.get("clarity"):
        st.markdown("**Clarity**")
        st.write(evaluation["clarity"])

    st.markdown("**Strengths**")
    st.write(evaluation["strengths"])
    st.markdown("**Areas for improvement**")
    st.write(evaluation["improvement"])
    st.markdown("**Reference answer**")
    st.write(question_row["Answer"])
    st.caption(f"Saved to {saved_path}")


def main() -> None:
    """Run the Streamlit interview coach interface."""
    st.set_page_config(page_title="AI Interview Coach", page_icon="💬", layout="wide")
    st.title("AI Interview Coach")
    st.write(
        "Practice software engineering interview questions and compare the hybrid baseline against the LLM evaluators."
    )

    dataset = get_dataset()
    ensure_session_state()

    scorer_label = st.selectbox("Scoring model", list(SCORER_OPTIONS.keys()), index=2)
    evaluator_mode, scoring_version = SCORER_OPTIONS[scorer_label]

    if evaluator_mode != "hybrid" and not is_llm_configured():
        st.warning(
            "No API key detected. Add OPENAI_API_KEY or DUKE_AI_API_KEY in your .env file to use the LLM scorers."
        )

    controls = st.columns([1, 1, 3])
    if controls[0].button("New Question", type="primary") or st.session_state.current_question is None:
        choose_new_question(dataset)
    if controls[1].button("Reset Session"):
        st.session_state.session_id = create_session_id()
        st.session_state.used_question_numbers = set()
        choose_new_question(dataset)

    question_row = st.session_state.current_question
    if question_row is None:
        st.stop()

    st.subheader("Question")
    st.write(question_row["Question"])
    info_columns = st.columns(3)
    info_columns[0].caption(f"Question #{format_question_number(question_row['Question Number'])}")
    info_columns[1].caption(f"Category: {question_row['Category']}")
    info_columns[2].caption(f"Difficulty: {question_row['Difficulty']}")

    with st.form("answer_form"):
        user_answer = st.text_area("Your answer", height=180)
        submitted = st.form_submit_button("Evaluate Answer")

    if submitted:
        if not user_answer.strip():
            st.error("Please enter an answer before evaluating.")
        elif evaluator_mode != "hybrid" and not is_llm_configured():
            st.error("LLM scoring is not configured. Add an API key in .env first.")
        else:
            with st.spinner("Evaluating answer..."):
                evaluation = evaluate_current_answer(user_answer.strip(), evaluator_mode)
                saved_path = save_ui_result(evaluation, scoring_version, user_answer.strip())
                st.session_state.last_result = (evaluation, scoring_version, saved_path)

    if st.session_state.last_result is not None:
        evaluation, last_scoring_version, saved_path = st.session_state.last_result
        render_result(evaluation, last_scoring_version, saved_path)


if __name__ == "__main__":
    main()
