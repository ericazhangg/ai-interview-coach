"""Command-line baseline for the AI Interview Coach.

This file was created with AI help and then reviewed, edited, and tested by me.
"""

import os
from pathlib import Path

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


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "Software_Questions.csv"


def ask_to_continue() -> bool:
    """Return True when the user wants another question."""
    user_choice = input("Try another question? (y/n): ").strip().lower()
    return user_choice in {"y", "yes"}


def get_evaluator_mode() -> str:
    """Return the configured evaluator mode."""
    return os.getenv("INTERVIEW_COACH_SCORER", "hybrid").strip().lower()


def main() -> None:
    """Run the baseline interview coach for one or more questions."""
    evaluator_mode = get_evaluator_mode()
    if evaluator_mode == "llm":
        scoring_version = LLM_SCORING_VERSION
        if not is_llm_configured():
            raise ValueError(
                "No API key found. Set OPENAI_API_KEY or DUKE_AI_API_KEY to use INTERVIEW_COACH_SCORER=llm."
            )
    elif evaluator_mode in {"llm_structured", "v6"}:
        scoring_version = STRUCTURED_LLM_SCORING_VERSION
        if not is_llm_configured():
            raise ValueError(
                "No API key found. Set OPENAI_API_KEY or DUKE_AI_API_KEY to use INTERVIEW_COACH_SCORER=llm_structured."
            )
    else:
        scoring_version = HYBRID_SCORING_VERSION

    dataset = load_and_clean_dataset(DATASET_PATH)
    model = load_embedding_model() if evaluator_mode == "hybrid" else None
    used_question_numbers: set[int | float] = set()
    session_id = create_session_id()
    results: list[dict[str, object]] = []

    print("AI Interview Coach")
    print(f"Evaluator: {evaluator_mode}")

    while True:
        print()
        selected_question = select_random_question(
            dataset,
            used_question_numbers=used_question_numbers,
        )
        used_question_numbers.add(selected_question["Question Number"])

        print("Question:")
        print(selected_question["Question"])
        print()
        user_answer = input("Your answer: ").strip()

        if evaluator_mode == "llm":
            evaluation = evaluate_answer_with_llm(
                question=selected_question["Question"],
                reference_answer=selected_question["Answer"],
                user_answer=user_answer,
            )
        elif evaluator_mode in {"llm_structured", "v6"}:
            evaluation = evaluate_answer_with_structured_llm(
                question=selected_question["Question"],
                reference_answer=selected_question["Answer"],
                user_answer=user_answer,
            )
        else:
            evaluation = evaluate_answer(
                user_answer=user_answer,
                reference_answer=selected_question["Answer"],
                model=model,
            )

        results.append(
            {
                "session_id": session_id,
                "scoring_version": scoring_version,
                "llm_model": evaluation.get("llm_model", ""),
                "question_number": format_question_number(selected_question["Question Number"]),
                "category": selected_question["Category"],
                "difficulty": selected_question["Difficulty"],
                "question": selected_question["Question"],
                "input_mode": "typed",
                "transcription_text": "",
                "tts_used": False,
                "user_answer": user_answer,
                "reference_answer": selected_question["Answer"],
                "semantic_score": evaluation["semantic_score"],
                "coverage_score": evaluation["coverage_score"],
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

        print("\nResults")
        print(
            f"Question number: {format_question_number(selected_question['Question Number'])}"
        )
        print(f"Category: {selected_question['Category']}")
        print(f"Difficulty: {selected_question['Difficulty']}")

        if evaluator_mode == "llm":
            print(f"LLM model: {evaluation['llm_model']}")
            print(f"Correctness: {evaluation['correctness']}")
            print(f"Completeness: {evaluation['completeness']}")
            print(f"Clarity: {evaluation['clarity']}")
        elif evaluator_mode in {"llm_structured", "v6"}:
            print(f"LLM model: {evaluation['llm_model']}")
            print(f"Correctness subscore: {evaluation['correctness_subscore']}/2")
            print(f"Completeness subscore: {evaluation['completeness_subscore']}/2")
            print(f"Clarity subscore: {evaluation['clarity_subscore']}/2")
            print(f"Correctness: {evaluation['correctness']}")
            print(f"Completeness: {evaluation['completeness']}")
            print(f"Clarity: {evaluation['clarity']}")
        else:
            print(f"Semantic score: {evaluation['semantic_score']:.3f}")
            print(f"Keyword coverage: {evaluation['coverage_score']:.3f}")

        print(f"Final score: {evaluation['similarity_score']:.3f}")
        print(f"Rating: {evaluation['rating']}/5")
        print()
        print("Strengths:")
        print(evaluation["strengths"])
        print()
        print("Areas for improvement:")
        print(evaluation["improvement"])
        print()
        print("Reference answer:")
        print(selected_question["Answer"])

        if len(used_question_numbers) == len(dataset):
            print("\nYou have completed all available questions in the dataset.")
            break

        print()
        if not ask_to_continue():
            break

    average_similarity = sum(result["final_score"] for result in results) / len(results)
    average_rating = sum(result["rating"] for result in results) / len(results)
    output_file = save_session_results(
        results,
        output_path=get_versioned_results_path(scoring_version),
    )

    print("\nSession summary")
    print(f"Questions answered: {len(results)}")
    print(f"Average final score: {average_similarity:.3f}")
    print(f"Average rating: {average_rating:.2f}/5")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
