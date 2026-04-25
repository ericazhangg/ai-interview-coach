"""Simple Streamlit UI for the AI Interview Coach."""

from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from interview_coach.audio import (
    is_audio_configured,
    transcribe_audio_bytes,
)
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
INTERVIEWER_IMAGE_PATH = Path("assets/interviewer_portrait.jpg")
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
    if "answer_text" not in st.session_state:
        st.session_state.answer_text = ""
    if "input_mode" not in st.session_state:
        st.session_state.input_mode = "typed"
    if "transcription_text" not in st.session_state:
        st.session_state.transcription_text = ""
    if "tts_used" not in st.session_state:
        st.session_state.tts_used = False
    if "auto_play_question_audio" not in st.session_state:
        st.session_state.auto_play_question_audio = True


def choose_new_question(dataset) -> None:
    """Pick and store a new random question for the current UI session."""
    selected_question = select_random_question(
        dataset,
        used_question_numbers=st.session_state.used_question_numbers,
    )
    st.session_state.used_question_numbers.add(selected_question["Question Number"])
    st.session_state.current_question = selected_question
    st.session_state.last_result = None
    st.session_state.answer_text = ""
    st.session_state.input_mode = "typed"
    st.session_state.transcription_text = ""
    st.session_state.tts_used = False


def get_interviewer_image_uri() -> str:
    """Return the interviewer portrait as an embeddable data URI."""
    if not INTERVIEWER_IMAGE_PATH.exists():
        return ""

    mime_type, _ = mimetypes.guess_type(INTERVIEWER_IMAGE_PATH.name)
    encoded_bytes = base64.b64encode(INTERVIEWER_IMAGE_PATH.read_bytes()).decode("utf-8")
    return f"data:{mime_type or 'image/jpeg'};base64,{encoded_bytes}"


def apply_app_styles() -> None:
    """Inject a lightweight interview-call theme for the Streamlit app."""
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(245, 158, 11, 0.18), transparent 30%),
                radial-gradient(circle at top right, rgba(14, 165, 233, 0.18), transparent 28%),
                linear-gradient(180deg, #f6f3ed 0%, #f2efe8 100%);
        }
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1180px;
        }
        .coach-hero {
            background: linear-gradient(135deg, #111827 0%, #1f2937 70%, #374151 100%);
            color: #f9fafb;
            border-radius: 22px;
            padding: 1.3rem 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 18px 50px rgba(17, 24, 39, 0.14);
        }
        .coach-eyebrow {
            display: inline-block;
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #fbbf24;
            margin-bottom: 0.45rem;
        }
        .coach-card {
            background: transparent;
            border: none;
            border-radius: 20px;
            padding: 0;
            box-shadow: none;
        }
        .coach-question-panel {
            background: #fffaf2;
            border: 1px solid #eadfce;
            border-radius: 18px;
            padding: 1rem 1.1rem;
            margin-bottom: 0.7rem;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.4);
        }
        .coach-label {
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #fbbf24;
            font-weight: 700;
        }
        .coach-portrait {
            width: 100%;
            height: auto;
            max-height: 520px;
            object-fit: contain;
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            background: transparent;
            display: block;
        }
        .coach-question {
            font-size: 1.16rem;
            line-height: 1.7;
            font-weight: 600;
            color: #111827;
            margin-top: 0.45rem;
        }
        .coach-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 0.85rem;
        }
        .coach-chip {
            padding: 0.32rem 0.7rem;
            border-radius: 999px;
            background: #f3eadb;
            border: 1px solid #e2d2b7;
            color: #1f2937;
            font-size: 0.82rem;
            font-weight: 600;
        }
        .result-shell {
            background: #fffdf9;
            border: 1px solid #eadfce;
            border-radius: 22px;
            padding: 1.15rem 1.2rem;
            box-shadow: 0 12px 30px rgba(120, 113, 108, 0.08);
            margin-top: 0.8rem;
        }
        .result-title {
            font-size: 1.15rem;
            font-weight: 700;
            color: #1f2937;
        }
        .result-subtitle {
            font-size: 0.92rem;
            color: #6b7280;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_browser_speech(
    text: str,
    button_label: str,
    element_id: str,
    *,
    autoplay: bool = False,
    caption: str | None = None,
) -> None:
    """Use the browser's speech engine for lightweight playback."""
    speech_text = json.dumps(text)
    autoplay_flag = "true" if autoplay else "false"
    components.html(
        f"""
        <div style="margin: 0.15rem 0 0.25rem 0; display:flex; gap:0.5rem; flex-wrap:wrap;">
            <button id="{element_id}" style="
                background: #f5f7fb;
                color: #111827;
                border: 1px solid #d1d5db;
                border-radius: 0.5rem;
                padding: 0.5rem 0.9rem;
                cursor: pointer;
                font-size: 0.95rem;
            ">
                {button_label}
            </button>
            <button id="{element_id}-pause" style="
                background: #fff7ed;
                color: #9a3412;
                border: 1px solid #fdba74;
                border-radius: 0.5rem;
                padding: 0.5rem 0.9rem;
                cursor: pointer;
                font-size: 0.95rem;
            ">
                Pause
            </button>
            <button id="{element_id}-resume" style="
                background: #eff6ff;
                color: #1d4ed8;
                border: 1px solid #93c5fd;
                border-radius: 0.5rem;
                padding: 0.5rem 0.9rem;
                cursor: pointer;
                font-size: 0.95rem;
            ">
                Resume
            </button>
            <button id="{element_id}-stop" style="
                background: #f8fafc;
                color: #334155;
                border: 1px solid #cbd5e1;
                border-radius: 0.5rem;
                padding: 0.5rem 0.9rem;
                cursor: pointer;
                font-size: 0.95rem;
            ">
                Stop
            </button>
        </div>
        <script>
        const text = {speech_text};
        let utterance = null;

        function speakQuestion() {{
            window.speechSynthesis.cancel();
            utterance = new SpeechSynthesisUtterance(text);
            window.speechSynthesis.speak(utterance);
        }}

        function pauseQuestion() {{
            if (window.speechSynthesis.speaking && !window.speechSynthesis.paused) {{
                window.speechSynthesis.pause();
            }}
        }}

        function resumeQuestion() {{
            if (window.speechSynthesis.paused) {{
                window.speechSynthesis.resume();
            }}
        }}

        function stopQuestion() {{
            window.speechSynthesis.cancel();
        }}

        const button = document.getElementById("{element_id}");
        const pauseButton = document.getElementById("{element_id}-pause");
        const resumeButton = document.getElementById("{element_id}-resume");
        const stopButton = document.getElementById("{element_id}-stop");
        if (button) {{
            button.addEventListener("click", speakQuestion);
        }}
        if (pauseButton) {{
            pauseButton.addEventListener("click", pauseQuestion);
        }}
        if (resumeButton) {{
            resumeButton.addEventListener("click", resumeQuestion);
        }}
        if (stopButton) {{
            stopButton.addEventListener("click", stopQuestion);
        }}

        if ({autoplay_flag}) {{
            window.setTimeout(speakQuestion, 250);
        }}
        </script>
        """,
        height=58,
    )
    if caption:
        st.caption(caption)


def render_hero() -> None:
    """Render the app header."""
    st.markdown(
        """
        <div class="coach-hero">
            <div class="coach-eyebrow">AI Interview Coach</div>
            <div style="font-size: 2rem; font-weight: 800; line-height: 1.1;">SWE Technical Interview Practice</div>
            <div style="margin-top: 0.55rem; max-width: 760px; color: #d1d5db; font-size: 1rem;">
                Hear the question, answer by voice or text, and get concise interview feedback in one focused practice space.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_interview_stage(question_row, audio_enabled: bool) -> None:
    """Render the main call-style interview layout."""
    question_number = format_question_number(question_row["Question Number"])
    st.markdown(
        f"""
        <div class="coach-question-panel">
            <div style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; color: #9a3412; font-weight: 700;">Interview question</div>
            <div class="coach-question">{question_row["Question"]}</div>
            <div class="coach-meta">
                <span class="coach-chip">Question #{question_number}</span>
                <span class="coach-chip">{question_row["Category"]}</span>
                <span class="coach-chip">{question_row["Difficulty"]}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    left_column, right_column = st.columns([1.3, 0.95], gap="large")

    with left_column:
        interviewer_image_uri = get_interviewer_image_uri()
        interviewer_image_html = (
            f'<img src="{interviewer_image_uri}" alt="Interviewer portrait" class="coach-portrait" />'
            if interviewer_image_uri
            else '<div class="coach-portrait" style="display:flex;align-items:center;justify-content:center;color:#94a3b8;">Interviewer</div>'
        )
        st.markdown(
            f"""
            <div class="coach-card">
                <div>{interviewer_image_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if audio_enabled:
            render_browser_speech(
                question_row["Question"],
                button_label="Read Question Aloud",
                element_id="question-speech-button",
                autoplay=st.session_state.auto_play_question_audio,
            )

    with right_column:
        render_camera_preview()


def render_camera_preview() -> None:
    """Show a mirrored live camera preview without capture controls."""
    components.html(
        """
        <div style="border-radius:18px; overflow:hidden; background:#020617; border:1px solid rgba(255,255,255,0.08);">
            <video id="candidate-preview" autoplay playsinline muted
                style="width:100%; height:320px; object-fit:cover; transform:scaleX(-1); background:#020617;"></video>
            <div id="camera-status" style="padding:0.7rem 0.9rem; color:#cbd5e1; font-size:0.9rem; background:#111827;">
                Requesting camera access...
            </div>
        </div>
        <script>
        const video = document.getElementById("candidate-preview");
        const status = document.getElementById("camera-status");

        async function startPreview() {
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                status.innerText = "Camera preview is not supported in this browser.";
                return;
            }
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
                video.srcObject = stream;
                status.innerText = "Live mirrored preview. Nothing is being recorded or saved.";
            } catch (error) {
                status.innerText = "Camera access is blocked or unavailable. You can still continue the interview without it.";
            }
        }

        startPreview();
        </script>
        """,
        height=380,
    )


def split_feedback_points(text: str) -> list[str]:
    """Convert stored feedback text into a short list."""
    if not text:
        return []
    return [item.strip() for item in text.split("|") if item.strip()]


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
        "input_mode": st.session_state.input_mode,
        "transcription_text": st.session_state.transcription_text,
        "tts_used": st.session_state.tts_used,
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


def render_result(
    evaluation: dict,
    scoring_version: str,
    saved_path: Path,
    audio_enabled: bool,
) -> None:
    """Display the evaluation output clearly in the UI."""
    question_row = st.session_state.current_question

    st.markdown(
        """
        <div class="result-shell">
            <div class="result-title">Evaluation Summary</div>
            <div class="result-subtitle">A snapshot of how the answer performed.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
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
            "Correctness", f"{evaluation.get('correctness_subscore', '')}/2"
        )
        subscore_columns[1].metric(
            "Completeness", f"{evaluation.get('completeness_subscore', '')}/2"
        )
        subscore_columns[2].metric(
            "Clarity", f"{evaluation.get('clarity_subscore', '')}/2"
        )

    strengths = split_feedback_points(evaluation["strengths"])
    improvements = split_feedback_points(evaluation["improvement"])
    feedback_columns = st.columns(2, gap="large")

    with feedback_columns[0]:
        st.markdown("**What went well**")
        if strengths:
            for point in strengths[:3]:
                st.markdown(f"- {point}")
        else:
            st.write("No strengths were returned for this response.")

    with feedback_columns[1]:
        st.markdown("**Tightest improvement areas**")
        if improvements:
            for point in improvements[:3]:
                st.markdown(f"- {point}")
        else:
            st.write("No improvement areas were returned for this response.")

    if audio_enabled:
        feedback_text = (
            f"Rating: {evaluation['rating']} out of 5. "
            f"Strengths: {evaluation['strengths']}. "
            f"Areas for improvement: {evaluation['improvement']}."
        )
        render_browser_speech(
            feedback_text,
            button_label="Read Feedback Aloud",
            element_id="feedback-speech-button",
            caption="Feedback audio uses browser speech with your current key setup.",
        )

    detail_tabs = st.tabs(["Breakdown", "Reference Answer"])
    with detail_tabs[0]:
        if evaluation.get("correctness"):
            st.markdown("**Correctness**")
            st.write(evaluation["correctness"])
        if evaluation.get("completeness"):
            st.markdown("**Completeness**")
            st.write(evaluation["completeness"])
        if evaluation.get("clarity"):
            st.markdown("**Clarity**")
            st.write(evaluation["clarity"])
    with detail_tabs[1]:
        st.write(question_row["Answer"])

    st.caption(f"Answer source: {st.session_state.input_mode}")
    st.caption(f"Saved to {saved_path}")


def main() -> None:
    """Run the Streamlit interview coach interface."""
    st.set_page_config(page_title="AI Interview Coach", page_icon="💬", layout="wide")

    dataset = get_dataset()
    ensure_session_state()
    apply_app_styles()
    render_hero()

    with st.sidebar:
        st.markdown("### Session Controls")
        scorer_label = st.selectbox("Scoring model", list(SCORER_OPTIONS.keys()), index=2)
        evaluator_mode, scoring_version = SCORER_OPTIONS[scorer_label]
        audio_enabled = is_audio_configured()

        if st.button("New Question", type="primary", use_container_width=True) or st.session_state.current_question is None:
            choose_new_question(dataset)
        if st.button("Reset Session", use_container_width=True):
            st.session_state.session_id = create_session_id()
            st.session_state.used_question_numbers = set()
            choose_new_question(dataset)

        if audio_enabled:
            st.checkbox(
                "Auto-read question",
                key="auto_play_question_audio",
                help="Your browser may still require one click before speech starts automatically.",
            )
        else:
            st.info("Voice features are off because no API key is configured.")

        st.caption(f"Current session: {st.session_state.session_id}")
        st.caption(f"Scoring version: {scoring_version}")

    if evaluator_mode != "hybrid" and not is_llm_configured():
        st.warning(
            "No API key detected. Add OPENAI_API_KEY or DUKE_AI_API_KEY in your .env file to use the LLM scorers."
        )

    question_row = st.session_state.current_question
    if question_row is None:
        st.stop()

    render_interview_stage(question_row, audio_enabled)

    with st.expander("Optional voice answer", expanded=False):
        if hasattr(st, "audio_input"):
            st.write(
                "Record a spoken response to transcribe it into text. You can edit the transcription before scoring."
            )
            recorded_audio = st.audio_input("Record your answer")
        else:
            recorded_audio = None
            st.info("Audio recording is not available in this Streamlit version. You can still type your answer below.")

        if recorded_audio is not None:
            st.audio(recorded_audio)
            if st.button("Transcribe Audio", disabled=not audio_enabled):
                try:
                    audio_bytes = recorded_audio.getvalue()
                    transcription_text = transcribe_audio_bytes(
                        audio_bytes=audio_bytes,
                        file_name=getattr(recorded_audio, "name", None),
                        mime_type=getattr(recorded_audio, "type", None),
                    )
                    st.session_state.answer_text = transcription_text
                    st.session_state.transcription_text = transcription_text
                    st.session_state.input_mode = "speech_to_text"
                    st.success("Audio transcribed. Review and edit the text below before scoring.")
                except Exception as error:
                    st.session_state.transcription_text = ""
                    st.session_state.input_mode = "typed"
                    st.warning(
                        f"Audio transcription failed: {error}. You can still type your answer manually."
                    )

    if st.session_state.input_mode == "speech_to_text":
        st.caption("Current answer source: transcribed speech. You can edit it below before scoring.")
    else:
        st.caption("Current answer source: typed text")

    with st.form("answer_form"):
        user_answer = st.text_area("Your answer", key="answer_text", height=180)
        submitted = st.form_submit_button("Evaluate Answer")

    if submitted:
        cleaned_user_answer = st.session_state.answer_text.strip()
        if not cleaned_user_answer:
            st.error("Please enter an answer before evaluating.")
        elif evaluator_mode != "hybrid" and not is_llm_configured():
            st.error("LLM scoring is not configured. Add an API key in .env first.")
        else:
            manual_answer = user_answer.strip()
            if manual_answer != st.session_state.transcription_text.strip():
                st.session_state.input_mode = "typed"
                st.session_state.transcription_text = ""
            with st.spinner("Evaluating answer..."):
                evaluation = evaluate_current_answer(cleaned_user_answer, evaluator_mode)
                saved_path = save_ui_result(evaluation, scoring_version, cleaned_user_answer)
                st.session_state.last_result = (evaluation, scoring_version, saved_path)

    if st.session_state.last_result is not None:
        evaluation, last_scoring_version, saved_path = st.session_state.last_result
        render_result(evaluation, last_scoring_version, saved_path, audio_enabled)


if __name__ == "__main__":
    main()
