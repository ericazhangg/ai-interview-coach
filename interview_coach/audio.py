"""Audio helpers for optional Streamlit voice features."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import BinaryIO

from openai import OpenAI

from interview_coach.llm_evaluator import get_api_key, get_base_url


DEFAULT_TTS_MODEL = "gpt-4o-mini-tts"
DEFAULT_TTS_VOICE = "marin"
DEFAULT_TRANSCRIPTION_MODEL = "gpt-4o-transcribe"
SUPPORTED_AUDIO_SUFFIXES = {
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/mp4": ".mp4",
    "audio/m4a": ".m4a",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/webm": ".webm",
}


def is_audio_configured() -> bool:
    """Return True when audio features can authenticate successfully."""
    return bool(get_api_key())


def get_audio_client() -> OpenAI:
    """Create an OpenAI client for speech features."""
    return OpenAI(
        api_key=get_api_key(),
        base_url=get_base_url(),
    )


def get_tts_model_name() -> str:
    """Return the configured text-to-speech model name."""
    return os.getenv("OPENAI_TTS_MODEL", DEFAULT_TTS_MODEL)


def get_tts_voice_name() -> str:
    """Return the configured TTS voice name."""
    return os.getenv("OPENAI_TTS_VOICE", DEFAULT_TTS_VOICE)


def get_transcription_model_name() -> str:
    """Return the configured speech-to-text model name."""
    return os.getenv("OPENAI_TRANSCRIPTION_MODEL", DEFAULT_TRANSCRIPTION_MODEL)


def synthesize_speech(text: str) -> bytes:
    """Generate spoken audio bytes that Streamlit can play back."""
    if not is_audio_configured():
        raise ValueError("Audio API is not configured.")
    if not text.strip():
        raise ValueError("No text was provided for speech synthesis.")

    response = get_audio_client().audio.speech.create(
        model=get_tts_model_name(),
        voice=get_tts_voice_name(),
        input=text,
        response_format="mp3",
    )

    if hasattr(response, "read"):
        return response.read()
    if hasattr(response, "content"):
        return response.content
    raise ValueError("The speech API response did not include readable audio bytes.")


def synthesize_question_audio(text: str) -> bytes:
    """Backward-compatible wrapper for question audio generation."""
    return synthesize_speech(text)


def guess_audio_suffix(file_name: str | None, mime_type: str | None) -> str:
    """Choose a file suffix for a temporary audio upload."""
    if file_name:
        suffix = Path(file_name).suffix.lower()
        if suffix:
            return suffix

    if mime_type and mime_type in SUPPORTED_AUDIO_SUFFIXES:
        return SUPPORTED_AUDIO_SUFFIXES[mime_type]

    return ".wav"


def transcribe_audio_file(audio_file: BinaryIO, file_name: str | None = None) -> str:
    """Transcribe an audio file object into text."""
    if not is_audio_configured():
        raise ValueError("Audio API is not configured.")

    response = get_audio_client().audio.transcriptions.create(
        model=get_transcription_model_name(),
        file=(file_name or "answer.wav", audio_file),
    )

    if isinstance(response, str):
        return response.strip()
    if hasattr(response, "text"):
        return str(response.text).strip()
    raise ValueError("The transcription API response did not include text output.")


def transcribe_audio_bytes(
    audio_bytes: bytes,
    file_name: str | None = None,
    mime_type: str | None = None,
) -> str:
    """Transcribe uploaded audio bytes into text."""
    if not is_audio_configured():
        raise ValueError("Audio API is not configured.")
    if not audio_bytes:
        raise ValueError("No audio data was provided for transcription.")

    suffix = guess_audio_suffix(file_name, mime_type)
    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
            temporary_file.write(audio_bytes)
            temporary_path = Path(temporary_file.name)

        with temporary_path.open("rb") as audio_file:
            inferred_name = file_name or temporary_path.name
            return transcribe_audio_file(audio_file=audio_file, file_name=inferred_name)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
