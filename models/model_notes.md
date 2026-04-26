# Model Notes

This project does not train a custom model locally.

Instead, it uses a mix of:

- a pretrained sentence-transformer for embedding-based baseline scoring
- OpenAI-compatible LLMs for rubric-based evaluation
- OpenAI-compatible speech-to-text for spoken answer transcription
- browser speech synthesis for text-to-speech fallback in the final UI

This folder documents the models used by the system and their roles in the pipeline.
