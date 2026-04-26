# ATTRIBUTION

I used AI tools a lot while building this project. Most of the larger code sections, refactors, UI changes, scripts, and documentation drafts were AI-generated first and then I reviewed, edited, debugged, reorganized, and tested them myself.

I mainly wrote the simpler parts myself when it was faster or easier to just do them directly.

## AI use

AI was used for:

- code generation
- refactoring
- debugging help
- documentation drafts
- UI changes
- benchmark and evaluation scripts

I was still responsible for:

- choosing the final design
- deciding which versions to keep
- running benchmarks
- checking results
- fixing bugs and path issues
- making sure the final submission was consistent

## Main files with AI-generated code

- `src/interview_coach/data.py`
- `src/interview_coach/coach.py`
- `src/interview_coach/llm_evaluator.py`
- `src/interview_coach/audio.py`
- `src/interview_coach/results.py`
- `src/main.py`
- `src/streamlit_app.py`

## Libraries used

- `pandas`
- `numpy`
- `scikit-learn`
- `sentence-transformers`
- `openai`
- `pydantic`
- `python-dotenv`
- `streamlit`

## Models and services used

- sentence embedding model: `all-MiniLM-L6-v2`
- OpenAI-compatible LLM API for `v5` and `v6`
- OpenAI-compatible transcription API for speech-to-text
- browser speech synthesis for read-aloud in the Streamlit app

## Dataset used

- `data/raw/Software_Questions.csv`

## Final note

Even when AI generated code or writing, I checked it, changed it, and tested it myself.
