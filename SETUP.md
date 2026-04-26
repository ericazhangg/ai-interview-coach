# SETUP

## 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. Add API key if you want to use the LLM scorers

Create a `.env` file in the project root.

Use one of these:

```env
OPENAI_API_KEY=your_key_here
```

or

```env
DUKE_AI_API_KEY=your_key_here
```

Optional:

```env
OPENAI_MODEL=gpt-5-nano
OPENAI_TRANSCRIPTION_MODEL=gpt-4o-transcribe
```

## 3. Run the app

Public deployed app:

```text
https://sweinterviewcoach.streamlit.app/
```

If you want to run it locally:

For the web app:

```bash
python -m streamlit run src/streamlit_app.py
```

For the CLI version:

```bash
python src/main.py
```

## 4. How to test it

### If you do not have an API key

You can still test the project with the hybrid scorer.

1. Run:

```bash
python -m streamlit run src/streamlit_app.py
```

2. In the sidebar, choose:
   - `v4 Hybrid Baseline`
3. Click `New Question`
4. Type an answer
5. Submit it and review the score and feedback

### If you do have an API key

You can test the final model.

1. Add your API key to `.env`
2. Run:

```bash
python -m streamlit run src/streamlit_app.py
```

3. In the sidebar, choose:
   - `v6 Structured LLM Rubric`
4. Type an answer or use speech transcription
5. Submit it and review:
   - final score
   - rubric subscores
   - feedback

## 5. Extra evaluation scripts

Run the version benchmark:

```bash
python src/scripts/run_version_benchmark.py
python src/scripts/compare_results.py
```

Run the prompt comparison:

```bash
python src/scripts/run_prompt_benchmark.py
python src/scripts/compare_prompt_results.py
```

Run the baseline comparison:

```bash
python src/scripts/run_baseline_benchmark.py
python src/scripts/summarize_baseline_benchmark.py
```

## Notes

- The dataset is already included in the repo.
- Without an API key, the hybrid scorer still works.
- LLM scoring and speech transcription require an API key.
- Browser read-aloud works separately from the LLM scoring.
