# AI Interview Coach

AI Interview Coach is a software engineering interview practice tool that asks technical questions, accepts typed or spoken answers, and scores responses with multiple evaluation approaches. The final deployed system uses a structured LLM rubric to give concise feedback on correctness, completeness, and clarity.

## What it Does

This project helps users practice software engineering interview questions in a more realistic format. It presents a technical interview question, lets the user answer by text or speech, and then evaluates the answer using several versions of the scoring pipeline, including a final structured LLM rubric model. The system returns a rating, rubric-based feedback, and supporting explanation, and it also includes benchmark scripts and evaluation artifacts for comparing model versions and documenting improvements.

## Quick Start

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

2. If you want to use the LLM scorers, create a `.env` file in the project root with either:

```env
OPENAI_API_KEY=your_key_here
```

or

```env
DUKE_AI_API_KEY=your_key_here
```

3. Run the web app:

```bash
python -m streamlit run src/streamlit_app.py
```

Public deployed app:

```text
https://sweinterviewcoach.streamlit.app/
```

Local CLI version:

```bash
python src/main.py
```

More detailed setup instructions are in `SETUP.md`.

## Video Links

- Demo video: [link](https://drive.google.com/file/d/1a924iX1oK5plHfwvjjR57iP2ZbZozQOz/view?usp=sharing)
- Technical walkthrough: [link](https://drive.google.com/file/d/1bMcg9tiBfnU2DPhYZ3JC6xJx_qW974n4/view?usp=sharing)

## Evaluation

The project benchmark compares `v3`, `v4`, `v5`, and `v6` on the same fixed test set in `data/evaluation/test_cases.csv`. The final `v6` structured LLM rubric model achieved the best overall calibration with `MAE = 0.450`, `within ±1 accuracy = 1.000`, and `Pearson correlation = 0.938`, while `v5` remained slightly stronger on exact accuracy and Spearman correlation. Additional evaluation artifacts include the benchmark summary in `docs/benchmark_metrics_table.md`, prompt comparison results in `docs/prompt_comparison_table.md`, failure analysis in `docs/error_analysis.md`, and iteration history in `docs/model_iterations.md`.

## Individual Contributions

This was an individual project. All design decisions, testing, debugging, benchmarking, deployment, and final submission work were completed by me.
