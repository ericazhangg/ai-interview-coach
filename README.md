# AI Interview Coach

This repository is organized around a `src`/`data` project structure for the AI Interview Coach final project.

## Structure

- `src/`
  - application source code, Streamlit UI, CLI entrypoint, and evaluation scripts
- `data/`
  - raw dataset, benchmark test cases, and generated result CSV files
- `models/`
  - model configs or loading artifacts when needed
- `notebooks/`
  - Jupyter notebooks for exploration and analysis
- `videos/`
  - demo video and technical walkthrough video
- `docs/`
  - report-ready charts, tables, and supplementary documentation
- `requirements.txt`
  - Python dependencies

## Main Entry Points

- CLI: `python src/main.py`
- Streamlit UI: `python -m streamlit run src/streamlit_app.py`
- Benchmark runner: `python src/scripts/run_version_benchmark.py`
- Benchmark comparison: `python src/scripts/compare_results.py`

## Important Paths

- Dataset: `data/raw/Software_Questions.csv`
- Benchmark cases: `data/evaluation/test_cases.csv`
- Generated results: `data/results/`
- Report assets: `docs/`
