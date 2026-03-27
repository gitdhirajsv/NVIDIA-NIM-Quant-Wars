# Jane Street Quant Wars

Multi-platform LLM benchmark for generating Jane Street forecasting pipelines across NVIDIA NIM, Ollama Cloud, and Hugging Face.

This project started from the NVIDIA NIM version and was later extended with separate Ollama and Hugging Face runners. The generated model reports are intentionally kept inside platform-specific folders so the repository stays easy to browse on GitHub.

## What is in this repo

- `platforms/nvidia/run_competition.py`: NVIDIA NIM model generation runner
- `platforms/ollama/run_competition.py`: Ollama Cloud model generation runner
- `platforms/huggingface/run_competition.py`: Hugging Face Inference model generation runner
- `platforms/*/generated_notebooks/`: generated notebook reports for each platform
- `evaluate_all.py`: unified evaluator that scans the platform report folders
- `leaderboard.csv` and `unified_leaderboard.csv`: evaluation snapshots
- `results_dashboard.html`, `unified_dashboard.html`, `RESULTS.md`: saved report artifacts

## Repository layout

```text
JaneStreet-Quant-Wars/
|-- README.md
|-- SCRIPTS.md
|-- requirements.txt
|-- .env.example
|-- evaluate_all.py
|-- leaderboard.csv
|-- unified_leaderboard.csv
|-- results_dashboard.html
|-- unified_dashboard.html
|-- RESULTS.md
`-- platforms/
    |-- nvidia/
    |   |-- run_competition.py
    |   `-- generated_notebooks/
    |-- ollama/
    |   |-- run_competition.py
    |   `-- generated_notebooks/
    `-- huggingface/
        |-- run_competition.py
        `-- generated_notebooks/
```

## Setup

Run everything from the repository root.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env` from `.env.example` and fill in the keys you want to use:

```env
NVIDIA_API_KEY=
HF_TOKEN=
CLOUD_KEY_1=
CLOUD_KEY_2=
CLOUD_KEY_3=
```

## Run each platform

NVIDIA NIM:

```bash
python platforms/nvidia/run_competition.py --parallel
```

Ollama Cloud:

```bash
python platforms/ollama/run_competition.py --parallel
```

Hugging Face:

```bash
python platforms/huggingface/run_competition.py --parallel
```

Each runner now writes its logs and generated notebooks back into its own platform folder.

## Evaluate results

```bash
python evaluate_all.py
```

The evaluator scans:

- `platforms/nvidia/generated_notebooks/`
- `platforms/ollama/generated_notebooks/`
- `platforms/huggingface/generated_notebooks/`

It writes fresh copies of:

- `leaderboard.csv`
- `unified_leaderboard.csv`

## Notes

- The platform report folders are kept on purpose.
- The old `.bat` wrappers were removed to keep the repo simpler and cross-platform.
- If Jane Street data is missing locally, the evaluator falls back to synthetic data so the pipeline still runs.

## Existing report artifacts

- `RESULTS.md`: narrative write-up of the benchmark
- `results_dashboard.html`: NVIDIA-focused HTML snapshot
- `unified_dashboard.html`: combined HTML snapshot
- `CLAUDE_EVALUATION.md`: saved summary notes
