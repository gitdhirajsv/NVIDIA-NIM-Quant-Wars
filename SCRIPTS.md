# Command Guide

Run commands from the repository root.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Generate reports

```bash
python platforms/nvidia/run_competition.py --parallel
python platforms/ollama/run_competition.py --parallel
python platforms/huggingface/run_competition.py --parallel
```

## Evaluate all platform folders

```bash
python evaluate_all.py
```

## Output folders

- `platforms/nvidia/generated_notebooks/`
- `platforms/ollama/generated_notebooks/`
- `platforms/huggingface/generated_notebooks/`

## Environment variables

- `NVIDIA_API_KEY`
- `HF_TOKEN`
- `CLOUD_KEY_1`
- `CLOUD_KEY_2`
- `CLOUD_KEY_3`
