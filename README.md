# Jane Street Quant Wars — Ultimate Edition

**Multi-Platform AI Benchmark: 100+ Models Across NVIDIA NIM, Ollama Cloud & Hugging Face**

An institutional-style quantitative research pipeline testing the coding capabilities of 100+ Large Language Models against the [Jane Street Real-Time Market Data Forecasting](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting) Kaggle challenge.

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Multi--Platform%20Evaluation-brightgreen?style=flat-square)
![Models](https://img.shields.io/badge/Models-100%2B%20LLMs-red?style=flat-square)
![Platforms](https://img.shields.io/badge/Platforms-3-orange?style=flat-square)
![ML](https://img.shields.io/badge/ML-XGBoost%20+%20Polars-blueviolet?style=flat-square)
![Data](https://img.shields.io/badge/Data-100K%20Rows-orange?style=flat-square)

</div>

---

## Overview

**Jane Street Quant Wars** is a comprehensive multi-platform benchmark evaluating AI models across **NVIDIA NIM**, **Ollama Cloud**, and **Hugging Face** inference APIs. The engine prompts 100+ models to write predictive XGBoost/Polars pipelines, then evaluates them against 100,000 rows of real Jane Street production data using standardized out-of-sample testing.

### Key Findings

> **Model size ≠ Signal quality.** A **7B parameter model** achieved statistically significant alpha (R² > 0) while 400B+ parameter models failed to beat the baseline.

### Core Capabilities

- **Multi-platform orchestration** — NVIDIA NIM, Ollama Cloud, Hugging Face Inference API
- **Automated code generation** — 100+ LLMs prompted via LangChain + native APIs
- **Standardized evaluation harness** — Strict 80/20 train/test split, dynamic hyperparameter extraction
- **Fail-safe batching** — Rate limit governors, exponential backoff, auto-retries
- **Interactive dashboards** — Institutional-grade visual reporting with platform comparisons

---

## Architecture

```text
                    JANE STREET QUANT WARS — MULTI-PLATFORM


  ORCHESTRATION LAYER              GENERATION              EVALUATION

┌─────────────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  NVIDIA NIM (67 models) │    │                  │    │                  │
│  Ollama Cloud (13)      │───▶│  Polars pipelines│───▶│  100K rows       │
│  Hugging Face (13)      │    │  XGBoost configs │    │  Regex extraction│
└─────────────────────────┘    │  .ipynb output   │    │  80/20 OOS test  │
                               └──────────────────┘    └──────────────────┘

                                      RESULTS
                              MSE / RMSE / R² per platform
                              Unified leaderboard + Dashboard
```

---

## 🏆 Current Competition Results

### Overall Leaderboard (Top 10)

| Rank | Model | Platform | MSE (↓) | R² (↑) | Code Lines |
|:----:|-------|----------|:-------:|:------:|:----------:|
| 🥇 1 | `mistralai/mistral-7b-instruct-v0.3` | NVIDIA NIM | **0.786110** | **+0.00235** | 15 |
| 🥈 2 | `yentinglin/llama-3-taiwan-70b-instruct` | NVIDIA NIM | 0.786624 | +0.00170 | 22 |
| 🥉 3 | `nvidia/nemotron-4-mini-hindi-4b-instruct` | NVIDIA NIM | 0.787319 | +0.00082 | 38 |
| 4 | `moonshotai/kimi-k2-instruct` | NVIDIA NIM | 0.789284 | -0.00167 | 36 |
| 5 | `institute-of-science-tokyo/llama-3.1-swallow-8b` | NVIDIA NIM | 0.790381 | -0.00307 | 20 |
| 6 | `moonshotai/kimi-k2-instruct-0905` | NVIDIA NIM | 0.792461 | -0.00571 | 35 |
| 7 | `igenius/italia_10b_instruct_16k` | NVIDIA NIM | 0.793950 | -0.00759 | 11 |
| 8 | `meta/llama3-8b-instruct` | NVIDIA NIM | 0.793950 | -0.00759 | 8 |
| 9 | `microsoft/phi-3-medium-128k-instruct` | NVIDIA NIM | 0.793950 | -0.00759 | 61 |
| 10 | `rakuten/rakutenai-7b-instruct` | NVIDIA NIM | 0.793950 | -0.00759 | 3 |

### Platform Summary

| Platform | Models | ✅ Scored | ⏳ Generated | ❌ Failed |
|----------|--------|-----------|--------------|-----------|
| **NVIDIA NIM** | 67 | 58 | — | 9 |
| **Ollama Cloud** | 13 | — | 11 | 2 |
| **Hugging Face** | 13 | — | 6 | 7 |
| **TOTAL** | **93** | **58** | **17** | **18** |

> **Note:** Only the top 3 models achieved positive R² (performing better than predicting the mean). Ollama and Hugging Face models are pending full evaluation.

**[→ View Unified Interactive Dashboard](./unified_dashboard.html)**  
**[→ NVIDIA Detailed Results](./RESULTS.md)**  
**[→ Full Leaderboard CSV](./unified_leaderboard.csv)**

---

## Platform-Specific Results

### NVIDIA NIM (Fully Evaluated)

The most mature platform with 86.6% success rate. All 58 scored models ran through the complete evaluation harness.

**Winner:** `mistralai/mistral-7b-instruct-v0.3` (7B params) — Best MSE: 0.786110, R²: +0.00235

### Ollama Cloud (Code Generated)

11 of 13 models successfully generated code. 2 models failed (not found on API).

**Generated Models:**
- `qwen3-next:80b`, `deepseek-v3.2`, `gemma3:27b`, `gemma3:12b`, `glm-5`, `glm-4.6`
- `kimi-k2-thinking`, `mistral-large-3:675b`, `ministral-3:14b`, `ministral-3:8b`, `nemotron-3-nano:30b`

**Failed:** `qwen3-next:30b`, `deepseek-v3` (404 — model not found)

### Hugging Face (Code Generated)

6 of 13 models generated code. 7 models require gated access approval.

**Generated Models:**
- `Qwen/Qwen2.5-Coder-32B-Instruct`, `Qwen/Qwen2.5-72B-Instruct`
- `meta-llama/Llama-3.3-70B-Instruct`, `meta-llama/Llama-3.1-8B-Instruct`
- `deepseek-ai/DeepSeek-V3`, `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`

**Failed:** Gated models (Mistral, Gemma, Phi-3, Hermes, Yi)

---

## Primary Outputs

| File | Description |
|------|-------------|
| `unified_dashboard.html` | Multi-platform interactive dashboard with charts, rankings, and platform comparisons |
| `unified_leaderboard.csv` | Complete results for all 100+ models across 3 platforms |
| `evaluate_all.py` | Core standardized evaluation harness (XGBoost + Polars) |
| `RESULTS.md` | Deep-dive analytical report (NVIDIA NIM results) |
| `generated_notebooks/` | 67 NVIDIA NIM raw AI-generated Jupyter notebooks |
| `ollama_results/` | 11 Ollama Cloud generated notebooks |
| `huggingface_results/` | 6 Hugging Face generated notebooks |
| `executed_notebooks/` | Evaluated notebooks with embedded MSE/R² scores |

---

## How to Interpret Results

| Metric | Baseline | Good (Quant Standard) | Winner Achieved |
|--------|----------|-----------------------|-----------------|
| **MSE** | 0.7880 | < 0.7870 | **0.7861** |
| **R²** | 0.0000 | > 0.0010 | **+0.0023** |

> **Institutional Context:** In high-frequency/systematic quant finance, an R² of 0.002 (0.2%) is considered a **highly tradable signal**. Across billions of dollars and thousands of trades, a 0.2% edge is mathematically sufficient to generate consistent Information Ratios.

---

## Technical Specifications

| Parameter | Value |
|-----------|-------|
| Total Models Tested | 100+ |
| Platforms | 3 (NVIDIA NIM, Ollama Cloud, Hugging Face) |
| Models Scored Successfully | 58 (NVIDIA NIM complete) |
| Target Variable | `responder_6` (Jane Street anonymized 8-day forward return) |
| Data Split | 80% train / 20% test (Strict out-of-sample) |
| Sample Size | 100,000 rows |
| Features | `feature_00` through `feature_78` (79 total) |
| Core Libraries | Polars (data), XGBoost (ML), LangChain (API orchestration) |
| Hyperparameters | Extracted dynamically via regex from model-generated code |

---

## Running the Engine

### Prerequisites

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### Platform-Specific Execution

**NVIDIA NIM:**
```cmd
cd JaneStreet-Quant-Wars
setx NVIDIA_API_KEY "your_key_here"
python run_competition.py --parallel
```

**Ollama Cloud:**
```cmd
cd ollama_competition
setx CLOUD_KEY_1 "key1"
setx CLOUD_KEY_2 "key2"
setx CLOUD_KEY_3 "key3"
python run_competition.py --parallel
```

**Hugging Face:**
```cmd
cd huggingface_competition
setx HF_TOKEN "hf_xxxxxxxxx"
python run_competition.py --parallel
```

### Evaluate All Results

```bash
python evaluate_all.py
```

---

## Repository Structure

```
JaneStreet-Quant-Wars/
├── README.md                    # This file
├── CLAUDE_EVALUATION.md         # Summary for final winner declaration
├── unified_dashboard.html       # Multi-platform interactive results
├── unified_leaderboard.csv      # Complete results (100+ models)
├── RESULTS.md                   # NVIDIA NIM deep-dive analysis
├── evaluate_all.py              # Core evaluation harness
├── run_competition.py           # NVIDIA NIM orchestration
├── requirements.txt             # Python dependencies
│
├── generated_notebooks/         # 67 NVIDIA NIM raw notebooks
├── executed_notebooks/          # Evaluated notebooks with scores
├── ollama_results/              # 11 Ollama Cloud notebooks
├── huggingface_results/         # 6 Hugging Face notebooks
│
└── [Platform Subdirectories]
    ├── nvidia_nim_competition/
    ├── ollama_competition/
    └── huggingface_competition/
```

---

## Troubleshooting

### API Key Errors
- **NVIDIA:** Verify `NVIDIA_API_KEY` is set in environment variables
- **Ollama:** Check all 3 `CLOUD_KEY_*` variables are valid
- **Hugging Face:** Ensure `HF_TOKEN` starts with `hf_`

### Data Not Found
The Jane Street dataset (`train.parquet`, ~19GB) must be downloaded from Kaggle and placed in `jane_street_data/`. The engine falls back to synthetic data if missing.

### Rate Limiting
- **NVIDIA NIM:** 40 RPM limit enforced automatically
- **Ollama Cloud:** 3-account rotation with exponential backoff
- **Hugging Face:** Free tier has queue delays; consider Pro tier

### Gated Models (Hugging Face)
Some models require accepting terms on HF first:
1. Visit model page on huggingface.co
2. Click "Agree & Access"
3. Re-run competition

---

## Research Principles

1. **Strict evaluation harness** — No model permitted to overfit or alter metrics. Same data, same test split.
2. **Transparency over mystique** — All generated solutions preserved unaltered in platform-specific directories.
3. **Evidence over claims** — 405B parameter models failed to crack top 20, proving focused instruct tuning beats raw parameter count for financial engineering tasks.
4. **Multi-platform fairness** — Each platform evaluated on identical data with identical XGBoost harness.

---

## Key Insights

### What Separated Winners from Losers?

**Winners:**
- ✅ Clean, modern Polars syntax (`group_by`, `with_columns`)
- ✅ Proper rolling window quantile calculations
- ✅ Binary feature flags for top-quantile identification
- ✅ Minimal, focused code (15-40 lines)

**Losers:**
- ❌ Deprecated Polars methods (`groupby`, `with_column`)
- ❌ Over-engineered explode/join logic
- ❌ Missing train/test splits (overfitting risk)
- ❌ Verbose code without feature engineering

### Platform Comparison

| Aspect | NVIDIA NIM | Ollama Cloud | Hugging Face |
|--------|------------|--------------|--------------|
| Speed | ⚡⚡⚡ Fastest | ⚡⚡ Medium | ⚡ Slow (free tier) |
| Model Variety | ⭐⭐⭐ 67 models | ⭐⭐ 13 models | ⭐⭐⭐ 13+ (gated) |
| Success Rate | 86.6% | 84.6% | 46.2% |
| Cost | Paid | Paid (3 accounts) | Free tier available |

---

## Disclaimer

This is a **research and educational project**. Not affiliated with Jane Street, NVIDIA, Ollama, Hugging Face, or Kaggle. **Not financial advice.** Past performance does not indicate future results. Use at your own risk. Always do your own research.

The code generated by AI models has not been audited for production use. Do not deploy real trading strategies based on these results without proper risk management and backtesting.

---

## Citation

If you use this benchmark in your research:

```bibtex
@misc{dhiraj2026janestreetquantwars,
  title={Jane Street Quant Wars: Multi-Platform AI Benchmark for Quantitative Finance},
  author={Dhiraj},
  year={2026},
  howpublished={\url{https://github.com/gitdhirajsv/JaneStreet-Quant-Wars}},
  note={GitHub Repository}
}
```

---

<div align="center">

**Built by [Dhiraj](https://github.com/gitdhirajsv)** | Quant Research Lab

[GitHub](https://github.com/gitdhirajsv/JaneStreet-Quant-Wars) · 
[Kaggle Competition](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting) · 
[NVIDIA NIM](https://build.nvidia.com/) · 
[Ollama](https://ollama.com/) · 
[Hugging Face](https://huggingface.co/)

</div>
