# NVIDIA NIM QUANT WARS

> **Multi-Model AI Battle Royale for Jane Street Real-Time Market Data Forecasting**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Models](https://img.shields.io/badge/Models-67%20Competed-blue)]()
[![Winner](https://img.shields.io/badge/Winner-Mistral%207B%20v0.3-gold)]()
[![Best MSE](https://img.shields.io/badge/Best%20MSE-0.7861-brightgreen)]()

---

## 🏆 WINNER: `mistralai/mistral-7b-instruct-v0.3`

| Metric | Value |
|--------|-------|
| 🥇 **Best MSE** | **0.786110** |
| **RMSE** | 0.886628 |
| **R²** | +0.002356 *(only top 3 beat the baseline)* |
| **Features Used** | 80 |
| **Code Lines Written** | 15 |

> A 7B model beat every 70B+ model in the arena. Clean feature engineering wins over raw scale.

📊 **[→ View Full Interactive Results Dashboard](./results_dashboard.html)**  
📋 **[→ Full Leaderboard CSV](./leaderboard.csv)**  
📝 **[→ Detailed Analysis Report](./RESULTS.md)**

---

## TOP 5 LEADERBOARD

| Rank | Model | MSE | R² |
|------|-------|-----|----|
| 🥇 | mistralai/mistral-7b-instruct-v0.3 | **0.786110** | +0.002356 |
| 🥈 | yentinglin/llama-3-taiwan-70b-instruct | 0.786624 | +0.001704 |
| 🥉 | nvidia/nemotron-4-mini-hindi-4b-instruct | 0.787319 | +0.000821 |
| 4 | moonshotai/kimi-k2-instruct | 0.789284 | -0.001673 |
| 5 | institute-of-science-tokyo/llama-3.1-swallow-8b-instruct-v0.1 | 0.790381 | -0.003065 |

> Evaluated on 100,000 rows of real Jane Street production data · All 58 scored notebooks included

---

## OVERVIEW

**NVIDIA NIM Quant Wars** pits 67 NVIDIA NIM LLMs against each other on the [Jane Street Real-Time Market Data Forecasting](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting) Kaggle challenge.

Each model generates Python code using Polars and XGBoost to predict `responder_6` from 79 anonymized financial features. The code is evaluated on real Jane Street partitioned parquet data.

**Research Question:** *Which LLM writes the best quantitative finance code?*

---

## FEATURES

- **67 NVIDIA NIM Models** competed — 58 scored, 63 generated notebooks
- **Automated Code Generation** using LangChain + NVIDIA NIM API
- **Real Data Evaluation** on Jane Street partitioned parquet (100K rows)
- **Interactive Dashboard** — `results_dashboard.html` with charts and full leaderboard
- **Full Transparency** — all 63 generated `.ipynb` notebooks included
- **Rate-Limit Safe** — 40 RPM compliant, exponential backoff, auto-retry
- **Health Check System** — pre-test models before competition

---

## COMPETITION STATS

| Metric | Count |
|--------|-------|
| Models entered | 67 |
| Notebooks generated | 63 (93.9%) |
| Models scored | 58 (86.6%) |
| 504 timeouts | 4 |
| Empty notebooks | 5 |
| Models with R² > 0 | **3** |
| Best MSE | **0.786110** |

---

## QUICK START

```bash
# 1. Clone
git clone https://github.com/gitdhirajsv/NVIDIA-NIM-Quant-Wars.git
cd NVIDIA-NIM-Quant-Wars

# 2. Set API Key
setx NVIDIA_API_KEY "nvapi-..."

# 3. Run the Battle (generates all notebooks)
start_battle.bat

# 4. Evaluate all models (generates leaderboard.csv)
venv\Scripts\python evaluate_all.py
```

Or open `results_dashboard.html` to see the pre-run results.

---

## FILE STRUCTURE

```
NVIDIA-NIM-Quant-Wars/
├── start_battle.bat              # One-click: setup + run competition
├── run_competition.py            # Main orchestrator (NVIDIA NIM → notebooks)
├── evaluate_all.py               # Evaluation engine → leaderboard.csv
├── battle_royale.py              # Standalone benchmark runner
├── health_check.py               # Pre-test model availability
├── download_data.py              # Jane Street data downloader (~19GB)
├── requirements.txt              # All Python dependencies
├── nvidia_nim_models.csv         # 200+ model endpoints reference
│
├── results_dashboard.html        # ⭐ Interactive results dashboard
├── leaderboard.csv               # ⭐ Full ranked results (58 models)
├── RESULTS.md                    # ⭐ Detailed analysis report
│
├── jane_street_xgboost_model.json# Pre-trained XGBoost baseline model
├── jane_street_data/             # Competition dataset (download separately)
├── executed_notebooks/           # Reference execution examples
└── *.ipynb                       # 63 AI-generated solution notebooks
```

---

## CONFIGURATION

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TOP_QUANTILE` | 15% | Feature engineering threshold |
| `RATE_LIMIT` | 40 RPM | API compliance |
| `DELAY` | 5s | Between model calls |
| `TARGET` | responder_6 | Primary prediction variable |
| `SAMPLE_ROWS` | 100,000 | Rows for evaluation (set None for full) |
| `MAX_RETRIES` | 3 | Retry attempts for timeouts |

---

## KEY FINDING: Small Beats Large

`mistral-7b-instruct-v0.3` (7B params) outperformed `meta/llama-3.1-405b-instruct` (405B params) — **57× smaller model wins**. In quantitative finance, feature engineering quality matters more than raw model size.

---

## DEPENDENCIES

```
langchain >= 0.1.0
langchain-nvidia-ai-endpoints >= 0.1.0
polars >= 1.0.0
xgboost >= 2.0.0
scikit-learn >= 1.0.0
nbformat >= 5.0.0
pyarrow >= 12.0.0
pandas >= 2.0.0
numpy >= 1.24.0
```

---

## ATTRIBUTION

- [LangChain](https://github.com/langchain-ai/langchain) — LLM orchestration
- [NVIDIA NIM](https://build.nvidia.com/) — Model inference API
- [Polars](https://pola.rs/) — Fast DataFrame library
- [XGBoost](https://xgboost.readthedocs.io/) — Gradient boosting
- [Jane Street Competition](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting) — Kaggle

---

## DISCLAIMER

This project is for **educational and research purposes only**. Not affiliated with Jane Street, NVIDIA, or Kaggle. Not financial advice. Generated code should be reviewed before execution.

---

## AUTHOR

**Dhiraj** · [GitHub](https://github.com/gitdhirajsv)

*Built with passion for systematic research and AI-driven quantitative analysis.*
