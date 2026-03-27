# NVIDIA NIM Quant Wars Engine

An institutional-style quantitative research pipeline built as a personal project. Not a trading bot. Not a financial product. Just a passion for systematic research and multi-model AI evaluation.

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Evaluation%20Complete-brightgreen?style=flat-square)
![Models](https://img.shields.io/badge/Models-67%20LLMs-red?style=flat-square)
![ML](https://img.shields.io/badge/ML-XGBoost%20+%20Polars-blueviolet?style=flat-square)
![Data](https://img.shields.io/badge/Data-100K%20Rows-orange?style=flat-square)

</div>

---

## Overview
**NVIDIA NIM Quant Wars** is a specialized research infrastructure project designed to test the quantitative coding capabilities of 67 different Large Language Models (LLMs) against the Jane Street Real-Time Market Data Forecasting Kaggle challenge. 

At a high level, the engine queries the NVIDIA NIM API to prompt 60+ models to write predictive pipeline code. It then places the generated code into an automated, rigorous evaluation harness running against 100,000 rows of real Jane Street production data. The models are evaluated purely on out-of-sample Mean Squared Error (MSE) and R² targeting `responder_6`.

The project exposes a hard reality in quant finance: **model size does not equal signal quality**. A 7B parameter model produced a statistically significant signal (R² > 0) while 400B+ parameter models failed to beat the baseline.

Core capabilities:
- **Automated orchestration** of 67 LLMs via LangChain + NVIDIA NIM endpoints.
- **Strict standardized evaluation harness** wrapping generated XGBoost/Polars logic.
- **Standardized test set** using real Jane Street partitioned parquet files.
- **Fail-safe batching** (40 RPM limit, exponential backoff, auto-retries).
- **Interactive HTML dashboard** for institutional-grade visual reporting.

---

## Architecture
```text
                     
                              NIM QUANT WARS ENGINE          
                     
                                    
  API ORCHESTRATOR           CODE GENERATION             EVALUATION HARNESS   
                                                                      
 67 NIM Endpoints           Polars pipelines          100K row sample (JS)
 40 RPM Governor            XGBoost config parsing    Regex param extraction  
 LangChain Prompts          Automated .ipynb output   Strict 80/20 train/test
              
                                                                
                                         
                            SIGNAL RESULTS   
                                              
                           MSE / RMSE / R²    
                           Feature count check 
                           Interactive Dashboard   
                          
```

---

## Final Competition Results

| Rank | Model | Parameters | MSE (↓) | R² (↑) | Code Quality |
|---|---|---|---|---|---|
| 🥇 1 | `mistralai/mistral-7b-instruct-v0.3` | **7B** | **0.786110** | **+0.00235** | Clean, 15 lines |
| 🥈 2 | `yentinglin/llama-3-taiwan-70b-instruct` | 70B | 0.786624 | +0.00170 | verbose |
| 🥉 3 | `nvidia/nemotron-4-mini-hindi-4b-instruct` | 4B | 0.787319 | +0.00082 | Clean |

*Note: Only the top 3 models achieved a positive R² (performing better than predicting the mean).*

**[→ View Full Interactive Results Dashboard](./results_dashboard.html)**  
**[→ Detailed Analysis Report](./RESULTS.md)**  
**[→ Raw Leaderboard CSV](./leaderboard.csv)**  

---

## Primary Outputs
| File | Description |
|---|---|
| `results_dashboard.html` | Interactive frontend dashboard (Charts, Leaderboard, Top 3) |
| `evaluate_all.py` | The core standardized evaluation harness |
| `RESULTS.md` | Deep-dive analytical report of the findings |
| `leaderboard.csv` | Raw metric outputs for all 58 scored models |
| `generated_notebooks/` | Directory containing all 63 raw AI-generated Jupyter notebooks |

---

## How to Interpret Results
| Metric | Baseline | Good (Quant standard) | Winner achieved |
|---|---|---|---|
| **MSE** | 0.7880 | < 0.7870 | **0.7861** |
| **R²** | 0.0000 | > 0.0010 | **+0.0023** |

*In high-frequency/systematic quant finance, an R² of 0.002 (0.2%) is considered a highly tradable, institutional-quality signal. Across billions of dollars and thousands of trades, a 0.2% edge is mathematically sufficient to generate consistent Information Ratios.*

---

## Technical Specifications
| Parameter | Value |
|---|---|
| Total Models tested | 67 |
| Models scored successfully | 58 (86.6%) |
| Target Variable | `responder_6` (Jane Street anonymized 8-day return) |
| Data Split | 80% train / 20% test (Strict OOS) |
| Sample Size | 100,000 rows |
| Features | `feature_00` through `feature_78` (79 total) |
| Core Libraries | Polars (data), XGBoost (ML), LangChain (API) |
| Hyperparameters | Extracted dynamically via regex from model code |

---

## Research Principles
- **Strict evaluation harness** — No model was permitted to overfit or alter the evaluation metric. Same data, same test.
- **Transparency over mystique** — All 63 generated solutions are kept entirely unaltered in `generated_notebooks/`.
- **Evidence over claims** — The 405B parameter Meta model failed to crack the top 20, proving that for specific financial engineering tasks, focused instruct tuning beats raw parameter count.

---

## Running the Engine
**Option 1 — Full Local Pipeline (Windows)**
```cmd
git clone https://github.com/gitdhirajsv/NVIDIA-NIM-Quant-Wars.git
cd NVIDIA-NIM-Quant-Wars
setx NVIDIA_API_KEY "your_key_here"
start_battle.bat
```

**Option 2 — Evaluation Only (CLI)**
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python evaluate_all.py
```

---

## Troubleshooting
**API Key Error:** Verify `NVIDIA_API_KEY` is set in your environment variables.
**Data Not Found:** The underlying Jane Street data (`train.parquet`) is ~19GB and must be downloaded manually from Kaggle and placed into `jane_street_data/`. The script will fall back to synthetic data if the real `.parquet` files are missing.
**Rate Limiting:** If running `start_battle.bat`, you may encounter 504 timeouts from NVIDIA APIs. The script has an exponential backoff built-in, but overloaded models will simply fail and be logged in `leaderboard.csv` as `TIMEOUT`.

---

## Disclaimer
This is a research and educational project. Not affiliated with Jane Street, NVIDIA, or Kaggle. Not financial advice. Past performance does not indicate future results. Use at your own risk. Always do your own research.

---

<div align="center">

Built by [Dhiraj](https://github.com/gitdhirajsv) | Quant Research Lab

</div>
