# NVIDIA NIM Quant Wars - Jane Street Forecasting

Battle Royale competition where 200+ NVIDIA NIM AI models compete to solve the Jane Street Market Forecasting challenge.

## Quick Start

```bash
# Run the battle (Windows)
start_battle.bat
```

This will:
1. Clean up old generated files
2. Create/check virtual environment
3. Install dependencies
4. Fetch models from NVIDIA API
5. Generate Python notebooks one-by-one (with rate limiting)

## Project Structure

```
NVIDIA-NIM-Quant-Wars/
├── start_battle.bat          # One-click launcher
├── run_competition.py        # Main orchestrator
├── battle_royale.py          # Alternative runner
├── download_data.py          # Jane Street data downloader
├── requirements.txt          # Python dependencies
└── jane_street_data/         # Data folder (download first)
```

## Prerequisites

- Python 3.10+
- NVIDIA API Key (set in `run_competition.py`)
- Kaggle account (for data download)

## Setup

```bash
# 1. Download Jane Street data (~19GB)
python download_data.py

# 2. Run the battle
start_battle.bat
```

## Configuration

Edit `run_competition.py`:
- `TEST_MODE = True` - Run only 2 models (testing)
- `TEST_MODE = False` - Run all 200+ models (full competition)
- `DELAY_SECONDS = 5` - Rate limit delay between models

## Generated Files

- `*.ipynb` - Python notebooks from each AI model (auto-generated)
- `competition_log_*.txt` - Execution logs

## License

MIT License - See LICENSE file
