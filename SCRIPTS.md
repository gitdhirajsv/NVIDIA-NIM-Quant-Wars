# 📦 Scripts & Usage Guide

Quick reference for running Jane Street Quant Wars competitions.

---

## 🚀 Quick Start (Recommended)

**Run Everything with One Click:**
```cmd
RUN_ALL.bat
```

This executes the complete pipeline:
1. Installs dependencies
2. Runs all 3 platform competitions
3. Evaluates results and generates dashboard

**Estimated time:** 30-60 minutes

---

## 📋 Step-by-Step Execution

### Option 1: Guided Setup

| Step | Script | Description |
|------|--------|-------------|
| 1 | `1_setup.bat` | Install Python dependencies |
| 2 | `2_run_all_competitions.bat` | Run all 3 platforms |
| 3 | `3_evaluate.bat` | Evaluate & generate dashboard |

### Option 2: Platform-SSpecific

| Platform | Script | Models | API Key Required |
|----------|--------|--------|------------------|
| **NVIDIA NIM** | `run_nvidia.bat` | 67 models | `NVIDIA_API_KEY` |
| **Ollama Cloud** | `run_ollama.bat` | 13 models | `CLOUD_KEY_1/2/3` |
| **Hugging Face** | `run_huggingface.bat` | 13 models | `HF_TOKEN` |

---

## 🔑 Setting API Keys

### Windows (Permanent)
```cmd
setx NVIDIA_API_KEY "your-nvidia-key"
setx CLOUD_KEY_1 "your-ollama-key-1"
setx CLOUD_KEY_2 "your-ollama-key-2"
setx CLOUD_KEY_3 "your-ollama-key-3"
setx HF_TOKEN "hf_xxxxxxxxxxxxx"
```

### Windows (Temporary - Current Session Only)
```cmd
set NVIDIA_API_KEY=your-nvidia-key
```

### Linux/Mac
```bash
export NVIDIA_API_KEY="your-nvidia-key"
export HF_TOKEN="hf_xxxxxxxxxxxxx"
```

---

## 📁 Repository Structure

```
JaneStreet-Quant-Wars/
│
├── 📄 README.md                    # Main documentation
├── 📄 SCRIPTS.md                   # This file - usage guide
├── 📄 requirements.txt             # Python dependencies
│
├── 🚀 RUN_ALL.bat                  # Run complete pipeline
├── 🚀 1_setup.bat                  # Install dependencies
├── 🚀 2_run_all_competitions.bat   # Run all platforms
├── 🚀 3_evaluate.bat               # Evaluate results
│
├── 🚀 run_nvidia.bat               # NVIDIA NIM only
├── 🚀 run_ollama.bat               # Ollama Cloud only
├── 🚀 run_huggingface.bat          # Hugging Face only
│
├── 📊 evaluate_all.py              # Core evaluation engine
├── 📊 unified_leaderboard.csv      # Results (after evaluation)
├── 📊 unified_dashboard.html       # Interactive dashboard
│
├── 📁 platforms/                   # Platform-specific code
│   ├── nvidia/                     # NVIDIA NIM competition
│   ├── ollama/                     # Ollama Cloud competition
│   └── huggingface/                # Hugging Face competition
│
└── 📁 results/                     # Generated after running
    ├── nvidia/                     # NVIDIA notebooks
    ├── ollama/                     # Ollama notebooks
    └── huggingface/                # Hugging Face notebooks
```

---

## 🧪 Evaluation Engine

**File:** `evaluate_all.py`

Evaluates all generated notebooks and produces:
- MSE (Mean Squared Error)
- R² (R-squared) scores
- Unified leaderboard CSV
- Interactive HTML dashboard

**Usage:**
```cmd
python evaluate_all.py
```

---

## 📝 Requirements

- **Python:** 3.10 or higher
- **RAM:** 8GB minimum (16GB recommended)
- **Disk:** 500MB for code + 19GB for Jane Street data (optional)
- **Internet:** Required for API calls

---

## 🐛 Troubleshooting

### "Module not found" error
Run `1_setup.bat` to install dependencies.

### "API key not set" error
Set environment variables as shown above.

### "Rate limit exceeded"
Wait 60 seconds and retry. The scripts have built-in retry logic.

### "Model not found" (Ollama)
Some models may not be available on Ollama Cloud. Check the log for details.

### "Gated model" (Hugging Face)
Visit the model's HF page and click "Agree & Access" to unlock.

---

## 📊 Output Files

After running competitions and evaluation:

| File | Description |
|------|-------------|
| `unified_leaderboard.csv` | Complete results for all models |
| `unified_dashboard.html` | Interactive visualization |
| `results/nvidia/` | NVIDIA NIM generated notebooks |
| `results/ollama/` | Ollama Cloud generated notebooks |
| `results/huggingface/` | Hugging Face generated notebooks |

---

## 💡 Tips

1. **Start small:** Run `run_nvidia.bat` first to test your setup
2. **Use parallel mode:** All scripts use `--parallel` flag for faster execution
3. **Check logs:** Each platform creates timestamped log files
4. **Save API keys:** Use `setx` (Windows) or add to `~/.bashrc` (Linux) for persistence

---

## 📞 Need Help?

- Check the main [README.md](./README.md) for overview
- Review [RESULTS.md](./RESULTS.md) for detailed analysis
- Open an issue on GitHub for bugs or questions

---

**Last Updated:** March 2026
