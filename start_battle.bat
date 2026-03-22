@echo off
title NVIDIA NIM Quant Wars - Launcher
echo Starting NVIDIA NIM Quant Wars...

:: Check for Virtual Environment
if not exist "venv" (
    echo [1/3] Creating temporary virtual environment...
    python -m venv venv
)

:: Install Dependencies
echo [2/3] Installing/Updating required libraries...
call venv\Scripts\activate
pip install langchain langchain-nvidia-ai-endpoints polars xgboost nbformat pandas pyarrow

:: Run the Battle
echo [3/3] Launching Competition Orchestrator...
python run_competition.py

echo.
echo ===================================================
echo   COMPETITION FINISHED. CHECK FOR NOTEBOOKS.
echo ===================================================
pause
