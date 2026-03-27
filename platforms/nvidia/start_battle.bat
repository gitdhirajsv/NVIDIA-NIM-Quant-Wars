@echo off
title NVIDIA NIM Quant Wars - Battle Royale
setlocal enabledelayedexpansion

echo ============================================================
echo   NVIDIA NIM QUANT WARS - BATTLE ROYALE
echo ============================================================
echo.

cd /d "%~dp0"

:: ============================================================
:: 0. NVIDIA API KEY SETUP
:: ============================================================
if "%NVIDIA_API_KEY%"=="" (
    echo   NVIDIA_API_KEY is not set in your environment.
    echo.
    set /p NVIDIA_API_KEY="  Enter your NVIDIA API Key: "
    echo.
)

if "%NVIDIA_API_KEY%"=="" (
    echo   ERROR: No API key provided. Exiting.
    pause
    exit /b 1
)

:: ============================================================
:: 1. CLEANUP UNNECESSARY FILES
:: ============================================================
echo [STEP 1/4] Cleaning up unnecessary files...
echo.

:: Delete all .ipynb files (generated notebooks will be recreated)
for %%f in (generated_notebooks\*.ipynb) do (
    del /q "%%f" 2>nul
)

:: Delete old log files
for %%f in (competition_log_*.txt execution_output.log ALL_SOLUTIONS.txt TOP_5_MODELS.txt) do (
    del /q "%%f" 2>nul
)

:: Delete image files
for %%f in (*.png *.jpg *.jpeg) do (
    del /q "%%f" 2>nul
)

:: Delete temporary Python files
for %%f in (test_data_loading.py working_example.py run_notebooks_one_by_one.py) do (
    del /q "%%f" 2>nul
)

:: Delete old bat files (keeping only this one)
for %%f in (qwen-here.bat run_one_by_one.bat) do (
    del /q "%%f" 2>nul
)

:: Delete instruction files that are no longer needed (README.md kept)
for %%f in (INSTRUCTIONS_FOR_QWEN.txt FILES_SAFE_TO_DELETE.txt) do (
    del /q "%%f" 2>nul
)

echo   Cleanup complete!
echo.

:: ============================================================
:: 2. CHECK VIRTUAL ENVIRONMENT
:: ============================================================
echo [STEP 2/4] Checking virtual environment...
echo.

if not exist "venv" (
    echo   Creating virtual environment...
    python -m venv venv
    echo   Virtual environment created!
) else (
    echo   Virtual environment found.
)
echo.

:: ============================================================
:: 3. INSTALL DEPENDENCIES
:: ============================================================
echo [STEP 3/4] Installing required libraries...
echo.

call venv\Scripts\activate.bat

pip install --quiet langchain langchain-nvidia-ai-endpoints polars xgboost nbformat pandas pyarrow scikit-learn

echo   Dependencies installed!
echo.

:: ============================================================
:: 4. RUN THE BATTLE
:: ============================================================
echo [STEP 4/4] Launching Battle Royale...
echo.
echo   This will:
echo   - Fetch models from NVIDIA API
echo   - Generate Python scripts for each model
echo   - Save notebooks one by one (with delays to prevent crashes)
echo.
echo   Press Ctrl+C to cancel, or wait to continue...
timeout /t 3 /nobreak >nul

python run_competition.py

:: ============================================================
:: FINISH
:: ============================================================
echo.
echo ============================================================
echo   BATTLE COMPLETE!
echo   Check generated .ipynb files in this folder.
echo ============================================================
echo.
pause
