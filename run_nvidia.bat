@echo off
echo ============================================================
echo NVIDIA NIM Competition
echo ============================================================
echo.
echo Runs 67 models via NVIDIA NIM API
echo Requires: NVIDIA_API_KEY environment variable
echo.
echo Models: Llama, Mistral, Qwen, Nemotron, Phi, etc.
echo Estimated time: 20-40 minutes
echo.
pause

python run_competition.py --parallel

echo.
echo Complete! Check generated_notebooks/ folder
pause
