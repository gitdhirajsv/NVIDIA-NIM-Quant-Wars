@echo off
echo ============================================================
echo Ollama Cloud Competition
echo ============================================================
echo.
echo Runs 13 models via Ollama Cloud API
echo Requires: CLOUD_KEY_1, CLOUD_KEY_2, CLOUD_KEY_3
echo.
echo Models: Qwen, DeepSeek, Gemma, GLM, Kimi, Mistral, Nemotron
echo Estimated time: 10-20 minutes
echo.
pause

cd ollama_competition
python run_competition.py --parallel
cd ..

echo.
echo Complete! Check ollama_results/ folder
pause
