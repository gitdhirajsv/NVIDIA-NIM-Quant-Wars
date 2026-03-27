@echo off
echo ============================================================
echo Hugging Face Competition
echo ============================================================
echo.
echo Runs 13 models via Hugging Face Inference API
echo Requires: HF_TOKEN environment variable
echo.
echo Models: Qwen, Llama, DeepSeek, Gemma, Phi, etc.
echo Estimated time: 15-30 minutes (free tier may be slower)
echo.
pause

cd huggingface_competition
python run_competition.py --parallel
cd ..

echo.
echo Complete! Check huggingface_results/ folder
pause
