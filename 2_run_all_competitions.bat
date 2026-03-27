@echo off
echo ============================================================
echo Jane Street Quant Wars - Run All Competitions
echo ============================================================
echo.
echo This will run competitions on all 3 platforms:
echo   - NVIDIA NIM (67 models)
echo   - Ollama Cloud (13 models)  
echo   - Hugging Face (13 models)
echo.
echo Make sure you have set your API keys first!
echo.
pause

echo.
echo [1/3] Running NVIDIA NIM Competition...
echo ============================================================
python run_competition.py --parallel

echo.
echo [2/3] Running Ollama Cloud Competition...
echo ============================================================
cd ollama_competition
python run_competition.py --parallel
cd ..

echo.
echo [3/3] Running Hugging Face Competition...
echo ============================================================
cd huggingface_competition
python run_competition.py --parallel
cd ..

echo.
echo ============================================================
echo All Competitions Complete!
echo ============================================================
echo.
echo Now run: 2_evaluate.bat to evaluate all results
echo.
pause
