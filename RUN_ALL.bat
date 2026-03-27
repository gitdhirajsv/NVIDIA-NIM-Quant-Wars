@echo off
echo ============================================================
echo Jane Street Quant Wars - Complete Pipeline
echo ============================================================
echo.
echo This will run the COMPLETE pipeline:
echo   1. Install dependencies
echo   2. Run all 3 platform competitions
echo   3. Evaluate all results
echo.
echo Estimated time: 30-60 minutes depending on API rate limits
echo.
pause

echo.
echo [STEP 1/3] Installing dependencies...
echo ============================================================
call 1_setup.bat

echo.
echo [STEP 2/3] Running all competitions...
echo ============================================================
call 2_run_all_competitions.bat

echo.
echo [STEP 3/3] Evaluating results...
echo ============================================================
call 3_evaluate.bat

echo.
echo ============================================================
echo COMPLETE! Check unified_dashboard.html for results
echo ============================================================
pause
