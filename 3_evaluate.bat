@echo off
echo ============================================================
echo Jane Street Quant Wars - Evaluate All Results
echo ============================================================
echo.
echo This will evaluate all generated notebooks and produce:
echo   - MSE scores for each model
echo   - R² (R-squared) scores  
echo   - Unified leaderboard CSV
echo   - Interactive HTML dashboard
echo.
pause

python evaluate_all.py

echo.
echo ============================================================
echo Evaluation Complete!
echo ============================================================
echo.
echo Results:
echo   - unified_leaderboard.csv - Full results
echo   - unified_dashboard.html - Interactive visualization
echo.
pause
