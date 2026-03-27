@echo off
echo ============================================================
echo Jane Street Quant Wars - Setup Script
echo ============================================================
echo.
echo This script will install all required Python dependencies.
echo.

python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ============================================================
echo Installation Complete!
echo ============================================================
echo.
echo Next steps:
echo 1. Set your API keys in environment variables
echo 2. Run one of the competition scripts below
echo.
pause
