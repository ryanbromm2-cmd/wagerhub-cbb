@echo off
title WagerHub — CBB Totals
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"

echo.
echo  =========================================
echo   WagerHub  ^|  CBB Totals Model
echo  =========================================
echo.

REM ── Check Python ──────────────────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo  ERROR: Python not found.
    echo  Download Python from https://python.org
    pause
    exit /b 1
)

REM ── Install dependencies if needed ────────────────────────────────────────
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo  First run: installing dependencies...
    echo  This only happens once and may take a minute.
    echo.
    pip install -r requirements.txt --quiet
    if errorlevel 1 (
        echo.
        echo  ERROR: Could not install dependencies.
        echo  Try running:  pip install -r requirements.txt
        pause
        exit /b 1
    )
    echo  Dependencies installed successfully.
    echo.
)

REM ── Open browser after short delay ───────────────────────────────────────
start "" /b cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:8501"

REM ── Launch dashboard ──────────────────────────────────────────────────────
echo  Opening WagerHub at http://localhost:8501
echo  Press Ctrl+C to stop.
echo.
streamlit run dashboard/app.py --server.port 8501 --server.headless false --browser.gatherUsageStats false

pause
