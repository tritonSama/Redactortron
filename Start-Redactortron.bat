@echo off
REM ============================================================
REM  Redactortron - click to run the web UI.
REM  On a brand-new PC it runs setup automatically the first
REM  time (needs internet), then launches every time after.
REM ============================================================
setlocal
cd /d "%~dp0"

set "VENV_PY=%~dp0.venv\Scripts\python.exe"
set "NEED_SETUP="

if not exist "%VENV_PY%" set "NEED_SETUP=1"
if not defined NEED_SETUP (
    REM Venv exists but may be copied from another PC or missing packages.
    "%VENV_PY%" -c "import redactortron" >nul 2>&1
    if errorlevel 1 set "NEED_SETUP=1"
)

if defined NEED_SETUP (
    echo First run on this PC - installing Redactortron.
    echo This needs internet and a few minutes...
    echo.
    call "%~dp0Setup-Redactortron.bat"
    if errorlevel 1 (
        echo.
        echo Setup did not finish. Fix the issue above and try again.
        pause
        exit /b 1
    )
)

echo Launching Redactortron web UI ...
echo (A browser tab opens automatically. Close this window to stop.)
"%VENV_PY%" -m redactortron ui
pause
