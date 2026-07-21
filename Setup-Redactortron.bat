@echo off
REM ============================================================
REM  Redactortron - one-time setup for THIS PC
REM  Builds a self-contained .venv inside this folder and
REM  installs all dependencies (+ Poppler for PDFs).
REM  Needs: Python 3.9+ installed, and internet access.
REM ============================================================
setlocal EnableDelayedExpansion
cd /d "%~dp0"

echo ============================================================
echo   Redactortron setup
echo   Folder: %CD%
echo ============================================================
echo.

REM --- 1. Locate a Python to bootstrap the virtual environment ---
set "BOOT="
where py >nul 2>&1 && set "BOOT=py -3"
if not defined BOOT (
    where python >nul 2>&1 && set "BOOT=python"
)
if not defined BOOT (
    echo Python 3.9+ was not found on this PC.
    echo Install it from https://www.python.org/downloads/
    echo and tick "Add python.exe to PATH", then run this again.
    echo.
    pause
    exit /b 1
)
echo Using Python bootstrap: %BOOT%

REM --- 2. Create the local virtual environment (.venv) ---
set "VENV_PY=%~dp0.venv\Scripts\python.exe"

REM A .venv copied from another PC has broken hard-coded paths - rebuild it.
if exist "%VENV_PY%" (
    "%VENV_PY%" -c "import sys" >nul 2>&1
    if errorlevel 1 (
        echo Existing .venv does not work on this PC ^(copied from another machine^).
        echo Rebuilding it...
        rmdir /s /q ".venv"
    )
)

if not exist "%VENV_PY%" (
    echo Creating virtual environment .venv ...
    %BOOT% -m venv .venv
    if errorlevel 1 (
        echo Could not create the virtual environment.
        pause
        exit /b 1
    )
)

REM --- 3. Install Redactortron + dependencies (+ bundled Poppler) ---
echo.
echo Upgrading pip ...
"%VENV_PY%" -m pip install --upgrade pip
echo.
echo Installing Redactortron and dependencies.
echo (First time downloads PyTorch/docTR/GLiNER - this can take a while.)
"%VENV_PY%" scripts\install_deps.py --with-poppler --api --fetch-models
if errorlevel 1 (
    echo.
    echo Dependency installation failed. See messages above.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   Setup complete.
echo   Double-click  Start-Redactortron.bat  to launch the app.
echo ============================================================
pause
