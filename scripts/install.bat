@echo off
REM Double-click / CMD installer for Redactortron
cd /d "%~dp0\.."
python scripts\install_deps.py --with-poppler --api %*
if errorlevel 1 exit /b %errorlevel%
echo.
echo Start GUI:  python -m redactortron ui
pause
