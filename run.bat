@echo off
setlocal
chcp 65001 >nul
title Speech to Text
cd /d "%~dp0"

set "VPY=%~dp0venv311\Scripts\python.exe"
set "LOGFILE=%~dp0launcher.log"
set "URL=http://127.0.0.1:8000"
set "HEALTH_URL=http://127.0.0.1:8000/health"

echo ============================================
echo   Speech to Text - Launcher
echo ============================================
echo.

if not exist "%VPY%" (
    echo ERROR: Python environment not found at venv311.
    echo Please install Python 3.11 and run: py -3.11 -m venv venv311
    echo.
    pause
    exit /b 1
)

echo Using: %VPY%
echo.

REM ---- If a server is already running, just open the browser ----
curl.exe -s --connect-timeout 2 --max-time 3 "%HEALTH_URL%" >nul 2>&1
if %errorlevel%==0 (
    echo Server is already running. Opening browser...
    start "" "%URL%"
    exit /b 0
)

echo Server starting - please wait.
echo The browser opens AUTOMATICALLY once the server is ready.
echo (Model loads in background; first start takes ~1-2 min. Do NOT close this window.)
echo Live log is shown below and saved to: %LOGFILE%
echo Press Ctrl+C to stop the server.
echo ============================================
echo.

REM AUTO_OPEN_BROWSER tells app.py to open the browser itself when port 8000 answers.
REM -u = unbuffered output so progress appears immediately.
set "AUTO_OPEN_BROWSER=1"
"%VPY%" -u app.py

echo.
echo ============================================
echo Server has stopped.
echo Full log: %LOGFILE%
echo ============================================
pause
