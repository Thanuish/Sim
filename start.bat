@echo off
rem Starts the teleop server. Double-click, or from a terminal:  start.bat [--task blocks] [--cameras] ...
rem Default: plain HTTP on port 8080 for the Quest over USB-C (adb reverse is set up automatically
rem when adb is on the PATH). For Wi-Fi (HTTPS on port 8443) run:  start.bat --https
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
    echo Run setup_windows.bat first.
    pause
    exit /b 1
)
where adb >nul 2>nul && adb reverse tcp:8080 tcp:8080 >nul 2>nul
if /i "%~1"=="--https" goto https
.venv\Scripts\python -m vrteleop.server --http %*
goto end
:https
shift
set ARGS=
:collect
if "%~1"=="" goto run_https
set ARGS=%ARGS% %1
shift
goto collect
:run_https
.venv\Scripts\python -m vrteleop.server %ARGS%
:end
pause
