@echo off
rem Opens the newest recorded episode (or the one given) in the MuJoCo viewer.
cd /d "%~dp0"
set EP=%~1
if "%EP%"=="" for /f "delims=" %%i in ('.venv\Scripts\python scripts\episodes.py --latest') do set EP=%%i
if "%EP%"=="" (echo No episodes recorded yet. & pause & exit /b 1)
echo Replaying %EP%
.venv\Scripts\python scripts\replay.py "%EP%"
pause
