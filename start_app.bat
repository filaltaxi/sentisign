@echo off
cd /d "%~dp0"
echo Starting SentiSign v2 at http://127.0.0.1:8000 (close this window to stop)
start "" http://127.0.0.1:8000
"%CD%\env\python.exe" server\app.py
pause
