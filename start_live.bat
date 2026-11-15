@echo off
setlocal
cd /d "%~dp0"
rem Runs SentiSign on this laptop's GPU and shares it at a temporary public https link through Cloudflare Tunnel.
if not exist tools mkdir tools
if exist tools\cloudflared.exe goto :run
echo Downloading Cloudflare Tunnel - cloudflared.exe, about 60 MB, from github.com/cloudflare/cloudflared ...
curl -L -o tools\cloudflared.exe https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe
:run
start "SentiSign server - close to stop" "%CD%\env\python.exe" server\app.py
echo.
echo Starting the public link. Look below for a line with https://....trycloudflare.com
echo Open that link on any device. The models take about a minute to load the first time.
echo Close this window to stop sharing.
echo.
tools\cloudflared.exe tunnel --url http://127.0.0.1:8000
pause
