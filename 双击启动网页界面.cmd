@echo off
title AudioSR Web UI (4GB VRAM) - keep this window open
cd /d "%~dp0"
set NO_PROXY=*
set HF_ENDPOINT=https://hf-mirror.com
echo ============================================================
echo  AudioSR Web UI
echo  - a browser page will open automatically
echo  - keep this window open while using the tool
echo  - press Ctrl+C here to stop the server
echo ============================================================
".venv\Scripts\python.exe" "webui_server.py"
pause
