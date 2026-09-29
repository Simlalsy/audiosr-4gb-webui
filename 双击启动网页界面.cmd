@echo off
chcp 65001 >nul
title AudioSR Web UI (4GB VRAM) - keep this window open
cd /d "%~dp0"
set NO_PROXY=*
set HF_ENDPOINT=https://hf-mirror.com

call "%~dp0_find_python.cmd"
if errorlevel 1 goto nopython
echo ============================================================
echo  AudioSR Web UI
echo  - a browser page will open automatically
echo  - keep this window open while using the tool
echo  - press Ctrl+C here to stop the server
echo ============================================================
"%PY_EXE%" "%~dp0webui_server.py"
pause
exit /b 0

:nopython
type "%~dp0提示_未找到Python环境.txt"
pause
exit /b 1
