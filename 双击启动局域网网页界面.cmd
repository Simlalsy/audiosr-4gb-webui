@echo off
chcp 65001 >nul
title AudioSR Web UI (LAN) - keep this window open
cd /d "%~dp0"
set NO_PROXY=*
set HF_ENDPOINT=https://hf-mirror.com

call "%~dp0_find_python.cmd"
if errorlevel 1 goto nopython
echo ============================================================
echo  AudioSR Web UI - LAN mode
echo  - the server listens on ALL network interfaces
echo  - this window prints the http:// URL for phones / other PCs
echo    on the same Wi-Fi or LAN
echo  - keep this window open while using the tool
echo  - press Ctrl+C here to stop the server
echo  - on first run, allow Python through the Windows Firewall
echo    (private networks) if Windows asks
echo ============================================================
"%PY_EXE%" "%~dp0webui_server.py" --host 0.0.0.0
pause
exit /b 0

:nopython
type "%~dp0提示_未找到Python环境.txt"
pause
exit /b 1
