@echo off
chcp 65001 >nul
title AudioSR - pick an audio file to process
cd /d "%~dp0"
set NO_PROXY=*
set HF_ENDPOINT=https://hf-mirror.com

call "%~dp0_find_python.cmd"
if errorlevel 1 goto nopython
echo ============================================================
echo  AudioSR file processor
echo  - a file-picker dialog will open; choose your audio
echo  - output folder opens automatically when finished
echo ============================================================
"%PY_EXE%" "%~dp0pick_and_run.py"
pause
exit /b 0

:nopython
type "%~dp0提示_未找到Python环境.txt"
pause
exit /b 1
