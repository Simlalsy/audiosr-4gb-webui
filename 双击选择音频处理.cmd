@echo off
title AudioSR - pick an audio file to process
cd /d "%~dp0"
set NO_PROXY=*
set HF_ENDPOINT=https://hf-mirror.com
echo ============================================================
echo  AudioSR file processor
echo  - a file-picker dialog will open; choose your audio
echo  - output folder opens automatically when finished
echo ============================================================
".venv\Scripts\python.exe" "pick_and_run.py"
pause
