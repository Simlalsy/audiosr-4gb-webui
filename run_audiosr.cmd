@echo off
rem ============================================================
rem  AudioSR 4GB-VRAM launcher (RTX 3050 Laptop / 4 GB)
rem  Usage:
rem    run_audiosr.cmd -i "your_audio.wav"
rem    run_audiosr.cmd -i "long_song.mp3" --chunking --chunk_duration 10
rem      (files longer than 12 s auto-enable chunking anyway)
rem  Output goes to  output\<timestamp>\<name>_AudioSR_Processed_48K.wav
rem
rem  Model weights: set AUDIOSR_CKPT to your pytorch_model.bin, or put the
rem  file one level ABOVE this repo (default: ..\pytorch_model.bin). When it
rem  is missing, the weights are downloaded from Hugging Face instead.
rem ============================================================
setlocal
set NO_PROXY=*
set HF_ENDPOINT=https://hf-mirror.com
if "%AUDIOSR_CKPT%"=="" set AUDIOSR_CKPT=%~dp0..\pytorch_model.bin
"%~dp0.venv\Scripts\python.exe" "%~dp0run_lowvram.py" -s "%~dp0output" %*
endlocal
