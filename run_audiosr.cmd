@echo off
rem ============================================================
rem  AudioSR 4GB-VRAM launcher (RTX 3050 Laptop / 4 GB)
rem  Usage:
rem    run_audiosr.cmd -i "your_audio.wav"
rem    run_audiosr.cmd -i "long_song.mp3" --chunking --chunk_duration 10
rem      (files longer than 12 s auto-enable chunking anyway)
rem  Output goes to  output\<timestamp>\<name>_AudioSR_Processed_48K.wav
rem
rem  Model weights are looked up automatically: %AUDIOSR_CKPT%, then
rem  pytorch_model.bin next to this script, then .\models\, then one level up.
rem  When none is found the weights are downloaded from Hugging Face (unless
rem  AUDIOSR_OFFLINE=1, in which case the run stops with an error).
rem ============================================================
setlocal
set NO_PROXY=*
set HF_ENDPOINT=https://hf-mirror.com
"%~dp0.venv\Scripts\python.exe" "%~dp0run_lowvram.py" -s "%~dp0output" %*
set RC=%ERRORLEVEL%
endlocal & exit /b %RC%
