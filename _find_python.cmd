@echo off
rem ---------------------------------------------------------------
rem  Shared helper: pick a usable Python interpreter.
rem  Success -> PY_EXE is set, exit code 0.
rem  Failure -> exit code 1 (the caller prints the guidance).
rem  1) bundled .venv (created by 安装依赖.cmd or the online release)
rem  2) a system python that already has the dependencies installed
rem ---------------------------------------------------------------
set "PY_EXE="
if exist "%~dp0.venv\Scripts\python.exe" set "PY_EXE=%~dp0.venv\Scripts\python.exe"
if defined PY_EXE exit /b 0

for %%I in (python.exe) do if not defined PY_EXE set "PY_EXE=%%~$PATH:I"
if not defined PY_EXE exit /b 1

rem a plain system python is only usable when the deps are already there
"%PY_EXE%" -c "import torch, soundfile, huggingface_hub" >nul 2>nul
if errorlevel 1 exit /b 1
exit /b 0
