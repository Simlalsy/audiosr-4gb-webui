@echo off
rem ===============================================================
rem  Locate a Python interpreter for the launcher scripts.
rem
rem    _find_python.cmd              -> PY_EXE   (bundled .venv first, else a
rem                                   system python that already has the deps)
rem    _find_python.cmd --base-only  -> PY_BASE  (compatible 3.9-3.11 base
rem                                   interpreter, used to create the venv)
rem
rem  Exit code 0 on success, 1 when nothing suitable was found.
rem  Override the base interpreter with:  set AUDIOSR_PY_BASE=<python.exe>
rem ===============================================================
set "PY_EXE="
set "PY_BASE="
set "VCHECK=import sys;sys.exit(0 if (3,9)<=sys.version_info[:2]<=(3,11) else 1)"

if "%~1"=="--base-only" goto find_base

if exist "%~dp0.venv\Scripts\python.exe" set "PY_EXE=%~dp0.venv\Scripts\python.exe"
if defined PY_EXE exit /b 0

call "%~dp0_find_python.cmd" --base-only
if errorlevel 1 exit /b 1
rem a plain system python is only usable when the deps are already there
"%PY_BASE%" -c "import torch, soundfile, huggingface_hub" >nul 2>nul
if errorlevel 1 exit /b 1
set "PY_EXE=%PY_BASE%"
exit /b 0

:find_base
rem 0) explicit override (the version is still validated)
if defined AUDIOSR_PY_BASE (
  if exist "%AUDIOSR_PY_BASE%" (
    "%AUDIOSR_PY_BASE%" -c "%VCHECK%" >nul 2>nul
    if not errorlevel 1 set "PY_BASE=%AUDIOSR_PY_BASE%"
  )
)
if defined PY_BASE exit /b 0

rem 1) the py launcher knows every installed CPython
for %%V in (3.10 3.11 3.9) do if not defined PY_BASE for /f "delims=" %%P in ('py -%%V -c "import sys;print(sys.executable)" 2^>nul') do set "PY_BASE=%%P"
if defined PY_BASE exit /b 0

rem 2) well-known install locations
for %%P in ("%LOCALAPPDATA%\Programs\Python\Python310\python.exe" "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" "%LOCALAPPDATA%\Programs\Python\Python39\python.exe" "%ProgramFiles%\Python310\python.exe" "%ProgramFiles%\Python311\python.exe" "C:\Python310\python.exe") do if not defined PY_BASE if exist %%~P set "PY_BASE=%%~P"
if defined PY_BASE exit /b 0

rem 3) python.exe on PATH - accept it only when the version is 3.9-3.11
rem    (e.g. the Python Install Manager shim ships 3.13+ and must be rejected)
for %%I in (python.exe) do if not defined CAND set "CAND=%%~$PATH:I"
if not defined CAND exit /b 1
"%CAND%" -c "%VCHECK%" >nul 2>nul
if errorlevel 1 exit /b 1
set "PY_BASE=%CAND%"
exit /b 0
