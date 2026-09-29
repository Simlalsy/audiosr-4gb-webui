@echo off
chcp 65001 >nul
title AudioSR - install dependencies (needs internet, run once)
cd /d "%~dp0"
set NO_PROXY=*
set HF_ENDPOINT=https://hf-mirror.com
type "%~dp0提示_安装依赖说明.txt"
set "PY_BASE="
for %%I in (python.exe) do if not defined PY_BASE set "PY_BASE=%%~$PATH:I"
if not defined PY_BASE (
  echo.
  echo [ERROR] python.exe not found on PATH.
  echo         Install Python 3.10 first and tick "Add python.exe to PATH":
  echo         https://www.python.org/downloads/release/python-31011/
  pause
  exit /b 1
)
echo Using base interpreter: %PY_BASE%
"%PY_BASE%" -m venv "%~dp0.venv"
if errorlevel 1 goto fail
"%~dp0.venv\Scripts\python.exe" -m pip install --upgrade pip
"%~dp0.venv\Scripts\python.exe" -m pip install -r "%~dp0requirements.txt" -i https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 goto fail
echo.
echo [OK] Dependencies installed. You can now double-click 双击启动网页界面.cmd
pause
exit /b 0

:fail
echo.
echo [FAIL] Install failed. Check the network; if python/urllib3 reports a
 echo        proxy error, run this in the window first:  set NO_PROXY=*
echo        (pip may also fail on the git+https diffusers line when GitHub is
 echo         unreachable - retry later or use a proxy.)
pause
exit /b 1
