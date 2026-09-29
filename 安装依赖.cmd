@echo off
chcp 65001 >nul
title AudioSR - install dependencies (needs internet, run once)
cd /d "%~dp0"
set NO_PROXY=*
set HF_ENDPOINT=https://hf-mirror.com
rem keep pip's download cache and temporary files inside this folder (not on C:)
set "TMP=%~dp0.tmp"
set "TEMP=%~dp0.tmp"
if not exist "%TMP%" mkdir "%TMP%"
set "PIP_CACHE=%~dp0.pip-cache"
type "%~dp0提示_安装依赖说明.txt"
call "%~dp0_find_python.cmd" --base-only
if errorlevel 1 goto nopython
if exist "%~dp0.venv\Scripts\python.exe" (
  echo.
  echo [INFO] .venv already exists - deleting it so it can be rebuilt.
  rmdir /s /q "%~dp0.venv"
)
echo Using base interpreter: %PY_BASE%
echo [INFO] GitHub is not required: the unused diffusers git dependency is skipped.
echo [INFO] pip cache and temp files stay in this folder (.pip-cache, .tmp).
"%PY_BASE%" -m venv "%~dp0.venv"
if errorlevel 1 goto fail
"%~dp0.venv\Scripts\python.exe" -m pip install --upgrade pip --cache-dir "%PIP_CACHE%"
"%~dp0.venv\Scripts\python.exe" -m pip install -r "%~dp0requirements.txt" -i https://pypi.tuna.tsinghua.edu.cn/simple --cache-dir "%PIP_CACHE%"
if errorlevel 1 goto fail
echo.
echo [OK] Dependencies installed. You can now double-click 双击启动网页界面.cmd
pause
exit /b 0

:nopython
type "%~dp0提示_没有兼容的Python.txt"
pause
exit /b 1

:fail
echo.
echo [FAIL] Install failed. Check the network; if python/urllib3 reports a
 echo        proxy error, run this in the window first:  set NO_PROXY=*
echo        (pip may also fail on the git+https diffusers line when GitHub is
 echo         unreachable - retry later or use a proxy.)
pause
exit /b 1
