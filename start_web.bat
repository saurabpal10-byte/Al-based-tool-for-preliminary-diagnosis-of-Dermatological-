@echo off
setlocal
pushd "%~dp0"

REM --- activate venv (make sure it exists) ---
call venv\Scripts\activate

REM --- make sure gradio is installed (safe to re-run) ---
python -m pip install --quiet gradio

REM --- optionally set a weights path (edit if needed) ---
set WEIGHTS=runs\resnet\best.pth

REM --- launch ---
python app_web.py
popd
endlocal
