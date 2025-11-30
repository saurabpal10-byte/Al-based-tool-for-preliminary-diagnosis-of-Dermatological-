@echo off
REM Ensure we run from the project folder
cd /d "%~dp0"
REM Activate venv (optional — used if you prefer the venv)
call venv\Scripts\activate
REM Run the GUI from source (shows console so errors are visible)
python gui_app.py
