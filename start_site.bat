@echo off
REM Activates venv and starts the site server
setlocal

REM adjust this if your venv folder is different
if exist venv\Scripts\activate.bat (
  call venv\Scripts\activate
) else (
  echo Virtualenv activation script not found in venv\Scripts\activate.bat
  echo Please run: python -m venv venv ^&^& venv\Scripts\activate
  pause
  exit /b 1
)

REM install lite dependencies if missing (no harm)
pip install --quiet flask pillow torch torchvision matplotlib || echo "Some packages may already be present"

REM run the site (use port 8000 by default)
python site_server.py

pause
