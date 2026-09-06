@echo off
REM ==============================================================================
REM PAXG Forecast Lab - Lenh mo ung dung web mot buoc bind 127.0.0.1
REM ==============================================================================
echo [PAXG Forecast Lab] Khoi dong Web Streamlit va GPU Queue Supervisor...
set PYTHONPATH=%~dp0src;%~dp0
call %~dp0.venv-paxg\Scripts\activate.bat
python %~dp0scripts\launch_app.py --host 127.0.0.1 --port 8501
pause
