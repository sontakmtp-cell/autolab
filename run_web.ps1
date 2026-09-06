# ==============================================================================
# PAXG Forecast Lab - Lenh mo ung dung web mot buoc bind 127.0.0.1 (PowerShell)
# ==============================================================================
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Write-Host "[PAXG Forecast Lab] Khoi dong Web Streamlit va GPU Queue Supervisor..." -ForegroundColor Yellow

$env:PYTHONPATH = "$ScriptDir\src;$ScriptDir"
$PythonExe = "$ScriptDir\.venv-paxg\Scripts\python.exe"

if (-not (Test-Path $PythonExe)) {
    Write-Error "Khong tim thay Python tai $PythonExe! Vui long kiem tra moi truong .venv-paxg."
    exit 1
}

& $PythonExe "$ScriptDir\scripts\launch_app.py" --host 127.0.0.1 --port 8501
