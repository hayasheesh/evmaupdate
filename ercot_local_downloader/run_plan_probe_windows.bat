@echo off
setlocal
cd /d "%~dp0"
python probe_ercot_plan_columns.py %*
if errorlevel 1 echo Probe failed. Read the error above.
pause
