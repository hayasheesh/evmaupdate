@echo off
setlocal
cd /d "%~dp0"
python download_ercot_sced_waveforms.py %*
if errorlevel 1 echo Download failed. Read the error above; no bank was built.
pause
