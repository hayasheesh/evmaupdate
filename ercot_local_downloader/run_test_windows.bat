@echo off
setlocal
cd /d "%~dp0"
python download_ercot_sced_waveforms.py --test-days 1 --out-dir ercot_sced_rtc_test_output %*
if errorlevel 1 echo Download failed. Read the error above; no bank was built.
pause
