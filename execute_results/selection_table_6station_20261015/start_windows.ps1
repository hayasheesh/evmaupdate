# 6 station AEMO の、EVの選び方・指令の本数・予測精度の表の入札と未見通過率を、画面を出さずに回す（WMI 経由。アプリの再起動で止まらない）。
#   powershell -NoProfile -ExecutionPolicy Bypass -File start_windows.ps1
$py = (Get-Command python).Source
$here = 'C:\Users\admin\Desktop\EVMALOCALUPDATE\execute_results\selection_table_6station_20261015'
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
$cmd = "C:\Windows\System32\cmd.exe /c `"`"$py`" -X utf8 -u `"$here\run.py`" aemo 8 24 >> `"$here\logs\run_aemo.log`" 2>&1`""
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd; CurrentDirectory = $here; ProcessStartupInformation = $si }
"aemo ReturnValue=$($r.ReturnValue) PID=$($r.ProcessId)"
