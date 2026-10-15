# CPU が熱いあいだ順番待ちの仕事を一時停止する見張り（heat_guard.py）を、画面を出さずに起動する（WMI 経由）。
#   powershell -NoProfile -ExecutionPolicy Bypass -File start_heat_guard.ps1
$py = (Get-Command python).Source
$here = 'C:\Users\admin\Desktop\EVMALOCALUPDATE\execute_results\queue_20261015'
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
$cmd = "C:\Windows\System32\cmd.exe /c `"`"$py`" -X utf8 -u `"$here\heat_guard.py`" >> `"$here\logs\heat_guard.log`" 2>&1`""
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd; CurrentDirectory = $here; ProcessStartupInformation = $si }
"heat_guard ReturnValue=$($r.ReturnValue) PID=$($r.ProcessId)"