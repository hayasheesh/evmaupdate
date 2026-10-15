# Windows の順番待ちの仕事（jobs_windows.json）を、画面を出さずに走らせる（WMI 経由。アプリの再起動で止まらない）。
#   powershell -NoProfile -ExecutionPolicy Bypass -File start_windows.ps1
$py = (Get-Command python).Source
$root = 'C:\Users\admin\Desktop\EVMALOCALUPDATE'
$here = "$root\execute_results\queue_20261015"
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
$cmd = "C:\Windows\System32\cmd.exe /c `"`"$py`" -X utf8 -u `"$here\queue.py`" windows >> `"$here\logs\queue_windows.log`" 2>&1`""
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd; CurrentDirectory = $here; ProcessStartupInformation = $si }
"queue ReturnValue=$($r.ReturnValue) PID=$($r.ProcessId)"
