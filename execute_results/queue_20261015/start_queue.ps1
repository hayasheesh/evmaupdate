# Windows の順番待ちの仕事 jobs_<name>.json を、画面を出さずに走らせる（WMI 経由。アプリの再起動で止まらない）。
#   powershell -NoProfile -ExecutionPolicy Bypass -File start_queue.ps1 <name>
param([Parameter(Mandatory = $true)][string]$name)
$py = (Get-Command python).Source
$root = 'C:\Users\admin\Desktop\EVMALOCALUPDATE'
$here = "$root\execute_results\queue_20261015"
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
$cmd = "C:\Windows\System32\cmd.exe /c `"`"$py`" -X utf8 -u `"$here\queue.py`" $name >> `"$here\logs\queue_$name.log`" 2>&1`""
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd; CurrentDirectory = $here; ProcessStartupInformation = $si }
"queue $name ReturnValue=$($r.ReturnValue) PID=$($r.ProcessId)"
