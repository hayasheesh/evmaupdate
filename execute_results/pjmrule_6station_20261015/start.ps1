# PJM の規則の 6 station の入札と学習を、画面を出さずに起動する（WMI 経由。アプリの再起動で止まらない）。
#   powershell -NoProfile -ExecutionPolicy Bypass -File start.ps1
# 入札は学習用3並列・検証用2並列（各3シナリオ並列）で、優先度を一段下げて作る（run_task.py）。
$py = (Get-Command python).Source
$root = 'C:\Users\admin\Desktop\EVMALOCALUPDATE'
$run = "$root\execute_results\pjmrule_6station_20261015"
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = "`"$py`" -X utf8 -u `"$run\run_task.py`" 3 2 3"
    CurrentDirectory = $root
    ProcessStartupInformation = $si
}
"pjm ReturnValue=$($r.ReturnValue) PID=$($r.ProcessId)"
