# 6 station AEMO（観測に全体のずれと入札の先読みを戻し、報酬を tanh_linear・倍率1に戻した版）を、画面を出さずに起動する（WMI 経由）。温度の見張りも起動し直す。
#   powershell -NoProfile -ExecutionPolicy Bypass -File start.ps1
# 入札は本線（wr_gs_6station_20261011）と同じものを使う（run_task.py は作り終えた日を飛ばす）。温度の見張りも起動し直す。
$py = (Get-Command python).Source
$root = 'C:\Users\admin\Desktop\EVMALOCALUPDATE'
$run = "$root\execute_results\resbid_6station_20261012"
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
foreach ($task in @('aemo')) {
    $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
        CommandLine = "`"$py`" -X utf8 -u `"$run\run_task.py`" $task 4 2 3"
        CurrentDirectory = $root
        ProcessStartupInformation = $si
    }
    "$task ReturnValue=$($r.ReturnValue) PID=$($r.ProcessId)"
}
$guard = "$root\execute_results\hyb2_cap77wd_7station_20261009"
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'gpu_guard' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
$g = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$guard\gpu_guard.ps1`""
    CurrentDirectory = $guard
    ProcessStartupInformation = $si
}
"guard ReturnValue=$($g.ReturnValue) PID=$($g.ProcessId)"
