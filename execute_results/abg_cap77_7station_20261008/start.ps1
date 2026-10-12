# 7 station の aemo を、入札から学習（ABG、定格7.7 kW）まで画面を出さずに起動する（WMI 経由。アプリの再起動で止まらない）。
#   powershell -NoProfile -ExecutionPolicy Bypass -File start.ps1
# 入札の並列数は 学習用6日 × シナリオ3、検証用2日 × シナリオ3（約32プロセス）。
$py = (Get-Command python).Source
$run = 'C:\Users\admin\Desktop\EVMALOCALUPDATE\execute_results\abg_cap77_7station_20261008'
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
foreach ($task in @('aemo')) {
    $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
        CommandLine = "`"$py`" -X utf8 -u `"$run\run_task.py`" $task 6 2 3"
        CurrentDirectory = 'C:\Users\admin\Desktop\EVMALOCALUPDATE'
        ProcessStartupInformation = $si
    }
    "$task ReturnValue=$($r.ReturnValue) PID=$($r.ProcessId)"
}
