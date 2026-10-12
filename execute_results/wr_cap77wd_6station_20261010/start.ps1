# 6 station（職場3局・住宅3局）の aemo と ercot を、入札から学習まで画面を出さずに起動する（WMI 経由。アプリの再起動で止まらない）。
#   powershell -NoProfile -ExecutionPolicy Bypass -File start.ps1
# 入札の並列数は 1本あたり 学習用4日 × シナリオ3、検証用2日 × シナリオ3。入札がそろうと学習へ進む（run_task.py）。
$py = (Get-Command python).Source
$run = 'C:\Users\admin\Desktop\EVMALOCALUPDATE\execute_results\wr_cap77wd_6station_20261010'
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
foreach ($task in @('aemo', 'ercot')) {
    $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
        CommandLine = "`"$py`" -X utf8 -u `"$run\run_task.py`" $task 4 2 3"
        CurrentDirectory = 'C:\Users\admin\Desktop\EVMALOCALUPDATE'
        ProcessStartupInformation = $si
    }
    "$task ReturnValue=$($r.ReturnValue) PID=$($r.ProcessId)"
}
