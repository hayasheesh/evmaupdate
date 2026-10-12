# 中央のルールベース＋蓄電池の大きさを振る評価を、済んだエピソードの続きから画面を出さずに起動する（WMI 経由）。
$py = (Get-Command python).Source
$run = 'C:\Users\admin\Desktop\EVMALOCALUPDATE\execute_results\rule_bess_6station_20261011'
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
$sizes = @{ aemo = '300,400,500,600,800,1000'; ercot = '200,300,400,500,600,800,1000' }
foreach ($task in @('aemo', 'ercot')) {
    $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
        CommandLine = "C:\Windows\System32\cmd.exe /c `"`"$py`" -X utf8 -u run_rule_bess.py $task $($sizes[$task]) 4 2 10 >> $task.log 2>> $task.err`""
        CurrentDirectory = $run
        ProcessStartupInformation = $si
    }
    "$task ReturnValue=$($r.ReturnValue) PID=$($r.ProcessId)"
}
