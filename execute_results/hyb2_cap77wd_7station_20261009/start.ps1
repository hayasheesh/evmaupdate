# 7 station の aemo と ercot の本線（大域＋局所、局所の報酬3.26倍）を、作ってある入札で学習だけ画面を出さずに起動する（WMI 経由）。
#   powershell -NoProfile -ExecutionPolicy Bypass -File start.ps1
$py = (Get-Command python).Source
$root = 'C:\Users\admin\Desktop\EVMALOCALUPDATE'
$run = "$root\execute_results\hyb2_cap77wd_7station_20261009"
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
foreach ($task in @('aemo', 'ercot')) {
    $out = "$run\logs\${task}_pretrain.stdout.log"
    $err = "$run\logs\${task}_pretrain.stderr.log"
    $cmd = "C:\Windows\System32\cmd.exe /c `"`"$py`" -X utf8 -u `"$run\stage_worker.py`" $task pretrain >> `"$out`" 2>> `"$err`"`""
    $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
        CommandLine = $cmd
        CurrentDirectory = $root
        ProcessStartupInformation = $si
    }
    "$task ReturnValue=$($r.ReturnValue) PID=$($r.ProcessId)"
}
