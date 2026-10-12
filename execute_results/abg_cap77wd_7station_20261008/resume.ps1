# 7 station の aemo と ercot の ABG 学習を、保存した学習状態から画面を出さずに再開する（WMI 経由）。
#   powershell -NoProfile -ExecutionPolicy Bypass -File resume.ps1
$py = (Get-Command python).Source
$root = 'C:\Users\admin\Desktop\EVMALOCALUPDATE'
$run = "$root\execute_results\abg_cap77wd_7station_20261008"
$runs = @{
    'aemo'  = "$root\archive\abg_cap77wd_aemoplan_7station_20261009_091047"
    'ercot' = "$root\archive\abg_cap77wd_ercotplan_7station_20261009_085250"
}
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
foreach ($task in @('aemo', 'ercot')) {
    $out = "$run\logs\${task}_pretrain_resume.stdout.log"
    $err = "$run\logs\${task}_pretrain_resume.stderr.log"
    $cmd = "C:\Windows\System32\cmd.exe /c `"`"$py`" -X utf8 -u `"$run\stage_worker.py`" $task pretrain --resume-run `"$($runs[$task])`" >> `"$out`" 2>> `"$err`"`""
    $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
        CommandLine = $cmd
        CurrentDirectory = $root
        ProcessStartupInformation = $si
    }
    "$task ReturnValue=$($r.ReturnValue) PID=$($r.ProcessId)"
}
