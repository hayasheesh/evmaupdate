# 6 station（職場3局・住宅3局）の aemo と ercot の本線（観測を減らした版）を、保存した学習状態から画面を出さずに再開する（WMI 経由）。
#   powershell -NoProfile -ExecutionPolicy Bypass -File resume.ps1
$py = (Get-Command python).Source
$root = 'C:\Users\admin\Desktop\EVMALOCALUPDATE'
$run = "$root\execute_results\wr_cap77wd_6station_20261010"
$runs = @{
    'aemo'  = "$root\archive\wr_obsdrop_cap77wd_aemoplan_6station_20261010_214058"
    'ercot' = "$root\archive\wr_obsdrop_cap77wd_ercotplan_6station_20261010_192728"
}
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
foreach ($task in @('aemo', 'ercot')) {
    Remove-Item -ErrorAction SilentlyContinue "$($runs[$task])\resume\STOP_REQUESTED"
    $out = "$run\logs\${task}_pretrain.stdout.log"
    $err = "$run\logs\${task}_pretrain.stderr.log"
    $cmd = "C:\Windows\System32\cmd.exe /c `"`"$py`" -X utf8 -u `"$run\stage_worker.py`" $task pretrain --resume-run `"$($runs[$task])`" >> `"$out`" 2>> `"$err`"`""
    $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
        CommandLine = $cmd
        CurrentDirectory = $root
        ProcessStartupInformation = $si
    }
    "$task ReturnValue=$($r.ReturnValue) PID=$($r.ProcessId)"
}
# 温度の見張りも同じく画面を出さずに起動する（見張りの本体は 7 station の置き場にある。止める対象に wr_obsdrop_cap77wd_*_6station_* を含む）
$guard = "$root\execute_results\hyb2_cap77wd_7station_20261009"
$g = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$guard\gpu_guard.ps1`""
    CurrentDirectory = $guard
    ProcessStartupInformation = $si
}
"guard ReturnValue=$($g.ReturnValue) PID=$($g.ProcessId)"
