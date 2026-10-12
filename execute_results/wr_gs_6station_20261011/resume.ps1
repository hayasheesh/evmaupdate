# 6 station、大域の報酬を1/4にした版の aemo と ercot を、保存した学習状態から画面を出さずに再開する（WMI 経由。アプリの再起動で止まらない）。
#   powershell -NoProfile -ExecutionPolicy Bypass -File resume.ps1
# 止めたときの STOP_REQUESTED は消さずに名前を変えて残す。温度の見張りも起動し直す。
$py = (Get-Command python).Source
$root = 'C:\Users\admin\Desktop\EVMALOCALUPDATE'
$run = "$root\execute_results\wr_gs_6station_20261011"
$runs = @{
    'aemo'  = "$root\archive\wr_obsdrop_cap77wd_aemoplan_6station_gs025_20261011_164049"
    'ercot' = "$root\archive\wr_obsdrop_cap77wd_ercotplan_6station_gs025_20261011_164235"
}
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
foreach ($task in @('aemo', 'ercot')) {
    $flag = "$($runs[$task])\resume\STOP_REQUESTED"
    if (Test-Path $flag) { Move-Item $flag "$flag.done_$stamp" }
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
$guard = "$root\execute_results\hyb2_cap77wd_7station_20261009"
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'gpu_guard\.ps1' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
$g = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$guard\gpu_guard.ps1`""
    CurrentDirectory = $guard
    ProcessStartupInformation = $si
}
"guard ReturnValue=$($g.ReturnValue) PID=$($g.ProcessId)"
