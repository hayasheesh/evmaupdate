# 2026-10-13 01:15 に止めた Windows の学習5本（通常 MADDPG と ABG の AEMO・ERCOT、PJM 6 station 本線）を、
# 保存した学習状態から画面を出さずに再開する（WMI 経由。アプリの再起動で止まらない）。ABG の自動停止の見張りも起動し直す。
#   powershell -NoProfile -ExecutionPolicy Bypass -File resume_windows.ps1
# 止めたときの STOP_REQUESTED は消さずに名前を変えて残す。run_status.json は「実行中」に戻す（mark_running.py）。
$py = (Get-Command python).Source
$root = 'C:\Users\admin\Desktop\EVMALOCALUPDATE'
$er = "$root\execute_results"
$here = "$er\resume_20261013"
$jobs = @(
    @{ folder = 'maddpg_6station_20261012'; task = 'aemo';  run = 'wr_obsdrop_cap77wd_aemoplan_6station_maddpg_20261012_132458'; status = $true },
    @{ folder = 'maddpg_6station_20261012'; task = 'ercot'; run = 'wr_obsdrop_cap77wd_ercotplan_6station_maddpg_20261012_132604'; status = $true },
    @{ folder = 'abg_6station_20261012';    task = 'aemo';  run = 'wr_obsdrop_cap77wd_aemoplan_6station_abg_20261012_163638'; status = $true },
    @{ folder = 'abg_6station_20261012';    task = 'ercot'; run = 'wr_obsdrop_cap77wd_ercotplan_6station_abg_20261012_163744'; status = $true },
    @{ folder = 'pjm_bank_6station_20261012'; task = 'pjm'; run = 'wr_obsdrop_cap77wd_pjmplan_6station_gs00625_20261012_163606'; status = $false }
)
# 引数に「フォルダ:task」を並べると、その実行だけを起動する（例：maddpg_6station_20261012:aemo）。
$selected = @($args)
if ($selected.Count -gt 0) { $jobs = @($jobs | Where-Object { $selected -contains "$($_.folder):$($_.task)" }) }
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
foreach ($j in $jobs) {
    $dir = "$er\$($j.folder)"
    $runDir = "$root\archive\$($j.run)"
    $flag = "$runDir\resume\STOP_REQUESTED"
    if (Test-Path $flag) { Move-Item $flag "$flag.done_$stamp" }
    $out = "$dir\logs\$($j.task)_pretrain.stdout.log"
    $err = "$dir\logs\$($j.task)_pretrain.stderr.log"
    $cmd = "C:\Windows\System32\cmd.exe /c `"`"$py`" -X utf8 -u `"$dir\stage_worker.py`" $($j.task) pretrain --resume-run `"$runDir`" >> `"$out`" 2>> `"$err`"`""
    $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
        CommandLine = $cmd
        CurrentDirectory = $root
        ProcessStartupInformation = $si
    }
    "$($j.folder) $($j.task) ReturnValue=$($r.ReturnValue) PID=$($r.ProcessId)"
    if ($j.status) { & $py -X utf8 "$here\mark_running.py" "$dir\$($j.task)_run_status.json" $r.ProcessId $runDir }
}
$abg = "$er\abg_6station_20261012"
if (Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'abg_stop\.py' }) { 'abg_stop already running'; return }
$a = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = "`"$py`" -X utf8 -u abg_stop.py"
    CurrentDirectory = $abg
    ProcessStartupInformation = $si
}
"abg_stop ReturnValue=$($a.ReturnValue) PID=$($a.ProcessId)"
