# 7 station の aemo の本線を、観測から全体のずれ・入札の先読みと指令の大きさ・帯の幅を外して、画面を出さずに起動する（WMI 経由）。
#   powershell -NoProfile -ExecutionPolicy Bypass -File start.ps1
# 再開するときは -Resume を付ける（archive の最新の obsdrop_cap77wd_aemoplan_7station_* から続ける）。
param([switch]$Resume)
$py = (Get-Command python).Source
$root = 'C:\Users\admin\Desktop\EVMALOCALUPDATE'
$run = "$root\execute_results\obsdrop_cap77wd_7station_20261010"
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
$task = 'aemo'
if ($Resume) {
    $dir = (Get-ChildItem "$root\archive" -Directory | Where-Object { $_.Name -like 'obsdrop_cap77wd_aemoplan_7station_*' } | Sort-Object Name | Select-Object -Last 1).FullName
    $out = "$run\logs\${task}_pretrain_resume.stdout.log"
    $err = "$run\logs\${task}_pretrain_resume.stderr.log"
    $cmd = "C:\Windows\System32\cmd.exe /c `"`"$py`" -X utf8 -u `"$run\stage_worker.py`" $task pretrain --resume-run `"$dir`" >> `"$out`" 2>> `"$err`"`""
} else {
    $out = "$run\logs\${task}_pretrain.stdout.log"
    $err = "$run\logs\${task}_pretrain.stderr.log"
    $cmd = "C:\Windows\System32\cmd.exe /c `"`"$py`" -X utf8 -u `"$run\stage_worker.py`" $task pretrain >> `"$out`" 2>> `"$err`"`""
}
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = $cmd
    CurrentDirectory = $root
    ProcessStartupInformation = $si
}
"$task ReturnValue=$($r.ReturnValue) PID=$($r.ProcessId)"
