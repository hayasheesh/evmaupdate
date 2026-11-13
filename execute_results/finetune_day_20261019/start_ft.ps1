# Starts the finetune trainings (day and control, 1000 episodes each) and the GPU watcher ft_watch.py
# without a window (WMI), so an app restart does not stop them. 2026-10-19.
#   powershell -NoProfile -ExecutionPolicy Bypass -File start_ft.ps1
$py = (Get-Command python).Source
$dir = 'C:\Users\admin\Desktop\EVMALOCALUPDATE\execute_results\finetune_day_20261019'
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
foreach ($mode in @('day', 'control')) {
    $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = "`"$py`" -X utf8 -u run_ft.py $mode --episodes 1000"; CurrentDirectory = $dir; ProcessStartupInformation = $si }
    "$mode ReturnValue=$($r.ReturnValue) PID=$($r.ProcessId)"
}
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = "`"$py`" -X utf8 -u ft_watch.py"; CurrentDirectory = $dir; ProcessStartupInformation = $si }
"watch ReturnValue=$($r.ReturnValue) PID=$($r.ProcessId)"
