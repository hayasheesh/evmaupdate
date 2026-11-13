# Starts the Windows part of the real-day finetune (average and control, 1000 episodes each) and the evaluation queue,
# without a window (WMI), so an app restart does not stop them. 2026-10-19.
#   & .\start_windows.ps1
$py = (Get-Command python).Source
$dir = 'C:\Users\admin\Desktop\EVMALOCALUPDATE\execute_results\realday_20261019'
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
foreach ($mode in @('average', 'control')) {
    $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = "`"$py`" -X utf8 -u run_rft.py $mode --episodes 1000"; CurrentDirectory = $dir; ProcessStartupInformation = $si }
    "$mode ReturnValue=$($r.ReturnValue) PID=$($r.ProcessId)"
}
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = "`"$py`" -X utf8 -u rft_eval_queue.py windows"; CurrentDirectory = $dir; ProcessStartupInformation = $si }
"queue ReturnValue=$($r.ReturnValue) PID=$($r.ProcessId)"
