# Starts the main-line (resbid) ABG training for AEMO and ERCOT without a window (WMI), so an app restart does not stop it.
#   powershell -NoProfile -ExecutionPolicy Bypass -File start.ps1 [abg_stop]
# With the argument abg_stop, starts only the stop watcher (abg_stop.py), after the run folders exist.
$py = (Get-Command python).Source
$run = 'C:\Users\admin\Desktop\EVMALOCALUPDATE\execute_results\abg_resbid_6station_20261013'
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
if ($args -contains 'abg_stop') {
    $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = "`"$py`" -X utf8 -u abg_stop.py"; CurrentDirectory = $run; ProcessStartupInformation = $si }
    "abg_stop ReturnValue=$($r.ReturnValue) PID=$($r.ProcessId)"
    return
}
foreach ($task in @('aemo', 'ercot')) {
    $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = "`"$py`" -X utf8 -u run_pretrain.py $task"; CurrentDirectory = $run; ProcessStartupInformation = $si }
    "$task ReturnValue=$($r.ReturnValue) PID=$($r.ProcessId)"
}
