# Starts the count-world finetune (1000 episodes), and restarts the GPU watcher ft_watch.py and the evaluation queue
# ft_eval_queue.py, without a window (WMI), so an app restart does not stop them. 2026-10-19.
#   & .\start_count.ps1
$py = (Get-Command python).Source
$dir = 'C:\Users\admin\Desktop\EVMALOCALUPDATE\execute_results\finetune_day_20261019'
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*ft_watch.py*' -or $_.CommandLine -like '*ft_eval_queue.py*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Confirm:$false; "stopped old $($_.ProcessId)" }
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = "`"$py`" -X utf8 -u run_ft.py count --episodes 1000"; CurrentDirectory = $dir; ProcessStartupInformation = $si }
"count ReturnValue=$($r.ReturnValue) PID=$($r.ProcessId)"
foreach ($s in @('ft_watch.py', 'ft_eval_queue.py')) {
    $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = "`"$py`" -X utf8 -u $s"; CurrentDirectory = $dir; ProcessStartupInformation = $si }
    "$s ReturnValue=$($r.ReturnValue) PID=$($r.ProcessId)"
}
