# ABG の大域の報酬の停滞を見て止める見張り（abg_stop.py）を、画面を出さずに起動する（WMI 経由）。
$py = (Get-Command python).Source
$run = 'C:/Users/admin/Desktop/EVMALOCALUPDATE/execute_results/abg_6station_20261012'
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = "`"$py`" -X utf8 -u abg_stop.py"; CurrentDirectory = $run; ProcessStartupInformation = $si }
"ReturnValue=$($r.ReturnValue) PID=$($r.ProcessId)"
