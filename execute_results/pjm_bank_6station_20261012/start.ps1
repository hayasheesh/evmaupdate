# PJM の 6 station の入札（run.py）を、画面を出さずに起動する（WMI 経由）。
$py = (Get-Command python).Source
$run = 'C:/Users/admin/Desktop/EVMALOCALUPDATE/execute_results/pjm_bank_6station_20261012'
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = "`"$py`" -X utf8 -u run.py"; CurrentDirectory = $run; ProcessStartupInformation = $si }
"ReturnValue=$($r.ReturnValue) PID=$($r.ProcessId)"
