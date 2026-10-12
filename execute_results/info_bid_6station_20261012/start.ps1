# 指令やEVを前日に知っていた場合の入札（run_all.py）を、画面を出さずに起動する（WMI 経由）。
$py = (Get-Command python).Source
$run = 'C:/Users/admin/Desktop/EVMALOCALUPDATE/execute_results/info_bid_6station_20261012'
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = "`"$py`" -X utf8 -u run_all.py"; CurrentDirectory = $run; ProcessStartupInformation = $si }
"ReturnValue=$($r.ReturnValue) PID=$($r.ProcessId)"
