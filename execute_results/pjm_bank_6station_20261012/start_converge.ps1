# 収束の見張りと PJM 6 station の学習の起動役（converge_then_train.py）を、画面を出さずに起動する（WMI 経由）。
$py = (Get-Command python).Source
$run = 'C:/Users/admin/Desktop/EVMALOCALUPDATE/execute_results/pjm_bank_6station_20261012'
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = "`"$py`" -X utf8 -u converge_then_train.py"; CurrentDirectory = $run; ProcessStartupInformation = $si }
"ReturnValue=$($r.ReturnValue) PID=$($r.ProcessId)"
