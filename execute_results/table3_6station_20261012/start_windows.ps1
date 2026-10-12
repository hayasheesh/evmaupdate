# 表3と図4の 6 station の計算を、蓄電池の評価が終わってから始める見張り役ごと、画面を出さずに起動する（WMI 経由）。
$py = (Get-Command python).Source
$run = 'C:\Users\admin\Desktop\EVMALOCALUPDATE\execute_results\table3_6station_20261012'
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = "`"$py`" -X utf8 -u run_windows.py"
    CurrentDirectory = $run
    ProcessStartupInformation = $si
}
"ReturnValue=$($r.ReturnValue) PID=$($r.ProcessId)"
