# 温度の見張り（gpu_guard.ps1）だけを起動し直す（WMI 経由、画面なし）。
$guard = 'C:/Users/admin/Desktop/EVMALOCALUPDATE/execute_results/hyb2_cap77wd_7station_20261009'
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'gpu_guard' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
$g = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$guard\gpu_guard.ps1`""; CurrentDirectory = $guard; ProcessStartupInformation = $si }
"guard ReturnValue=$($g.ReturnValue) PID=$($g.ProcessId)"
