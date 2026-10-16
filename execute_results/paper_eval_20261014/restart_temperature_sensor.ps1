$ErrorActionPreference = 'Stop'
$monitorPath = 'C:\Users\admin\Tools\LibreHardwareMonitor_v0.9.6\app\LibreHardwareMonitor.exe'
if (Get-Process -Name LibreHardwareMonitor -ErrorAction SilentlyContinue) {
    exit 0
}
if (-not (Test-Path -LiteralPath $monitorPath -PathType Leaf)) {
    throw "Hardware monitor executable not found: $monitorPath"
}
Start-Process -FilePath $monitorPath -WorkingDirectory (Split-Path -LiteralPath $monitorPath) -Verb RunAs -WindowStyle Hidden
