param([int] $Hours = 10, [int] $IntervalSeconds = 30)
# Windows の GPU（RTX 4080）の状態を一定間隔で CSV に追記するだけ。何も止めない・変えない。
# 他のプロセスが読んでいても書けるように、追記のたびに共有読み書きでファイルを開いて閉じる。
$Log = 'C:\Users\admin\Desktop\EVMALOCALUPDATE\execute_results\watch_20261002\gpu_log.csv'
$Err = 'C:\Users\admin\Desktop\EVMALOCALUPDATE\execute_results\watch_20261002\gpu_logger_errors.log'
$Smi = 'C:\Windows\System32\nvidia-smi.exe'
$Utf8 = New-Object System.Text.UTF8Encoding($false)
function Append-Line([string] $Path, [string] $Line) {
    $fs = [System.IO.File]::Open($Path, [System.IO.FileMode]::Append, [System.IO.FileAccess]::Write, [System.IO.FileShare]::ReadWrite)
    try { $bytes = $Utf8.GetBytes($Line + "`n"); $fs.Write($bytes, 0, $bytes.Length) } finally { $fs.Close() }
}
if (-not (Test-Path -LiteralPath $Log)) {
    Append-Line $Log 'time,temperature_c,utilization_pct,memory_used_mib,power_w,fan_pct,clock_sm_mhz,throttle_reasons'
}
$end = (Get-Date).AddHours($Hours)
while ((Get-Date) -lt $end) {
    $stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    try {
        $row = & $Smi --query-gpu=temperature.gpu,utilization.gpu,memory.used,power.draw,fan.speed,clocks.sm,clocks_throttle_reasons.active --format=csv,noheader,nounits 2>&1
        if ($LASTEXITCODE -eq 0) { Append-Line $Log "$stamp,$(($row -join ' ') -replace '\s*,\s*', ',')" }
        else { Append-Line $Log "$stamp,ERROR,$(($row -join ' ') -replace ',', ' ')" }
    } catch {
        try { Append-Line $Err "$stamp $($_.Exception.Message)" } catch { }
    }
    Start-Sleep -Seconds $IntervalSeconds
}
