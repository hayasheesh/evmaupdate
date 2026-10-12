# Windows の GPU 温度を30秒ごとに記録し、危ないときは7 stationの学習を止める（WMI 経由で画面なしに動かす）。
#   powershell -NoProfile -ExecutionPolicy Bypass -File gpu_guard.ps1
# 止める条件（利用者の指示：水冷が不調かもしれないので、70 °C 前後か上がり続けたら止める）
#   - 70 °C 以上
#   - 60 °C 以上で、直近20分の最低から 8 °C 以上上がっている
# 止め方：学習に停止を依頼する（今のエピソードの後で状態を保存して止まる）。2分たっても学習が残っているか、
# 75 °C 以上になったら、学習のプロセスを強制終了する。
$root = 'C:\Users\admin\Desktop\EVMALOCALUPDATE'
$run = "$root\execute_results\hyb_cap77wd_7station_20261009"
$log = "$run\gpu_guard_temperature.csv"
$events = "$run\gpu_guard_events.log"
$runs = @(
    "$root\archive\hyb_cap77wd_aemoplan_7station_20261009_194224",
    "$root\archive\hyb_cap77wd_ercotplan_7station_20261009_194423"
)
function Write-Line($path, $text) {
    $fs = [System.IO.File]::Open($path, [System.IO.FileMode]::Append, [System.IO.FileAccess]::Write, [System.IO.FileShare]::ReadWrite)
    $sw = New-Object System.IO.StreamWriter($fs, [System.Text.Encoding]::UTF8)
    $sw.WriteLine($text); $sw.Close()
}
function Get-Trainers {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'hyb_cap77wd_7station_20261009\\stage_worker.py' }
}
if (-not (Test-Path $log)) { Write-Line $log 'time,temp_c,util_pct,power_w' }
Write-Line $events "$(Get-Date -Format s) guard started"
$hist = New-Object System.Collections.Generic.List[double]
while ($true) {
    $q = (& nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,power.draw --format=csv,noheader,nounits) -split ','
    $t = [double]$q[0].Trim()
    Write-Line $log ("{0},{1},{2},{3}" -f (Get-Date -Format s), $t, $q[1].Trim(), $q[2].Trim())
    $hist.Add($t); if ($hist.Count -gt 40) { $hist.RemoveAt(0) }
    $minRecent = ($hist | Measure-Object -Minimum).Minimum
    $reason = $null
    if ($t -ge 70) { $reason = "temperature $t C >= 70" }
    elseif ($t -ge 60 -and ($t - $minRecent) -ge 8) { $reason = "temperature $t C rose $($t - $minRecent) C in 20 min" }
    if ($reason) {
        Write-Line $events "$(Get-Date -Format s) EMERGENCY STOP: $reason"
        foreach ($r in $runs) { New-Item -ItemType File -Force "$r\resume\STOP_REQUESTED" | Out-Null }
        $deadline = (Get-Date).AddSeconds(120)
        while ((Get-Date) -lt $deadline) {
            Start-Sleep -Seconds 10
            $t2 = [double]((& nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits).Trim())
            Write-Line $log ("{0},{1},," -f (Get-Date -Format s), $t2)
            if (-not (Get-Trainers)) { break }
            if ($t2 -ge 75) { Write-Line $events "$(Get-Date -Format s) $t2 C >= 75, killing now"; break }
        }
        $left = Get-Trainers
        foreach ($p in $left) { taskkill /T /F /PID $p.ProcessId | Out-Null; Write-Line $events "$(Get-Date -Format s) killed PID $($p.ProcessId)" }
        if (-not $left) { Write-Line $events "$(Get-Date -Format s) trainers stopped after saving" }
        Set-Content -Path "$run\GPU_GUARD_STOPPED.txt" -Value "$(Get-Date -Format s) $reason" -Encoding utf8
        break
    }
    Start-Sleep -Seconds 30
}
