# Windows の GPU と CPU の温度、水冷ポンプの回転数を30秒ごとに記録し、危ないときは7 stationの学習（hyb2）を止める（WMI 経由で画面なしに動かす）。
# CPU の温度とポンプの回転数は LibreHardwareMonitor（C:\Users\admin\Tools\LibreHardwareMonitor_v0.9.6、管理者で起動）が
# http://127.0.0.1:8085/data.json に出す値を読む。読めないときは記録に空欄を残し、それだけでは止めない。
# ポンプは、マザーボードの Fan #6（/lpc/nct6798d/0/fan/5、出力100%固定で約3,000 rpm）と見ている（未確認）。
#   powershell -NoProfile -ExecutionPolicy Bypass -File gpu_guard.ps1
# 止める条件（利用者の指示：水冷が不調かもしれないので、70 °C 前後か上がり続けたら止める）
#   - 70 °C 以上
#   - 直近2分（4回）の中央値が 60 °C 以上で、直近20分の最低から 8 °C 以上上がっている
#   - 直近20分の最低が 58 °C 以上（下がらずに高いまま）
#   - CPU 本体（Package）が 80 °C 以上、直近2分の中央値が 70 °C 以上、または直近2分の中央値が 60 °C 以上で
#     直近20分の最低から 15 °C 以上上がっている。学習中は 47〜50 °C（2026-10-10 に測定、TjMax は約100 °C）
#   - ポンプの回転数が2回続けて 1,000 rpm 未満
# ときどき電力が170〜195 Wまで上がり（利用者が同じGPUで別の作業をしていた時刻）、温度が30秒〜1分だけ57〜62 °Cになる（通常は49〜50 °C）。
# この一時的な上昇では止めない。
# 止め方：学習に停止を依頼する（今のエピソードの後で状態を保存して止まる）。2分たっても学習が残っているか、
# 75 °C 以上になったら、学習のプロセスを強制終了する。
$root = 'C:\Users\admin\Desktop\EVMALOCALUPDATE'
$run = "$root\execute_results\hyb2_cap77wd_7station_20261009"
$log = "$run\guard_temperature.csv"
$events = "$run\gpu_guard_events.log"
function Write-Line($path, $text) {
    $fs = [System.IO.File]::Open($path, [System.IO.FileMode]::Append, [System.IO.FileAccess]::Write, [System.IO.FileShare]::ReadWrite)
    $sw = New-Object System.IO.StreamWriter($fs, [System.Text.Encoding]::UTF8)
    $sw.WriteLine($text); $sw.Close()
}
function Get-Trainers {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match '(hyb2_cap77wd_7station_20261009|obsdrop_cap77wd_7station_20261010|wr_cap77wd_6station_20261010|wr_gs_6station_20261011|maddpg_6station_20261012|pjm_bank_6station_20261012|abg_6station_20261012|resbid_6station_20261012)\\stage_worker.py' }
}
function Get-Lhm {
    try { $j = Invoke-RestMethod -Uri 'http://127.0.0.1:8085/data.json' -TimeoutSec 5 } catch { return @($null, $null) }
    $vals = @{}
    $stack = New-Object System.Collections.Stack; $stack.Push($j)
    while ($stack.Count -gt 0) {
        $n = $stack.Pop()
        if ($n.SensorId) { $vals[$n.SensorId] = $n.Value }
        foreach ($c in $n.Children) { $stack.Push($c) }
    }
    $num = { param($v) if ($v -match '([-0-9.]+)') { [double]$Matches[1] } else { $null } }
    return @((& $num $vals['/intelcpu/0/temperature/26']), (& $num $vals['/lpc/nct6798d/0/fan/5']))
}
if (-not (Test-Path $log)) { Write-Line $log 'time,gpu_c,util_pct,power_w,cpu_package_c,pump_rpm' }
Write-Line $events "$(Get-Date -Format s) guard started"
$hist = New-Object System.Collections.Generic.List[double]
$cpuHist = New-Object System.Collections.Generic.List[double]
$pumpLow = 0
while ($true) {
    $q = (& nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,power.draw --format=csv,noheader,nounits) -split ','
    $t = [double]$q[0].Trim()
    $cpu, $pump = Get-Lhm
    Write-Line $log ("{0},{1},{2},{3},{4},{5}" -f (Get-Date -Format s), $t, $q[1].Trim(), $q[2].Trim(), $cpu, $pump)
    if ($cpu -ne $null) { $cpuHist.Add($cpu); if ($cpuHist.Count -gt 40) { $cpuHist.RemoveAt(0) } }
    $cpuLast4 = @($cpuHist | Select-Object -Last 4 | Sort-Object)
    $cpuMed = if ($cpuLast4.Count -ge 4) { ($cpuLast4[1] + $cpuLast4[2]) / 2 } else { 0 }
    $cpuMin = if ($cpuHist.Count -gt 0) { ($cpuHist | Measure-Object -Minimum).Minimum } else { 0 }
    if ($pump -ne $null -and $pump -lt 1000) { $pumpLow += 1 } elseif ($pump -ne $null) { $pumpLow = 0 }
    $hist.Add($t); if ($hist.Count -gt 40) { $hist.RemoveAt(0) }
    $minRecent = ($hist | Measure-Object -Minimum).Minimum
    $last4 = @($hist | Select-Object -Last 4 | Sort-Object)
    $med = if ($last4.Count -ge 4) { ($last4[1] + $last4[2]) / 2 } else { 0 }
    $reason = $null
    if ($t -ge 70) { $reason = "temperature $t C >= 70" }
    elseif ($med -ge 60 -and ($med - $minRecent) -ge 8) { $reason = "2-min median $med C rose $($med - $minRecent) C in 20 min" }
    elseif ($hist.Count -ge 40 -and $minRecent -ge 58) { $reason = "20-min minimum $minRecent C >= 58" }
    elseif ($cpu -ne $null -and $cpu -ge 80) { $reason = "CPU package $cpu C >= 80" }
    elseif ($cpuMed -ge 70) { $reason = "CPU package 2-min median $cpuMed C >= 70" }
    elseif ($cpuMed -ge 60 -and ($cpuMed - $cpuMin) -ge 15) { $reason = "CPU package 2-min median $cpuMed C rose $($cpuMed - $cpuMin) C in 20 min" }
    elseif ($pumpLow -ge 2) { $reason = "pump $pump rpm < 1000 twice" }
    if ($reason) {
        Write-Line $events "$(Get-Date -Format s) EMERGENCY STOP: $reason"
        $runs = Get-ChildItem "$root\archive" -Directory | Where-Object { $_.Name -like 'hyb2_cap77wd_*_7station_*' -or $_.Name -like 'obsdrop_cap77wd_*_7station_*' -or $_.Name -like 'wr_cap77wd_*_6station_*' -or $_.Name -like 'wr_obsdrop_cap77wd_*_6station_*' }
        foreach ($r in $runs) { New-Item -ItemType File -Force "$($r.FullName)\resume\STOP_REQUESTED" | Out-Null; Write-Line $events "$(Get-Date -Format s) stop requested for $($r.Name)" }
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
