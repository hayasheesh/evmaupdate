param(
    [int[]] $WaitPids = @(39436),
    [string] $WaitCommandPattern = 'ft_all\.sh|ft_regen\.py',
    [int] $QuietMinutes = 5
)
# 利用者の GPU の処理（ft_all.sh）が終わるのを待ってから、GB の学習を保存済みの完全な状態から再開する。
# 指定の pid が消え、さらに一致するコマンドのプロセスが QuietMinutes 分続けて無いことを確かめてから起動する。
# 電源設定（boost）には触らない。
$ErrorActionPreference = 'Stop'
$ProjectRoot = 'C:\Users\admin\Desktop\EVMALOCALUPDATE'
$RunRoot = Join-Path $ProjectRoot 'execute_results\gb_128cmd_marl_20261002'
$RunDir = Join-Path $ProjectRoot 'archive\prod_elexonplan_AB_7station_20261002_215844'
$StatePath = Join-Path $RunRoot 'wait_status.json'
$PowerShell = 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe'
$state = [ordered]@{
    waitPids = $WaitPids; waitCommandPattern = $WaitCommandPattern; quietMinutes = $QuietMinutes
    launcherPid = $PID; startedAt = (Get-Date).ToString('o'); stage = 'waiting_for_learner_stop'
    stageAt = (Get-Date).ToString('o'); resumeFromEpisode = $null; error = $null
}
function Save-State([string] $Stage) {
    $state.stage = $Stage; $state.stageAt = (Get-Date).ToString('o')
    $state | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath "$StatePath.tmp" -Encoding UTF8
    Move-Item -LiteralPath "$StatePath.tmp" -Destination $StatePath -Force
}
function Test-UserGpuJob {
    if ($WaitPids | Where-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue }) { return $true }
    $hit = Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -ne $PID -and $_.CommandLine -match $WaitCommandPattern }
    return [bool]$hit
}
try {
    Save-State 'waiting_for_learner_stop'
    $old = Get-Content -LiteralPath (Join-Path $RunRoot 'resume_status.json') -Raw | ConvertFrom-Json
    while ($old.pid -and (Get-Process -Id $old.pid -ErrorAction SilentlyContinue)) { Start-Sleep -Seconds 15 }
    $latest = Get-Content -LiteralPath (Join-Path $RunDir 'resume\latest.json') -Raw | ConvertFrom-Json
    $state.resumeFromEpisode = $latest.completed_training_episode
    if (Test-Path -LiteralPath (Join-Path $RunDir 'resume\STOP_REQUESTED')) { throw 'learner exited but STOP_REQUESTED is still present' }
    # 前回の起動の記録を残して、resume.ps1 が新しく書けるようにする
    $tag = 'stopped_' + (Get-Date -Format 'MMdd_HHmm')
    foreach ($name in 'resume_status.json', 'resume.stdout.log', 'resume.stderr.log') {
        $p = Join-Path $RunRoot $name
        if (Test-Path -LiteralPath $p) { Move-Item -LiteralPath $p -Destination (Join-Path $RunRoot "${tag}_$name") }
    }
    Save-State 'waiting_for_user_gpu_job'
    $quietSince = $null
    while ($true) {
        if (Test-UserGpuJob) { $quietSince = $null }
        elseif (-not $quietSince) { $quietSince = Get-Date }
        elseif (((Get-Date) - $quietSince).TotalMinutes -ge $QuietMinutes) { break }
        Start-Sleep -Seconds 30
    }
    Save-State 'resume_started'
    & $PowerShell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $RunRoot 'resume.ps1')
    Save-State 'resume_exited'
} catch {
    $state.error = [string]$_
    Save-State 'failed'
    throw
}
