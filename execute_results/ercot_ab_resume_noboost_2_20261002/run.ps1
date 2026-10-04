param(
    [int] $Episodes = 2000,
    [int] $CpuRateBasisPoints = 10000
)
# ERCOT AB を保存済みの完全な状態から2000回まで続ける（fast_resume.py は
# ercot_ab_fast_resume_20261001 と同じもの）。
# 電源設定（boost）には触らない。起動時と終了時の値を読んで記録するだけ。
$ErrorActionPreference = 'Stop'
$ProjectRoot = 'C:\Users\admin\Desktop\EVMALOCALUPDATE'
$RunRoot = Join-Path $ProjectRoot 'execute_results\ercot_ab_resume_noboost_2_20261002'
$RunDir = Join-Path $ProjectRoot 'archive\prod_ercotplan_AB_7station_20260930_221551'
$Python = 'C:\Users\admin\AppData\Local\Programs\Python\Python310\python.exe'
$StatusPath = Join-Path $RunRoot 'status.json'
if (Test-Path -LiteralPath $StatusPath) { throw 'Run status already exists; do not start a duplicate run' }
if (Test-Path -LiteralPath (Join-Path $RunDir 'resume\STOP_REQUESTED')) { throw 'STOP_REQUESTED is still present in the run directory' }

Add-Type -TypeDefinition @"
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
public static class ErcotNoBoostCpuJob {
    [StructLayout(LayoutKind.Sequential)]
    public struct CpuRateControlInformation { public uint ControlFlags; public uint CpuRate; }
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern IntPtr CreateJobObjectW(IntPtr a, string n);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool SetInformationJobObject(IntPtr h, int c, ref CpuRateControlInformation i, uint s);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool AssignProcessToJobObject(IntPtr h, IntPtr p);
    public static IntPtr Create(uint basisPoints) {
        IntPtr job = CreateJobObjectW(IntPtr.Zero, null);
        if (job == IntPtr.Zero) throw new Win32Exception(Marshal.GetLastWin32Error());
        var info = new CpuRateControlInformation();
        info.ControlFlags = 0x1 | 0x4;
        info.CpuRate = basisPoints;
        if (!SetInformationJobObject(job, 15, ref info, (uint)Marshal.SizeOf(typeof(CpuRateControlInformation))))
            throw new Win32Exception(Marshal.GetLastWin32Error());
        return job;
    }
    public static void Assign(IntPtr job, IntPtr process) {
        if (!AssignProcessToJobObject(job, process)) throw new Win32Exception(Marshal.GetLastWin32Error());
    }
}
"@

function Read-AcBoostMode {
    $output = & powercfg /query SCHEME_CURRENT SUB_PROCESSOR PERFBOOSTMODE
    $line = $output | Select-String -Pattern '0x[0-9a-fA-F]+' | Select-Object -Last 2 | Select-Object -First 1
    $match = [regex]::Match([string]$line, '0x[0-9a-fA-F]+')
    if (-not $match.Success) { return $null }
    return [Convert]::ToInt32($match.Value.Substring(2), 16)
}

$latest = Get-Content -LiteralPath (Join-Path $RunDir 'resume\latest.json') -Raw | ConvertFrom-Json
$env:PYTHONUTF8 = '1'
$state = [ordered]@{
    modelName = 'prod_ercotplan_AB_7station'; runDir = $RunDir
    sourceDir = (Join-Path $ProjectRoot 'execute_results\ercot_ab_2000_20260930\runtime_source')
    startEpisode = $latest.completed_training_episode; targetEpisode = $Episodes; exactResume = $true
    cpuRateBasisPoints = $CpuRateBasisPoints; powerSettingsTouched = $false
    acBoostModeAtStart = (Read-AcBoostMode); acBoostModeAtEnd = $null
    startedAt = (Get-Date).ToString('o'); completedAt = $null; launcherPid = $PID; pid = $null
    stage = 'starting'; exitCode = $null; error = $null
}
function Save-State {
    $script:state | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath "$StatusPath.tmp" -Encoding UTF8
    Move-Item -LiteralPath "$StatusPath.tmp" -Destination $StatusPath -Force
}
Save-State
try {
    $job = [ErcotNoBoostCpuJob]::Create([uint32]$CpuRateBasisPoints)
    $arguments = @('-X', 'utf8', '-u', (Join-Path $RunRoot 'fast_resume.py'), '--episodes', "$Episodes")
    $proc = Start-Process -FilePath $Python -ArgumentList $arguments -WorkingDirectory $ProjectRoot `
        -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $RunRoot 'pretrain.stdout.log') `
        -RedirectStandardError (Join-Path $RunRoot 'pretrain.stderr.log')
    try { [ErcotNoBoostCpuJob]::Assign($job, $proc.Handle) } catch {
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        throw "Could not place learner under CPU cap: $_"
    }
    $state.pid = $proc.Id
    $state.stage = 'training'
    Save-State
    Write-Host "[ERCOT AB resume] pid=$($proc.Id) checkpoint=$($state.startEpisode) target=$Episodes cap=$CpuRateBasisPoints $(Get-Date -Format o)"
    $proc.WaitForExit()
    $proc.Refresh()
    $state.exitCode = $proc.ExitCode
    if ($proc.ExitCode -ne 0) { throw "learner failed with exit code $($proc.ExitCode)" }
    $latest = Get-Content -LiteralPath (Join-Path $RunDir 'resume\latest.json') -Raw | ConvertFrom-Json
    $state.stage = if ($latest.completed_training_episode -ge $Episodes) { 'complete' } else { 'stopped' }
} catch {
    $state.stage = 'failed'
    $state.error = [string]$_
    throw
} finally {
    $state.completedAt = (Get-Date).ToString('o')
    $state.acBoostModeAtEnd = (Read-AcBoostMode)
    Save-State
    Write-Host "[ERCOT AB resume] stage=$($state.stage) $(Get-Date -Format o)"
}

