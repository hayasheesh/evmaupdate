param(
    [int] $Episodes = 2000,
    [int] $CpuRateBasisPoints = 7500
)
# ERCOT の 7 station で、標準 MADDPG を新規に2000回学習する（worker.py）。AB 学習（ercot_ab_2000_20260930）との違いは学習の方式だけ。
# Job Object で論理32 CPUの75%に抑える。電源設定（boost）には触らない。起動時と終了時の値を読んで記録するだけ。
$ErrorActionPreference = 'Stop'
$ProjectRoot = 'C:\Users\admin\Desktop\EVMALOCALUPDATE'
$RunRoot = Join-Path $ProjectRoot 'execute_results\ercot_std_maddpg_20261003'
$Python = 'C:\Users\admin\AppData\Local\Programs\Python\Python310\python.exe'
$StatusPath = Join-Path $RunRoot 'status.json'
$RunDirMarker = Join-Path $RunRoot 'run_dir.json'
if (Test-Path -LiteralPath $StatusPath) { throw 'Run status already exists; do not start a duplicate run' }

Add-Type -TypeDefinition @"
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
public static class ErcotStdMaddpgCpuJob {
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

$env:PYTHONUTF8 = '1'
$state = [ordered]@{
    modelName = 'prod_ercotplan_MADDPGstd_7station'; algorithm = 'maddpg'; runDir = $null
    targetEpisode = $Episodes; cpuRateBasisPoints = $CpuRateBasisPoints; powerSettingsTouched = $false
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
    $job = [ErcotStdMaddpgCpuJob]::Create([uint32]$CpuRateBasisPoints)
    $arguments = @('-X', 'utf8', '-u', (Join-Path $RunRoot 'worker.py'), '--episodes', "$Episodes")
    $proc = Start-Process -FilePath $Python -ArgumentList $arguments -WorkingDirectory $ProjectRoot `
        -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $RunRoot 'pretrain.stdout.log') `
        -RedirectStandardError (Join-Path $RunRoot 'pretrain.stderr.log')
    try { [ErcotStdMaddpgCpuJob]::Assign($job, $proc.Handle) } catch {
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        throw "Could not place learner under CPU cap: $_"
    }
    $state.pid = $proc.Id
    $state.stage = 'training'
    Save-State
    Write-Host "[ERCOT std MADDPG] pid=$($proc.Id) target=$Episodes cap=$CpuRateBasisPoints $(Get-Date -Format o)"
    while (-not $proc.WaitForExit(5000)) {
        if (-not $state.runDir -and (Test-Path -LiteralPath $RunDirMarker)) {
            $state.runDir = (Get-Content -LiteralPath $RunDirMarker -Raw | ConvertFrom-Json).runDir
            Save-State
        }
    }
    $proc.Refresh()
    $state.exitCode = $proc.ExitCode
    if ($proc.ExitCode -ne 0) { throw "learner failed with exit code $($proc.ExitCode)" }
    if (-not $state.runDir -and (Test-Path -LiteralPath $RunDirMarker)) {
        $state.runDir = (Get-Content -LiteralPath $RunDirMarker -Raw | ConvertFrom-Json).runDir
    }
    $latest = Get-Content -LiteralPath (Join-Path $state.runDir 'resume\latest.json') -Raw | ConvertFrom-Json
    $state.stage = if ($latest.completed_training_episode -ge $Episodes) { 'complete' } else { 'stopped' }
} catch {
    $state.stage = 'failed'
    $state.error = [string]$_
    throw
} finally {
    $state.completedAt = (Get-Date).ToString('o')
    $state.acBoostModeAtEnd = (Read-AcBoostMode)
    Save-State
    Write-Host "[ERCOT std MADDPG] stage=$($state.stage) $(Get-Date -Format o)"
}
