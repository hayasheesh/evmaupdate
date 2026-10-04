param([int]$CpuRateBasisPoints = 2500)
$ErrorActionPreference = 'Stop'
$ProjectRoot = 'C:\Users\admin\Desktop\EVMALOCALUPDATE'
$RunRoot = Join-Path $ProjectRoot 'execute_results\cross_market_aemo2200_to_ercot_20260930'
$StatusPath = Join-Path $RunRoot 'status.json'
$Python = 'C:\Users\admin\AppData\Local\Programs\Python\Python310\python.exe'
if ($CpuRateBasisPoints -ne 2500) { throw 'Only the configured 25% CPU evaluation is permitted' }
if (Test-Path -LiteralPath $StatusPath) { throw 'Run status already exists; do not start a duplicate evaluation' }

Add-Type -TypeDefinition @"
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
public static class CrossMarketEvalCpuJob {
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

$state = [ordered]@{
    model = 'prod_aemoplan_AB_7station'; checkpointEpisode = 2200
    trainingMarket = 'aemo_plan_deviation'; evaluationMarket = 'ercot_plan_deviation'
    evaluationPipeline = 'marl_raw'; bidDays = 5; commandsPerDay = 24; evSeedsPerCommand = 3
    cpuRateBasisPoints = 2500; cudaVisibleDevices = ''
    startedAt = (Get-Date).ToString('o'); completedAt = $null
    launcherPid = $PID; pid = $null; stage = 'starting'; exitCode = $null; error = $null
}
function Save-State {
    $script:state | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath "$StatusPath.tmp" -Encoding UTF8
    Move-Item -LiteralPath "$StatusPath.tmp" -Destination $StatusPath -Force
}
Save-State
$env:PYTHONUTF8 = '1'
$env:CUDA_VISIBLE_DEVICES = ''
try {
    $job = [CrossMarketEvalCpuJob]::Create([uint32]$CpuRateBasisPoints)
    $proc = Start-Process -FilePath $Python -ArgumentList @('-X', 'utf8', '-u', (Join-Path $RunRoot 'evaluate.py')) `
        -WorkingDirectory $ProjectRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $RunRoot 'evaluate.stdout.log') `
        -RedirectStandardError (Join-Path $RunRoot 'evaluate.stderr.log')
    try { [CrossMarketEvalCpuJob]::Assign($job, $proc.Handle) } catch {
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        throw "Could not place evaluator under CPU cap: $_"
    }
    $state.pid = $proc.Id
    $state.stage = 'evaluating'
    Save-State
    Write-Host "[cross-market] pid=$($proc.Id) AEMO-2200 -> ERCOT holdout 5x24x3 CPU-cap=$CpuRateBasisPoints $(Get-Date -Format o)"
    $proc.WaitForExit()
    $proc.Refresh()
    $state.exitCode = $proc.ExitCode
    if ($proc.ExitCode -ne 0) { throw "evaluator failed with exit code $($proc.ExitCode)" }
    if (-not (Test-Path -LiteralPath (Join-Path $RunRoot 'summary_checked.json'))) {
        throw 'evaluator exited without a checked summary'
    }
    $state.stage = 'complete'
    $state.completedAt = (Get-Date).ToString('o')
    Save-State
} catch {
    $state.stage = 'failed'
    $state.error = [string]$_
    $state.completedAt = (Get-Date).ToString('o')
    Save-State
    throw
}
