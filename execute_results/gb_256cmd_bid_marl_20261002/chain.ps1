param(
    [int] $WaitForPid = 0,
    [int] $CpuRateBasisPoints = 7500,
    [int] $DayWorkers = 6,
    [int] $ScenarioWorkers = 4
)
# GB（BOA − FPN）の入札バンク（設計指令256本、EV 3本、学習用25日・検証用5日）を作り、
# 完成したら同じ入札でAB学習2000回へ進む。
# -WaitForPid を渡すと、そのプロセス（ERCOT学習の起動スクリプト）の終了を待ってから始める。
# 入札とMARLは同じ Job Object に入れ、論理32 CPUの75%に抑える。電源設定（boost）は触らない。
$ErrorActionPreference = 'Stop'
$ProjectRoot = 'C:\Users\admin\Desktop\EVMALOCALUPDATE'
$RunRoot = Join-Path $ProjectRoot 'execute_results\gb_256cmd_bid_marl_20261002'
$Python = 'C:\Users\admin\AppData\Local\Programs\Python\Python310\python.exe'
$StatusPath = Join-Path $RunRoot 'status.json'
$TrainBank = Join-Path $ProjectRoot 'execute_results\bid_banks\train_25_minmedmax_3of128ev_256cmd_all_commands_elexon_plan_deviation'
$TestBank = Join-Path $ProjectRoot 'execute_results\bid_banks\validation_5_minmedmax_3of128ev_256cmd_all_commands_elexon_plan_deviation'
if (Test-Path -LiteralPath $StatusPath) { throw 'Run status already exists; do not start a duplicate chain' }

Add-Type -TypeDefinition @"
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
public static class GbChainCpuJob {
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
    signalSet = 'elexon_plan_deviation'; activationScenarios = 256; fixedEvScenarios = 3
    evScenarioCandidateCount = 128; trainDays = 25; validationDays = 5
    dayWorkers = $DayWorkers; scenarioWorkersPerDay = $ScenarioWorkers
    cpuRateBasisPoints = $CpuRateBasisPoints; powerSettingsTouched = $false
    trainBank = $TrainBank; validationBank = $TestBank
    model = 'prod_elexonplan256_AB_7station'; targetEpisode = 2000
    waitForPid = $WaitForPid; launcherPid = $PID
    startedAt = (Get-Date).ToString('o'); completedAt = $null
    stage = 'waiting'; pid = $null; exitCode = $null; error = $null; stageStartedAt = @{}
}
function Save-State {
    $script:state | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath "$StatusPath.tmp" -Encoding UTF8
    Move-Item -LiteralPath "$StatusPath.tmp" -Destination $StatusPath -Force
}
Save-State

try {
    if ($WaitForPid -gt 0) {
        Write-Host "[GB chain] waiting for pid $WaitForPid $(Get-Date -Format o)"
        while (Get-Process -Id $WaitForPid -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 30 }
        Write-Host "[GB chain] pid $WaitForPid exited $(Get-Date -Format o)"
    }
    $env:PYTHONUTF8 = '1'
    $env:OMP_NUM_THREADS = '1'
    $env:MKL_NUM_THREADS = '1'
    $env:OPENBLAS_NUM_THREADS = '1'
    Get-ChildItem Env: | Where-Object { $_.Name -like 'EVMA_*' } | ForEach-Object { Remove-Item "Env:$($_.Name)" }
    $env:EVMA_ACTIVATION_SIGNAL_SET = 'elexon_plan_deviation'
    $env:EVMA_LOWER_TRAIN_UPPER_BID_ACTIVATION_SCENARIOS = '256'
    $env:EVMA_LOWER_TRAIN_UPPER_BID_EV_SCENARIO_CANDIDATES = '128'

    $job = [GbChainCpuJob]::Create([uint32]$CpuRateBasisPoints)
    $stages = @(
        @{ Name = 'bid_preflight'; Args = @('-u', 'tools/build_all_upper_bid_banks.py', '--activation-signal-set', 'elexon_plan_deviation', '--preflight-only') },
        @{ Name = 'bid_train'; Args = @('-u', 'tools/build_training_bid_bank.py', '--split', 'train', '--days', '25', '--train-split-count', '25', '--paired-train-days', '25', '--paired-test-days', '5', '--workers', "$DayWorkers", '--scenario-workers', "$ScenarioWorkers", '--output-dir', $TrainBank) },
        @{ Name = 'bid_test'; Args = @('-u', 'tools/build_training_bid_bank.py', '--split', 'test', '--days', '5', '--train-split-count', '25', '--paired-train-days', '25', '--paired-test-days', '5', '--workers', "$DayWorkers", '--scenario-workers', "$ScenarioWorkers", '--output-dir', $TestBank) },
        @{ Name = 'marl_preflight'; Args = @('-X', 'utf8', '-u', (Join-Path $RunRoot 'worker.py'), '--episodes', '2000', '--preflight-only') },
        @{ Name = 'pretrain'; Args = @('-X', 'utf8', '-u', (Join-Path $RunRoot 'worker.py'), '--episodes', '2000') }
    )
    foreach ($stage in $stages) {
        $state.stage = $stage.Name
        $state.stageStartedAt[$stage.Name] = (Get-Date).ToString('o')
        if ($stage.Name -like 'marl*' -or $stage.Name -eq 'pretrain') {
            Get-ChildItem Env: | Where-Object { $_.Name -like 'EVMA_*' } | ForEach-Object { Remove-Item "Env:$($_.Name)" }
        }
        $proc = Start-Process -FilePath $Python -ArgumentList $stage.Args -WorkingDirectory $ProjectRoot `
            -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput (Join-Path $RunRoot "$($stage.Name).stdout.log") `
            -RedirectStandardError (Join-Path $RunRoot "$($stage.Name).stderr.log")
        try { [GbChainCpuJob]::Assign($job, $proc.Handle) } catch {
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
            throw "Could not place $($stage.Name) under the CPU cap: $_"
        }
        $state.pid = $proc.Id
        Save-State
        Write-Host "[GB chain] $($stage.Name) pid=$($proc.Id) cap=$CpuRateBasisPoints started $(Get-Date -Format o)"
        $proc.WaitForExit()
        $proc.Refresh()
        $state.exitCode = $proc.ExitCode
        Save-State
        if ($proc.ExitCode -ne 0) { throw "$($stage.Name) failed with exit code $($proc.ExitCode)" }
    }
    $state.stage = 'complete'
} catch {
    $state.stage = 'failed'
    $state.error = [string]$_
    throw
} finally {
    $state.completedAt = (Get-Date).ToString('o')
    Save-State
    Write-Host "[GB chain] stage=$($state.stage) $(Get-Date -Format o)"
}
