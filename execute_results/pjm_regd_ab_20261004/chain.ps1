param(
    [int] $BankCpuRateBasisPoints = 5000,
    [int] $TrainCpuRateBasisPoints = 7500
)
# PJM RegD（疑似指令 pjm_regd_phase_shift）で、入札（学習用25日・検証用5日、設計指令128本 × EV 3本）を作り、
# 終わったら AB 学習（2000回）へ進む。入札の引数は GB のとき（bid_bank_runs/aemo_plan_bank_20260925/run_split.ps1）と同じ。
# 入札は2つの区分を同時に走らせ、1つの Job Object で論理32 CPUの50%に抑える。学習は別の Job Object で75%に抑える。
# 電源設定（boost）には触らない。
$ErrorActionPreference = 'Stop'
$ProjectRoot = 'C:\Users\admin\Desktop\EVMALOCALUPDATE'
$RunRoot = Join-Path $ProjectRoot 'execute_results\pjm_regd_ab_20261004'
$Python = 'C:\Users\admin\AppData\Local\Programs\Python\Python310\python.exe'
$StatusPath = Join-Path $RunRoot 'chain_status.json'
$TrainBank = Join-Path $ProjectRoot 'execute_results\bid_banks\train_25_minmedmax_3of128ev_128cmd_all_commands_pjm_regd_phase_shift'
$TestBank = Join-Path $ProjectRoot 'execute_results\bid_banks\validation_5_minmedmax_3of128ev_128cmd_all_commands_pjm_regd_phase_shift'
if (Test-Path -LiteralPath $StatusPath) { throw 'chain_status.json already exists; do not start a duplicate' }

Add-Type -TypeDefinition @"
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
public static class PjmChainCpuJob {
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

$state = [ordered]@{
    signalSet = 'pjm_regd_phase_shift'; model = 'prod_pjmregd_AB_7station'; targetEpisode = 2000
    bankCpuRateBasisPoints = $BankCpuRateBasisPoints; trainCpuRateBasisPoints = $TrainCpuRateBasisPoints
    powerSettingsTouched = $false; acBoostModeAtStart = (Read-AcBoostMode); acBoostModeAtEnd = $null
    launcherPid = $PID; startedAt = (Get-Date).ToString('o'); completedAt = $null
    stage = 'bank'; stageStartedAt = [ordered]@{ bank = (Get-Date).ToString('o') }
    bankPids = [ordered]@{}; bankExitCodes = [ordered]@{}; trainPid = $null; trainExitCode = $null; error = $null
}
function Save-State {
    $script:state | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath "$StatusPath.tmp" -Encoding UTF8
    Move-Item -LiteralPath "$StatusPath.tmp" -Destination $StatusPath -Force
}
function Test-BankComplete([string] $Bank, [int] $Days) {
    $manifest = Join-Path $Bank 'manifest.json'
    if (-not (Test-Path -LiteralPath $manifest)) { return $false }
    $m = Get-Content -LiteralPath $manifest -Raw | ConvertFrom-Json
    return ($m.complete -eq $true -and [int]$m.completed_days -eq $Days)
}
Save-State
try {
    $env:PYTHONUTF8 = '1'
    $env:OMP_NUM_THREADS = '1'
    $env:MKL_NUM_THREADS = '1'
    $env:OPENBLAS_NUM_THREADS = '1'
    $env:EVMA_ACTIVATION_SIGNAL_SET = 'pjm_regd_phase_shift'
    Remove-Item Env:EVMA_ACTIVATION_SCENARIO_DIR -ErrorAction SilentlyContinue
    $bankJob = [PjmChainCpuJob]::Create([uint32]$BankCpuRateBasisPoints)
    $splits = [ordered]@{ train = @(25, 7); test = @(5, 3) }
    $procs = @{}
    foreach ($split in $splits.Keys) {
        $days = $splits[$split][0]; $workers = $splits[$split][1]
        $arguments = @('tools/build_training_bid_bank.py', '--split', $split, '--days', "$days",
            '--train-split-count', '25', '--paired-train-days', '25', '--paired-test-days', '5',
            '--workers', "$workers", '--scenario-workers', '4')
        $p = Start-Process -FilePath $Python -ArgumentList $arguments -WorkingDirectory $ProjectRoot -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput (Join-Path $RunRoot "bank_$split.stdout.log") -RedirectStandardError (Join-Path $RunRoot "bank_$split.stderr.log")
        try { [PjmChainCpuJob]::Assign($bankJob, $p.Handle) } catch { Stop-Process -Id $p.Id -Force; throw "CPU cap failed for bank ${split}: $_" }
        $procs[$split] = $p
        $state.bankPids[$split] = $p.Id
        Save-State
        Write-Host "[PJM chain] bank $split pid=$($p.Id) $(Get-Date -Format o)"
    }
    foreach ($split in $splits.Keys) {
        $procs[$split].WaitForExit()
        $procs[$split].Refresh()
        $state.bankExitCodes[$split] = $procs[$split].ExitCode
        Save-State
        Write-Host "[PJM chain] bank $split exit=$($procs[$split].ExitCode) $(Get-Date -Format o)"
    }
    foreach ($split in $splits.Keys) { if ($state.bankExitCodes[$split] -ne 0) { throw "bank $split exited with code $($state.bankExitCodes[$split])" } }
    if (-not (Test-BankComplete $TrainBank 25)) { throw 'train bank is incomplete' }
    if (-not (Test-BankComplete $TestBank 5)) { throw 'test bank is incomplete' }
    Remove-Item Env:EVMA_ACTIVATION_SIGNAL_SET -ErrorAction SilentlyContinue

    $state.stage = 'preflight'
    $state.stageStartedAt['preflight'] = (Get-Date).ToString('o')
    Save-State
    $pre = Start-Process -FilePath $Python -ArgumentList @('-X', 'utf8', '-u', (Join-Path $RunRoot 'worker.py'), '--preflight-only') `
        -WorkingDirectory $ProjectRoot -WindowStyle Hidden -PassThru -Wait `
        -RedirectStandardOutput (Join-Path $RunRoot 'preflight.stdout.log') -RedirectStandardError (Join-Path $RunRoot 'preflight.stderr.log')
    if ($pre.ExitCode -ne 0) { throw "preflight exited with code $($pre.ExitCode)" }

    $state.stage = 'training'
    $state.stageStartedAt['training'] = (Get-Date).ToString('o')
    $trainJob = [PjmChainCpuJob]::Create([uint32]$TrainCpuRateBasisPoints)
    $t = Start-Process -FilePath $Python -ArgumentList @('-X', 'utf8', '-u', (Join-Path $RunRoot 'worker.py'), '--episodes', '2000') `
        -WorkingDirectory $ProjectRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $RunRoot 'pretrain.stdout.log') -RedirectStandardError (Join-Path $RunRoot 'pretrain.stderr.log')
    try { [PjmChainCpuJob]::Assign($trainJob, $t.Handle) } catch { Stop-Process -Id $t.Id -Force; throw "CPU cap failed for training: $_" }
    $state.trainPid = $t.Id
    Save-State
    Write-Host "[PJM chain] training pid=$($t.Id) $(Get-Date -Format o)"
    $t.WaitForExit()
    $t.Refresh()
    $state.trainExitCode = $t.ExitCode
    if ($t.ExitCode -ne 0) { throw "training exited with code $($t.ExitCode)" }
    $state.stage = 'complete'
} catch {
    $state.stage = 'failed'
    $state.error = [string]$_
    throw
} finally {
    $state.completedAt = (Get-Date).ToString('o')
    $state.acBoostModeAtEnd = (Read-AcBoostMode)
    Save-State
    Write-Host "[PJM chain] stage=$($state.stage) $(Get-Date -Format o)"
}
