param(
    [int] $BidPid = 0,
    [string] $GpuJobPids = '',
    [int] $CpuRateBasisPoints = 7500
)
# GB の AB 学習を、次の2つが済んでから始める（chain.ps1 の後半と同じ処理）。
#   1. 検証用入札（-BidPid の build_training_bid_bank.py）が終わり、学習用・検証用の入札バンクが完成している
#   2. 利用者の別の GPU 作業（-GpuJobPids）が終わっている
# Job Object で論理32 CPUの75%に抑える。電源設定（boost）は触らない。
$ErrorActionPreference = 'Stop'
$GpuPids = @($GpuJobPids -split ',' | Where-Object { $_.Trim() } | ForEach-Object { [int]$_.Trim() })
$ProjectRoot = 'C:\Users\admin\Desktop\EVMALOCALUPDATE'
$RunRoot = Join-Path $ProjectRoot 'execute_results\gb_128cmd_marl_20261002'
$Python = 'C:\Users\admin\AppData\Local\Programs\Python\Python310\python.exe'
$StatusPath = Join-Path $RunRoot 'marl_status.json'
$TrainBank = Join-Path $ProjectRoot 'execute_results\bid_banks\train_25_minmedmax_3of128ev_128cmd_all_commands_elexon_plan_deviation'
$TestBank = Join-Path $ProjectRoot 'execute_results\bid_banks\validation_5_minmedmax_3of128ev_128cmd_all_commands_elexon_plan_deviation'
if (Test-Path -LiteralPath $StatusPath) { throw 'marl_status.json already exists; do not start a duplicate' }

Add-Type -TypeDefinition @"
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
public static class Gb128MarlCpuJob {
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
    model = 'prod_elexonplan_AB_7station'; targetEpisode = 2000
    bidPid = $BidPid; gpuJobPids = $GpuPids; cpuRateBasisPoints = $CpuRateBasisPoints; powerSettingsTouched = $false
    launcherPid = $PID; startedAt = (Get-Date).ToString('o'); completedAt = $null
    stage = 'waiting_for_bid'; pid = $null; exitCode = $null; error = $null; stageStartedAt = [ordered]@{}
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
    if ($BidPid -gt 0) {
        while (Get-Process -Id $BidPid -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 30 }
    }
    if (-not (Test-BankComplete $TrainBank 25)) { throw 'train bid bank is not complete' }
    if (-not (Test-BankComplete $TestBank 5)) { throw 'validation bid bank is not complete' }
    $state.stage = 'waiting_for_gpu'
    $state.stageStartedAt['waiting_for_gpu'] = (Get-Date).ToString('o')
    Save-State
    Write-Host "[GB MARL] bid banks complete; waiting for GPU job(s) $($GpuPids -join ',') $(Get-Date -Format o)"
    while ($GpuPids | Where-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue }) { Start-Sleep -Seconds 30 }

    $env:PYTHONUTF8 = '1'
    $env:OMP_NUM_THREADS = '1'
    $env:MKL_NUM_THREADS = '1'
    $env:OPENBLAS_NUM_THREADS = '1'
    Get-ChildItem Env: | Where-Object { $_.Name -like 'EVMA_*' } | ForEach-Object { Remove-Item "Env:$($_.Name)" }
    $job = [Gb128MarlCpuJob]::Create([uint32]$CpuRateBasisPoints)
    $stages = @(
        @{ Name = 'marl_preflight'; Args = @('-X', 'utf8', '-u', (Join-Path $RunRoot 'worker.py'), '--episodes', '2000', '--preflight-only') },
        @{ Name = 'pretrain'; Args = @('-X', 'utf8', '-u', (Join-Path $RunRoot 'worker.py'), '--episodes', '2000') }
    )
    foreach ($stage in $stages) {
        $state.stage = $stage.Name
        $state.stageStartedAt[$stage.Name] = (Get-Date).ToString('o')
        $proc = Start-Process -FilePath $Python -ArgumentList $stage.Args -WorkingDirectory $ProjectRoot -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput (Join-Path $RunRoot "$($stage.Name).stdout.log") `
            -RedirectStandardError (Join-Path $RunRoot "$($stage.Name).stderr.log")
        try { [Gb128MarlCpuJob]::Assign($job, $proc.Handle) } catch {
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
            throw "Could not place $($stage.Name) under the CPU cap: $_"
        }
        $state.pid = $proc.Id
        Save-State
        Write-Host "[GB MARL] $($stage.Name) pid=$($proc.Id) cap=$CpuRateBasisPoints started $(Get-Date -Format o)"
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
    Write-Host "[GB MARL] stage=$($state.stage) $(Get-Date -Format o)"
}
