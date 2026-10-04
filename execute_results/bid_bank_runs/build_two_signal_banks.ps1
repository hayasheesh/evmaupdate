$ErrorActionPreference = 'Stop'
$ProjectRoot = 'C:\Users\admin\Desktop\EVMALOCALUPDATE'
$RunRoot = Join-Path $ProjectRoot 'execute_results\bid_bank_runs\two_signal_20260924'
$StatePath = Join-Path $RunRoot 'status.json'
$Python = 'C:\Users\admin\AppData\Local\Programs\Python\Python310\python.exe'
$BoostSubgroup = 'SUB_PROCESSOR'
$BoostSetting = 'PERFBOOSTMODE'
$CpuLimit = 75
New-Item -ItemType Directory -Path $RunRoot -Force | Out-Null

Add-Type -TypeDefinition @"
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
public static class CpuJobLimit {
    [StructLayout(LayoutKind.Sequential)]
    public struct CpuRateControlInformation {
        public uint ControlFlags;
        public uint CpuRate;
    }
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern IntPtr CreateJobObjectW(IntPtr lpJobAttributes, string lpName);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool SetInformationJobObject(
        IntPtr hJob, int JobObjectInfoClass,
        ref CpuRateControlInformation lpJobObjectInfo, uint cbJobObjectInfo);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool AssignProcessToJobObject(IntPtr hJob, IntPtr hProcess);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool TerminateJobObject(IntPtr hJob, uint uExitCode);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool CloseHandle(IntPtr hObject);
    public static IntPtr CreateHardCap(uint percent) {
        IntPtr job = CreateJobObjectW(IntPtr.Zero, null);
        if (job == IntPtr.Zero) throw new Win32Exception(Marshal.GetLastWin32Error());
        var info = new CpuRateControlInformation();
        info.ControlFlags = 0x1 | 0x4;
        info.CpuRate = percent * 100;
        if (!SetInformationJobObject(job, 15, ref info,
                (uint)Marshal.SizeOf(typeof(CpuRateControlInformation)))) {
            int error = Marshal.GetLastWin32Error();
            CloseHandle(job);
            throw new Win32Exception(error);
        }
        return job;
    }
    public static void Assign(IntPtr job, IntPtr process) {
        if (!AssignProcessToJobObject(job, process))
            throw new Win32Exception(Marshal.GetLastWin32Error());
    }
    public static void Terminate(IntPtr job) {
        if (job != IntPtr.Zero) TerminateJobObject(job, 1);
    }
    public static void Close(IntPtr job) {
        if (job != IntPtr.Zero) CloseHandle(job);
    }
}
"@

$activeText = (& powercfg.exe /getactivescheme | Out-String)
$guidMatch = [regex]::Match($activeText, '[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}')
if (-not $guidMatch.Success) { throw 'Could not read active Windows power scheme GUID' }
$SchemeGuid = $guidMatch.Value

function Read-BoostIndices {
    $text = (& powercfg.exe /query $SchemeGuid $BoostSubgroup $BoostSetting | Out-String)
    $matches = [regex]::Matches($text, '0x([0-9a-fA-F]{8})')
    if ($matches.Count -lt 2) { throw 'Could not read AC/DC CPU boost indices' }
    return [pscustomobject]@{
        AC = [Convert]::ToInt32($matches[$matches.Count - 2].Groups[1].Value, 16)
        DC = [Convert]::ToInt32($matches[$matches.Count - 1].Groups[1].Value, 16)
    }
}
function Invoke-PowerCfg {
    param([string[]]$Arguments)
    & powercfg.exe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "powercfg failed ($LASTEXITCODE): $($Arguments -join ' ')"
    }
}

$OriginalBoost = Read-BoostIndices
$State = [ordered]@{
    startedAt = (Get-Date).ToString('o')
    updatedAt = (Get-Date).ToString('o')
    status = 'starting'
    activePowerScheme = $SchemeGuid
    originalBoostAC = $OriginalBoost.AC
    originalBoostDC = $OriginalBoost.DC
    boostDuringBuild = 0
    cpuHardCapPercent = $CpuLimit
    dayWorkers = 1
    scenarioWorkers = 8
    signals = @('aemo_bess_dispatch', 'ercot')
    steps = @()
    selectedErcotBoundaries = $null
    completedAt = $null
    error = $null
}
function Save-State {
    $script:State.updatedAt = (Get-Date).ToString('o')
    $script:State | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $StatePath -Encoding UTF8
}
Save-State

function Start-LimitedProcess {
    param([string]$Name, [string]$FilePath, [string[]]$Arguments)
    $stdout = Join-Path $RunRoot "$Name.stdout.log"
    $stderr = Join-Path $RunRoot "$Name.stderr.log"
    $job = [CpuJobLimit]::CreateHardCap([uint32]$CpuLimit)
    $proc = $null
    try {
        $proc = Start-Process -FilePath $FilePath -ArgumentList $Arguments -WorkingDirectory $ProjectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
        try {
            [CpuJobLimit]::Assign($job, $proc.Handle)
        } catch {
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
            throw "Could not place process $($proc.Id) under the $CpuLimit% CPU cap: $_"
        }
        $step = [ordered]@{
            name = $Name
            pid = $proc.Id
            command = ($Arguments -join ' ')
            status = 'running'
            startedAt = (Get-Date).ToString('o')
            completedAt = $null
            exitCode = $null
            stdout = $stdout
            stderr = $stderr
            cpuHardCapPercent = $CpuLimit
            boostMode = 'disabled'
        }
        $script:State.steps += $step
        Save-State
        $proc.WaitForExit()
        $proc.Refresh()
        $step.completedAt = (Get-Date).ToString('o')
        $step.exitCode = $proc.ExitCode
        $step.status = if ($proc.ExitCode -eq 0) { 'complete' } else { 'failed' }
        Save-State
        return [int]$proc.ExitCode
    } finally {
        [CpuJobLimit]::Close($job)
    }
}

function Run-Step {
    param([string]$Name, [string]$FilePath, [string[]]$Arguments)
    $script:State.status = "running:$Name"
    Save-State
    try {
        $exitCode = Start-LimitedProcess -Name $Name -FilePath $FilePath -Arguments $Arguments
        if ($exitCode -ne 0) {
            $script:State.status = "step_failed:$Name"
            Save-State
            return $false
        }
        return $true
    } catch {
        $script:State.steps += [ordered]@{
            name = $Name
            status = 'failed_to_start'
            completedAt = (Get-Date).ToString('o')
            error = [string]$_
        }
        $script:State.status = "step_failed:$Name"
        Save-State
        return $false
    }
}

$Failures = @()
try {
    $State.status = 'disabling_cpu_boost'
    Save-State
    Invoke-PowerCfg -Arguments @('/setacvalueindex', $SchemeGuid, $BoostSubgroup, $BoostSetting, '0')
    Invoke-PowerCfg -Arguments @('/setdcvalueindex', $SchemeGuid, $BoostSubgroup, $BoostSetting, '0')
    Invoke-PowerCfg -Arguments @('/setactive', $SchemeGuid)
    $disabled = Read-BoostIndices
    if ($disabled.AC -ne 0 -or $disabled.DC -ne 0) {
        throw "Boost verification failed: AC=$($disabled.AC), DC=$($disabled.DC)"
    }
    $State.status = 'boost_disabled'
    Save-State

    $ok = Run-Step -Name 'aemo_bess_dispatch_bank' -FilePath $Python -Arguments @(
        'tools/build_all_upper_bid_banks.py',
        '--train-split-count', '25',
        '--train-days', '25',
        '--test-days', '5',
        '--workers', '1',
        '--activation-signal-set', 'aemo_bess_dispatch'
    )
    if (-not $ok) { $Failures += 'aemo_bess_dispatch_bank' }

    $auditOk = Run-Step -Name 'ercot_rtc_audit' -FilePath $Python -Arguments @(
        'tools/build_ercot_sced_scenarios.py',
        '--input-dir', 'ercot_local_downloader/ercot_sced_rtc_output',
        '--audit-only',
        '--output-dir', 'execute_results/bid_bank_runs/two_signal_20260924/ercot_rtc_audit'
    )
    if (-not $auditOk) {
        $Failures += 'ercot_rtc_audit'
    } else {
        $auditPath = Join-Path $RunRoot 'ercot_rtc_audit\metadata.json'
        $audit = Get-Content -LiteralPath $auditPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $eligibleDates = @(
            $audit.days |
            Where-Object { $null -eq $_.reasons -or $_.reasons.Count -eq 0 } |
            ForEach-Object { [string]$_.date } |
            Sort-Object -Unique
        )
        if ($eligibleDates.Count -lt 10) {
            $Failures += 'ercot_rtc_insufficient_audited_dates'
            $State.status = 'ercot_audit_needs_review'
            $State.error = "Only $($eligibleDates.Count) dates with at least one valid resource-day"
            Save-State
        } else {
            $trainCount = [Math]::Max(1, [int][Math]::Floor($eligibleDates.Count * 0.60))
            $validationCount = [Math]::Max($trainCount + 1, [int][Math]::Floor($eligibleDates.Count * 0.80))
            $validationCount = [Math]::Min($validationCount, $eligibleDates.Count - 1)
            $trainEnd = $eligibleDates[$trainCount - 1]
            $validationEnd = $eligibleDates[$validationCount - 1]
            $State.selectedErcotBoundaries = [ordered]@{
                policy = 'chronological 60/20/20 by audited eligible calendar dates; common boundaries for all resources'
                eligibleDateCount = $eligibleDates.Count
                trainEnd = $trainEnd
                validationEnd = $validationEnd
                eligibleFirstDate = $eligibleDates[0]
                eligibleLastDate = $eligibleDates[$eligibleDates.Count - 1]
            }
            Save-State
            $rtcB = Join-Path $ProjectRoot 'data\ercot\sced\processed_5min\rtc_b'
            if (Test-Path -LiteralPath $rtcB) {
                $existing = @(Get-ChildItem -LiteralPath $rtcB -Force -ErrorAction SilentlyContinue)
                if ($existing.Count -gt 0) {
                    throw "Refusing to overwrite existing RTC+B data at $rtcB"
                }
            }
            $buildErcot = Run-Step -Name 'ercot_rtc_build' -FilePath $Python -Arguments @(
                'tools/build_ercot_sced_scenarios.py',
                '--input-dir', 'ercot_local_downloader/ercot_sced_rtc_output',
                '--train-end', $trainEnd,
                '--validation-end', $validationEnd,
                '--output-dir', 'data/ercot/sced/processed_5min/rtc_b'
            )
            if (-not $buildErcot) {
                $Failures += 'ercot_rtc_build'
            } else {
                $bankErcot = Run-Step -Name 'ercot_bid_bank' -FilePath $Python -Arguments @(
                    'tools/build_all_upper_bid_banks.py',
                    '--train-split-count', '25',
                    '--train-days', '25',
                    '--test-days', '5',
                    '--workers', '1',
                    '--activation-signal-set', 'ercot'
                )
                if (-not $bankErcot) { $Failures += 'ercot_bid_bank' }
            }
        }
    }
} catch {
    $State.error = [string]$_
    $Failures += 'orchestrator'
} finally {
    try {
        $State.status = 'restoring_cpu_boost'
        Save-State
        Invoke-PowerCfg -Arguments @('/setacvalueindex', $SchemeGuid, $BoostSubgroup, $BoostSetting, [string]$OriginalBoost.AC)
        Invoke-PowerCfg -Arguments @('/setdcvalueindex', $SchemeGuid, $BoostSubgroup, $BoostSetting, [string]$OriginalBoost.DC)
        Invoke-PowerCfg -Arguments @('/setactive', $SchemeGuid)
        $restored = Read-BoostIndices
        $State.restoredBoostAC = $restored.AC
        $State.restoredBoostDC = $restored.DC
    } catch {
        $State.error = "Power boost restore error: $_"
        $Failures += 'restore_cpu_boost'
    }
    $State.completedAt = (Get-Date).ToString('o')
    if ($Failures.Count -eq 0) {
        $State.status = 'complete'
    } else {
        $State.status = 'completed_with_failures'
        $State.failures = @($Failures)
    }
    Save-State
}

