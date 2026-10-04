$ErrorActionPreference = 'Stop'
$ProjectRoot = 'C:\Users\admin\Desktop\EVMALOCALUPDATE'
$RunRoot = Join-Path $ProjectRoot 'execute_results\bid_bank_runs\ev1_probe_20260924'
$StatePath = Join-Path $RunRoot 'status.json'
$Python = 'C:\Users\admin\AppData\Local\Programs\Python\Python310\python.exe'
$Probe = Join-Path $ProjectRoot 'execute_results\bid_bank_runs\ev1_median_probe.py'
$Scheme = '8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c'
$CpuLimit = 75
New-Item -ItemType Directory -Path $RunRoot -Force | Out-Null

Add-Type -TypeDefinition @"
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
public static class Ev1ProbeJob {
    [StructLayout(LayoutKind.Sequential)]
    public struct CpuRateControlInformation {
        public uint ControlFlags;
        public uint CpuRate;
    }
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern IntPtr CreateJobObjectW(IntPtr lpJobAttributes, string lpName);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool SetInformationJobObject(
        IntPtr hJob, int infoClass,
        ref CpuRateControlInformation info, uint size);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool AssignProcessToJobObject(IntPtr hJob, IntPtr process);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool CloseHandle(IntPtr hObject);
    public static IntPtr Create(uint percent) {
        IntPtr job = CreateJobObjectW(IntPtr.Zero, null);
        if (job == IntPtr.Zero) throw new Win32Exception(Marshal.GetLastWin32Error());
        var info = new CpuRateControlInformation();
        info.ControlFlags = 0x1 | 0x4;
        info.CpuRate = percent * 100;
        if (!SetInformationJobObject(job, 15, ref info,
                (uint)Marshal.SizeOf(typeof(CpuRateControlInformation)))) {
            int e = Marshal.GetLastWin32Error();
            CloseHandle(job);
            throw new Win32Exception(e);
        }
        return job;
    }
    public static void Assign(IntPtr job, IntPtr process) {
        if (!AssignProcessToJobObject(job, process))
            throw new Win32Exception(Marshal.GetLastWin32Error());
    }
    public static void Close(IntPtr job) {
        if (job != IntPtr.Zero) CloseHandle(job);
    }
}
"@

function Get-Boost {
    $text = (& powercfg.exe /query $Scheme SUB_PROCESSOR PERFBOOSTMODE | Out-String)
    $values = [regex]::Matches($text, '0x([0-9a-fA-F]{8})')
    if ($values.Count -lt 2) { throw 'Could not read original AC/DC boost settings' }
    return [pscustomobject]@{
        AC = [Convert]::ToInt32($values[$values.Count - 2].Groups[1].Value, 16)
        DC = [Convert]::ToInt32($values[$values.Count - 1].Groups[1].Value, 16)
    }
}
function Set-PowerValue {
    param([string[]]$PowerArgs)
    & powercfg.exe @PowerArgs
    if ($LASTEXITCODE -ne 0) { throw "powercfg failed: $($PowerArgs -join ' ')" }
}

$Original = Get-Boost
$State = [ordered]@{
    status = 'starting'
    startedAt = (Get-Date).ToString('o')
    originalBoostAC = $Original.AC
    originalBoostDC = $Original.DC
    boostDuringRun = 0
    cpuHardCapPercent = $CpuLimit
    scenarioWorkers = 24
    evScenarioSelection = 'median from same 128-candidate pool'
    serviceDate = '2024-04-02'
    commandCount = 128
    outputDir = 'execute_results/bid_banks/probe_aemo_bess_1ev_median_20260924'
    processId = $null
    completedAt = $null
    exitCode = $null
    error = $null
}
function Save-State {
    $State.updatedAt = (Get-Date).ToString('o')
    $State | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $StatePath -Encoding UTF8
}
Save-State
$job = [IntPtr]::Zero
$proc = $null
try {
    $State.status = 'disabling_cpu_boost'
    Save-State
    Set-PowerValue -PowerArgs @('/setacvalueindex', $Scheme, 'SUB_PROCESSOR', 'PERFBOOSTMODE', '0')
    Set-PowerValue -PowerArgs @('/setdcvalueindex', $Scheme, 'SUB_PROCESSOR', 'PERFBOOSTMODE', '0')
    Set-PowerValue -PowerArgs @('/setactive', $Scheme)
    $check = Get-Boost
    if ($check.AC -ne 0 -or $check.DC -ne 0) { throw 'CPU boost disable verification failed' }

    $env:EVMA_ACTIVATION_SIGNAL_SET = 'aemo_bess_dispatch'
    $env:EVMA_BID_SOLVE_CACHE = '0'
    $env:OMP_NUM_THREADS = '1'
    $env:MKL_NUM_THREADS = '1'
    $env:OPENBLAS_NUM_THREADS = '1'
    Remove-Item Env:EVMA_ACTIVATION_SCENARIO_DIR -ErrorAction SilentlyContinue

    $job = [Ev1ProbeJob]::Create([uint32]$CpuLimit)
    $stdout = Join-Path $RunRoot 'probe.stdout.log'
    $stderr = Join-Path $RunRoot 'probe.stderr.log'
    $proc = Start-Process -FilePath $Python -ArgumentList @($Probe) -WorkingDirectory $ProjectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    [Ev1ProbeJob]::Assign($job, $proc.Handle)
    $State.status = 'running'
    $State.processId = $proc.Id
    $State.stdout = $stdout
    $State.stderr = $stderr
    Save-State
    $proc.WaitForExit()
    $proc.Refresh()
    $State.exitCode = $proc.ExitCode
    $State.status = if ($proc.ExitCode -eq 0) { 'complete' } else { 'failed' }
} catch {
    $State.error = [string]$_
    $State.status = 'failed'
} finally {
    [Ev1ProbeJob]::Close($job)
    try {
        Set-PowerValue -PowerArgs @('/setacvalueindex', $Scheme, 'SUB_PROCESSOR', 'PERFBOOSTMODE', [string]$Original.AC)
        Set-PowerValue -PowerArgs @('/setdcvalueindex', $Scheme, 'SUB_PROCESSOR', 'PERFBOOSTMODE', [string]$Original.DC)
        Set-PowerValue -PowerArgs @('/setactive', $Scheme)
        $State.restoredBoostAC = (Get-Boost).AC
        $State.restoredBoostDC = (Get-Boost).DC
    } catch {
        $State.error = "Restore failed: $_"
        $State.status = 'restore_failed'
    }
    $State.completedAt = (Get-Date).ToString('o')
    Save-State
}

