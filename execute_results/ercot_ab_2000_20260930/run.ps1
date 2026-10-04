param(
    [int] $Episodes = 2000,
    [int] $CpuRateBasisPoints = 7500
)
$ErrorActionPreference = 'Stop'
$ProjectRoot = 'C:\Users\admin\Desktop\EVMALOCALUPDATE'
$RunRoot = Join-Path $ProjectRoot 'execute_results\ercot_ab_2000_20260930'
$RunDir = $null
$SourceDir = Join-Path $RunRoot 'runtime_source'
$Python = 'C:\Users\admin\AppData\Local\Programs\Python\Python310\python.exe'
$StatusPath = Join-Path $RunRoot 'status.json'
if ($Episodes -ne 2000 -or $CpuRateBasisPoints -ne 7500) { throw 'Only the configured 2000-episode, 75% CPU run is permitted' }
if (Test-Path -LiteralPath $StatusPath) { throw 'Run status already exists; do not start a duplicate run' }
$RunDirMarker = Join-Path $RunRoot 'run_dir.json'

Add-Type -TypeDefinition @"
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
public static class ErcotABTrainCpuJob {
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

function Invoke-PowerSetting {
    param([string[]] $PowerArgs)
    & powercfg @PowerArgs | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "powercfg failed: $($PowerArgs -join ' ')" }
}
function Get-ActiveSchemeGuid {
    $output = & powercfg /getactivescheme
    if ($LASTEXITCODE -ne 0) { throw 'Could not read active power scheme' }
    $match = [regex]::Match(($output -join ' '), '[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}')
    if (-not $match.Success) { throw 'Could not parse active power scheme' }
    return $match.Value
}
function Get-AcBoostMode {
    param([string] $SchemeGuid)
    $output = & powercfg /query $SchemeGuid SUB_PROCESSOR PERFBOOSTMODE
    if ($LASTEXITCODE -ne 0) { throw 'Could not read AC boost mode' }
    $line = $output | Where-Object { $_ -match 'AC' -and $_ -match '0x[0-9a-fA-F]+' } | Select-Object -Last 1
    $match = [regex]::Match([string]$line, '0x[0-9a-fA-F]+')
    if (-not $match.Success) { throw 'Could not parse AC boost mode' }
    return [Convert]::ToInt32($match.Value.Substring(2), 16)
}
$env:PYTHONUTF8 = '1'
$scheme = Get-ActiveSchemeGuid
$originalBoost = Get-AcBoostMode -SchemeGuid $scheme
$state = [ordered]@{
    modelName = 'prod_ercotplan_AB_7station'; runDir = $RunDir; sourceDir = $SourceDir
    startEpisode = 0; targetEpisode = $Episodes; exactResume = $false
    cpuRateBasisPoints = $CpuRateBasisPoints; powerSchemeGuid = $scheme
    originalAcBoostMode = $originalBoost; acBoostModeDuringRun = 0
    powerSettingsRestored = $false; startedAt = (Get-Date).ToString('o')
    completedAt = $null; launcherPid = $PID; pid = $null
    stage = 'starting'; exitCode = $null; error = $null
}
function Save-State {
    $script:state | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath "$StatusPath.tmp" -Encoding UTF8
    Move-Item -LiteralPath "$StatusPath.tmp" -Destination $StatusPath -Force
}
Save-State
$boostChanged = $false
try {
    Invoke-PowerSetting -PowerArgs @('/setacvalueindex', $scheme, 'SUB_PROCESSOR', 'PERFBOOSTMODE', '0')
    Invoke-PowerSetting -PowerArgs @('/setactive', $scheme)
    $boostChanged = $true
    if ((Get-AcBoostMode -SchemeGuid $scheme) -ne 0) { throw 'AC processor boost did not turn off' }
    $job = [ErcotABTrainCpuJob]::Create([uint32]$CpuRateBasisPoints)
    $arguments = @('-u', (Join-Path $RunRoot 'worker.py'), '--episodes', "$Episodes")
    $proc = Start-Process -FilePath $Python -ArgumentList $arguments -WorkingDirectory $ProjectRoot `
        -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $RunRoot 'pretrain.stdout.log') `
        -RedirectStandardError (Join-Path $RunRoot 'pretrain.stderr.log')
    try { [ErcotABTrainCpuJob]::Assign($job, $proc.Handle) } catch {
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        throw "Could not place learner under CPU cap: $_"
    }
    $state.pid = $proc.Id
    $state.stage = 'training'
    Save-State
    Write-Host "[ERCOT AB] pid=$($proc.Id) target=$Episodes cap=$CpuRateBasisPoints boost=0 $(Get-Date -Format o)"
    while (-not $proc.WaitForExit(1000)) {
        if (-not $RunDir -and (Test-Path -LiteralPath $RunDirMarker)) {
            $RunDir = (Get-Content -LiteralPath $RunDirMarker -Raw | ConvertFrom-Json).runDir
            $state.runDir = $RunDir
            Save-State
        }
    }
    $proc.Refresh()
    $state.exitCode = $proc.ExitCode
    if ($proc.ExitCode -ne 0) { throw "learner failed with exit code $($proc.ExitCode)" }
    if (-not $RunDir -and (Test-Path -LiteralPath $RunDirMarker)) {
        $RunDir = (Get-Content -LiteralPath $RunDirMarker -Raw | ConvertFrom-Json).runDir
        $state.runDir = $RunDir
    }
    if (-not $RunDir) { throw 'Learner exited without a saved run directory' }
    $latest = Get-Content -LiteralPath (Join-Path $RunDir 'resume\latest.json') -Raw | ConvertFrom-Json
    $state.stage = if ($latest.completed_training_episode -ge $Episodes) { 'complete' } else { 'stopped' }
    $state.completedAt = (Get-Date).ToString('o')
    Save-State
} catch {
    $state.stage = 'failed'
    $state.completedAt = (Get-Date).ToString('o')
    $state.error = [string]$_
    Save-State
    throw
} finally {
    if ($boostChanged) {
        try {
            if ((Get-AcBoostMode -SchemeGuid $scheme) -eq 0) {
                Invoke-PowerSetting -PowerArgs @('/setacvalueindex', $scheme, 'SUB_PROCESSOR', 'PERFBOOSTMODE', "$originalBoost")
                if ((Get-ActiveSchemeGuid) -eq $scheme) { Invoke-PowerSetting -PowerArgs @('/setactive', $scheme) }
            }
            $state.powerSettingsRestored = ((Get-AcBoostMode -SchemeGuid $scheme) -eq $originalBoost)
        } catch { $state.error = "Power setting restoration failed: $_" }
    }
    Save-State
    Write-Host "[ERCOT AB] stage=$($state.stage) boost_restored=$($state.powerSettingsRestored) $(Get-Date -Format o)"
}
