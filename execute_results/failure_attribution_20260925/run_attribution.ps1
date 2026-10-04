param(
    [string] $Name = 'abg_ep200',
    [string[]] $ScriptArgs = @(),
    [int] $CpuRateBasisPoints = 625
)
# Runs attribute_failures.py under its own Job Object hard CPU cap
# (625 = 6.25% of 32 logical CPUs = 2) so that it and the running bid bank
# together stay near the 28-core budget. Power settings are not touched.
$ErrorActionPreference = 'Stop'
$ProjectRoot = 'C:\Users\admin\Desktop\EVMALOCALUPDATE'
$RunRoot = Join-Path $ProjectRoot 'execute_results\failure_attribution_20260925'
$Python = 'C:\Users\admin\AppData\Local\Programs\Python\Python310\python.exe'

Add-Type -TypeDefinition @"
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
public static class AttributionCpuJob {
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

$env:PYTHONUTF8 = '1'
$stdout = Join-Path $RunRoot "$Name.stdout.log"
$stderr = Join-Path $RunRoot "$Name.stderr.log"
$job = [AttributionCpuJob]::Create([uint32]$CpuRateBasisPoints)
$proc = Start-Process -FilePath $Python -ArgumentList (@('execute_results/failure_attribution_20260925/attribute_failures.py') + $ScriptArgs) `
    -WorkingDirectory $ProjectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
try { [AttributionCpuJob]::Assign($job, $proc.Handle) } catch {
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    throw "Could not place the attribution run under the CPU cap: $_"
}
Write-Host "[attribution] pid=$($proc.Id) started $(Get-Date -Format o)"
$proc.WaitForExit()
Write-Host "[attribution] exit=$($proc.ExitCode) $(Get-Date -Format o)"
