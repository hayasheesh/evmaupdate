param([int]$CpuRateBasisPoints = 2500, [int]$GbEpisode = 1240)
# GB の360組で、完全情報LPの判定、GBで学習した AB（行動器だけ／force あり）、ルールベース、中央LP + force の5つを並べて走らせる。
# AEMO・ERCOT の表（force_comparison_20261002）と同じ方法で、市場だけを GB にする。
# 5つとも1つの Job Object に入れ、論理32 CPUの25%に抑える。電源設定は触らない。
$ErrorActionPreference = 'Stop'
$ProjectRoot = 'C:\Users\admin\Desktop\EVMALOCALUPDATE'
$RunRoot = Join-Path $ProjectRoot 'execute_results\unseen_decomposition_gb_20261003'
$Python = 'C:\Users\admin\AppData\Local\Programs\Python\Python310\python.exe'
$StatusPath = Join-Path $RunRoot 'status.json'
if (Test-Path -LiteralPath $StatusPath) { throw 'Run status already exists' }
Add-Type -TypeDefinition @"
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
public static class DecompGbCpuJob {
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
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:GB_MODEL_EPISODE = "$GbEpisode"
$tasks = [ordered]@{
    lp_certify = @('-X', 'utf8', '-u', (Join-Path $RunRoot 'lp_certify.py'))
    gb_marl_raw = @('-X', 'utf8', '-u', (Join-Path $RunRoot 'run_model.py'), '--model', 'gb')
    gb_marl_force = @('-X', 'utf8', '-u', (Join-Path $RunRoot 'run_model.py'), '--model', 'gb', '--pipeline', 'marl_force')
    rule = @('-X', 'utf8', '-u', (Join-Path $RunRoot 'run_model.py'), '--model', 'rule')
    central_lp_force = @('-X', 'utf8', '-u', (Join-Path $RunRoot 'run_model.py'), '--model', 'central_lp', '--pipeline', 'marl_force')
}
$state = [ordered]@{ cpuRateBasisPoints = $CpuRateBasisPoints; gbEpisode = $GbEpisode; startedAt = (Get-Date).ToString('o'); completedAt = $null; tasks = [ordered]@{} }
function Save-State { $script:state | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $StatusPath -Encoding UTF8 }
$job = [DecompGbCpuJob]::Create([uint32]$CpuRateBasisPoints)
$procs = @{}
foreach ($name in $tasks.Keys) {
    $p = Start-Process -FilePath $Python -ArgumentList $tasks[$name] -WorkingDirectory $ProjectRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $RunRoot "$name.stdout.log") -RedirectStandardError (Join-Path $RunRoot "$name.stderr.log")
    try { [DecompGbCpuJob]::Assign($job, $p.Handle) } catch { Stop-Process -Id $p.Id -Force; throw "CPU cap failed for ${name}: $_" }
    $procs[$name] = $p
    $state.tasks[$name] = [ordered]@{ pid = $p.Id; exitCode = $null }
    Write-Host "[decomp] $name pid=$($p.Id) $(Get-Date -Format o)"
}
Save-State
foreach ($name in $tasks.Keys) {
    $procs[$name].WaitForExit()
    $procs[$name].Refresh()
    $state.tasks[$name].exitCode = $procs[$name].ExitCode
    Save-State
    Write-Host "[decomp] $name exit=$($procs[$name].ExitCode) $(Get-Date -Format o)"
}
$state.completedAt = (Get-Date).ToString('o')
Save-State
