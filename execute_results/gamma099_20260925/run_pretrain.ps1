param([int] $CpuRateBasisPoints = 1250)
# noalloc main line (prod_f500_dense8_noalloc_7station, 2026-09-21) with only
# the global-critic discount changed: GAMMA_GLOBAL 0.95 -> 0.99. The mixer bias
# cap rises 50 -> 150 with it, since the tracking reward's Bellman fixed point
# goes from 20 to 100. Everything else is the noalloc run's recorded runtime.
# The old bank is used as built (its settings predate the current bank
# contract) and is never rebuilt. Job Object hard CPU cap 1250 = 4 of 32
# logical CPUs while the new bid bank is still being built. Power settings are
# not touched.
$ErrorActionPreference = 'Stop'
$ProjectRoot = 'C:\Users\admin\Desktop\EVMALOCALUPDATE'
$RunRoot = Join-Path $ProjectRoot 'execute_results\gamma099_20260925'
$Python = 'C:\Users\admin\AppData\Local\Programs\Python\Python310\python.exe'

Add-Type -TypeDefinition @"
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
public static class PretrainCpuJob {
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
$env:EVMA_USE_CENTRAL_EV_RESIDUAL_ALLOCATOR = '0'
$env:EVMA_ACTOR_EV_COUNT = '1'
$env:EVMA_GLOBAL_BALANCE_REWARD_MODE = 'bounded_absolute_error'
$env:EVMA_GLOBAL_BALANCE_REWARD_ERROR_SCALE_KW = '150'
$env:EVMA_GRAD_CLIP_MAX = '5.0'
$env:EVMA_GRAD_CLIP_MAX_GLOBAL = '5.0'
$env:EVMA_ACTIVATION_SCENARIO_DIR = 'data/aemo/nem/processed_5min/dense8'
$env:EVMA_LOWER_TRAIN_RESEARCH_MINIMUM_BID_QUANTITY_KW = '500'
$env:EVMA_LOWER_BID_LOOKAHEAD_BLOCKS = '24'
$env:EVMA_GAMMA_GLOBAL = '0.99'
$env:EVMA_MIXER_B_MAX = '150'
$env:EVMA_LOWER_TRAIN_ACCEPT_BANK_AS_IS = '1'
$env:EVMA_LOWER_TRAIN_BUILD_BID_BANK = '0'
Remove-Item Env:EVMA_LOCAL_FLEET_RESIDUAL_OBS -ErrorAction SilentlyContinue
Remove-Item Env:EVMA_GLOBAL_BALANCE_REWARD_LINEAR_TAIL_KW -ErrorAction SilentlyContinue
Remove-Item Env:EVMA_Q_MIX_GLOBAL_WEIGHT -ErrorAction SilentlyContinue

$arguments = @('-u', 'pre_train.py', '--episodes', '2000',
    '--model-name', 'prod_f500_dense8_noalloc_g099_7station',
    '--bank-dir', 'execute_results/bid_banks/train_25_7s_f500_dense8',
    '--test-bank-dir', 'execute_results/bid_banks/validation_5_7s_f500_dense8')
$stdout = Join-Path $RunRoot 'pretrain.stdout.log'
$stderr = Join-Path $RunRoot 'pretrain.stderr.log'
$job = [PretrainCpuJob]::Create([uint32]$CpuRateBasisPoints)
$proc = Start-Process -FilePath $Python -ArgumentList $arguments -WorkingDirectory $ProjectRoot `
    -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
try { [PretrainCpuJob]::Assign($job, $proc.Handle) } catch {
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    throw "Could not place the pretrain under the CPU cap: $_"
}
Write-Host "[pretrain] pid=$($proc.Id) started $(Get-Date -Format o)"
$proc.WaitForExit()
Write-Host "[pretrain] exit=$($proc.ExitCode) $(Get-Date -Format o)"
