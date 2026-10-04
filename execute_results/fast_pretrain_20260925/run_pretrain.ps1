param(
    [Parameter(Mandatory = $true)] [string] $Name,
    [int] $Episodes = 600,
    [int] $EpsilonEnd = 1000,
    [int] $NoiseEnd = 1500,
    [string] $LrActor = '1e-6',
    [string] $LrCritic = '3e-6',
    [string] $Tau = '0.001',
    [int] $UpdatesPerStep = 1,
    [string] $LocalRewardScale = '1.0',
    [int] $CheckpointInterval = 50,
    [int] $CpuRateBasisPoints = 1250
)
# noalloc main line (prod_f500_dense8_noalloc_7station, 2026-09-21) with a
# shorter budget. Only the exploration schedule ends, the learning rates, the
# target-network rate, the updates per environment step and the local reward
# multiplier are passed in; the local and global critics share one learning
# rate and one target rate, as they do in the noalloc run. The actor
# observes no fleet residual. The old bank is used as built and never rebuilt.
# Job Object hard CPU cap in basis points of all 32 logical CPUs (1250 = 4).
# Power settings are not touched.
$ErrorActionPreference = 'Stop'
$ProjectRoot = 'C:\Users\admin\Desktop\EVMALOCALUPDATE'
$RunRoot = Join-Path $ProjectRoot 'execute_results\fast_pretrain_20260925'
$Python = 'C:\Users\admin\AppData\Local\Programs\Python\Python310\python.exe'

Add-Type -TypeDefinition @"
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
public static class FastPretrainCpuJob {
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
$env:EVMA_LOWER_TRAIN_ACCEPT_BANK_AS_IS = '1'
$env:EVMA_LOWER_TRAIN_BUILD_BID_BANK = '0'
$env:EVMA_EPSILON_END_EPISODE = "$EpsilonEnd"
$env:EVMA_OU_NOISE_END_EPISODE = "$NoiseEnd"
$env:EVMA_LR_ACTOR = $LrActor
$env:EVMA_LR_CRITIC_LOCAL = $LrCritic
$env:EVMA_LR_GLOBAL_CRITIC = $LrCritic
$env:EVMA_TAU = $Tau
$env:EVMA_TAU_GLOBAL = $Tau
$env:EVMA_TRAIN_UPDATES_PER_STEP = "$UpdatesPerStep"
$env:EVMA_LOCAL_REWARD_SCALE = $LocalRewardScale
foreach ($unset in 'EVMA_LOCAL_FLEET_RESIDUAL_OBS', 'EVMA_GLOBAL_BALANCE_REWARD_LINEAR_TAIL_KW',
    'EVMA_Q_MIX_GLOBAL_WEIGHT', 'EVMA_GAMMA', 'EVMA_GAMMA_GLOBAL', 'EVMA_MIXER_B_MAX',
    'EVMA_WARMUP_STEPS', 'EVMA_OU_SIGMA', 'EVMA_OU_NOISE_GAIN',
    'EVMA_MARL_ALGORITHM') {
    Remove-Item "Env:$unset" -ErrorAction SilentlyContinue
}

$arguments = @('-u', 'pre_train.py', '--episodes', "$Episodes",
    '--model-name', "prod_f500_dense8_noalloc_fast_$Name",
    '--bank-dir', 'execute_results/bid_banks/train_25_7s_f500_dense8',
    '--test-bank-dir', 'execute_results/bid_banks/validation_5_7s_f500_dense8',
    '--resume-checkpoint-interval', "$CheckpointInterval")
$stdout = Join-Path $RunRoot "$Name.stdout.log"
$stderr = Join-Path $RunRoot "$Name.stderr.log"
$job = [FastPretrainCpuJob]::Create([uint32]$CpuRateBasisPoints)
$proc = Start-Process -FilePath $Python -ArgumentList $arguments -WorkingDirectory $ProjectRoot `
    -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
try { [FastPretrainCpuJob]::Assign($job, $proc.Handle) } catch {
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    throw "Could not place the pretrain under the CPU cap: $_"
}
Write-Host "[pretrain $Name] pid=$($proc.Id) episodes=$Episodes eps_end=$EpsilonEnd noise_end=$NoiseEnd lr_a=$LrActor lr_c=$LrCritic tau=$Tau updates_per_step=$UpdatesPerStep local_reward_scale=$LocalRewardScale started $(Get-Date -Format o)"
$proc.WaitForExit()
Write-Host "[pretrain $Name] exit=$($proc.ExitCode) $(Get-Date -Format o)"
