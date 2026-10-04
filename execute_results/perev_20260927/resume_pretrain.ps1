param(
    [int] $Episodes = 2000,
    [int] $CpuRateBasisPoints = 2500
)
# Exact resume of archive/prod_aemoplan_AB_perev_7station_20260928_123439 with the
# environment run_pretrain.ps1 started it with. The resume guard compares the
# source files with the fingerprint taken at launch, which resume_source_backup
# matches. Files edited since launch are swapped for their backup copies until
# the trainer reports its restored state (every module is imported and the
# guard has passed by then), and put back afterwards. The earlier logs are kept
# as pretrain_partN.*.log.
$ErrorActionPreference = 'Stop'
$ProjectRoot = 'C:\Users\admin\Desktop\EVMALOCALUPDATE'
$RunRoot = Join-Path $ProjectRoot 'execute_results\perev_20260927'
$RunDir = Join-Path $ProjectRoot 'archive\prod_aemoplan_AB_perev_7station_20260928_123439'
$Backup = Join-Path $RunRoot 'resume_source_backup'
$Python = 'C:\Users\admin\AppData\Local\Programs\Python\Python310\python.exe'
# Every fingerprinted source (training/training_resume.py) plus the command
# library loader, swapped only where it differs from the launch copy.
$Candidates = @('Config.py', 'EnvConfig.py', 'pre_train.py', 'environment\EVEnv.py',
    'environment\central_residual_allocator.py', 'environment\normalize.py', 'environment\observation_config.py',
    'training\Agent\maddpg.py', 'training\Agent\standard_maddpg.py', 'training\Agent\actor.py',
    'training\Agent\critic.py', 'training\Agent\noise.py', 'training\Agent\replay_buffer.py',
    'training\lower_bid_training.py', 'training\run_after_day_ahead_bid.py', 'training\train.py',
    'training\training_resume.py', 'market\activation_scenarios.py')
$Swapped = @($Candidates | Where-Object {
    (Get-FileHash (Join-Path $ProjectRoot $_)).Hash -ne (Get-FileHash (Join-Path $Backup $_)).Hash })
Write-Host "[resume] launch copies swapped in for: $($Swapped -join ', ')"

Add-Type -TypeDefinition @"
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
public static class PerEvResumeCpuJob {
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
# As run_pretrain.ps1.
$env:EVMA_USE_CENTRAL_EV_RESIDUAL_ALLOCATOR = '0'
$env:EVMA_ACTOR_EV_COUNT = '1'
$env:EVMA_LOWER_BID_LOOKAHEAD_BLOCKS = '24'
$env:EVMA_GLOBAL_BALANCE_REWARD_MODE = 'bounded_absolute_error'
$env:EVMA_GLOBAL_BALANCE_REWARD_ERROR_SCALE_KW = '150'
$env:EVMA_GLOBAL_BALANCE_REWARD_LINEAR_TAIL_KW = '60'
$env:EVMA_LOCAL_FLEET_RESIDUAL_OBS = '1'
$env:EVMA_GRAD_CLIP_MAX = '5.0'
$env:EVMA_GRAD_CLIP_MAX_GLOBAL = '5.0'
$env:EVMA_ACTIVATION_SIGNAL_SET = 'aemo_plan_deviation'
$env:EVMA_LOWER_TRAIN_BUILD_BID_BANK = '0'
$env:EVMA_LOCAL_CRITIC_PER_EV = '1'
foreach ($unset in 'EVMA_ACTIVATION_SCENARIO_DIR', 'EVMA_LOWER_TRAIN_RESEARCH_MINIMUM_BID_QUANTITY_KW',
    'EVMA_LOWER_TRAIN_ACCEPT_BANK_AS_IS', 'EVMA_LOWER_TRAIN_BID_BANK_DIR', 'EVMA_LOWER_TRAIN_TEST_BID_BANK_DIR',
    'EVMA_LOCAL_SHAPING_REDUCTION', 'EVMA_LOCAL_SURPLUS_SHAPING_COEF', 'EVMA_LOCAL_SHAPING_CLIP',
    'EVMA_LOCAL_URGENCY_BASIS', 'EVMA_LOCAL_URGENCY_GAIN', 'EVMA_LOCAL_DEPARTURE_REWARD_MODE',
    'EVMA_LOCAL_DEPARTURE_SMOOTH_LINEAR', 'EVMA_LOCAL_DEPARTURE_SMOOTH_QUADRATIC',
    'EVMA_TRAIN_FORCE_CHARGING', 'EVMA_TRAIN_FORCE_SLACK_KWH', 'EVMA_LOCAL_FORCED_PENALTY_PER_POINT',
    'EVMA_Q_MIX_GLOBAL_WEIGHT', 'EVMA_GAMMA', 'EVMA_GAMMA_GLOBAL', 'EVMA_MIXER_B_MAX',
    'EVMA_WARMUP_STEPS', 'EVMA_TRAIN_UPDATES_PER_STEP', 'EVMA_OU_SIGMA', 'EVMA_OU_NOISE_GAIN',
    'EVMA_OU_NOISE_END_EPISODE', 'EVMA_EPSILON_END_EPISODE', 'EVMA_LR_ACTOR', 'EVMA_LR_CRITIC_LOCAL',
    'EVMA_LR_GLOBAL_CRITIC', 'EVMA_TAU', 'EVMA_TAU_GLOBAL', 'EVMA_LOCAL_REWARD_SCALE',
    'EVMA_MARL_ALGORITHM', 'EVMA_LOCAL_REWARD_MODE') {
    Remove-Item "Env:$unset" -ErrorAction SilentlyContinue
}

$stdout = Join-Path $RunRoot 'pretrain.stdout.log'
$stderr = Join-Path $RunRoot 'pretrain.stderr.log'
$part = 1
while (Test-Path (Join-Path $RunRoot "pretrain_part$part.stdout.log")) { $part++ }
Move-Item $stdout (Join-Path $RunRoot "pretrain_part$part.stdout.log")
Move-Item $stderr (Join-Path $RunRoot "pretrain_part$part.stderr.log")

$hold = Join-Path $RunRoot 'current_source_hold'
New-Item -ItemType Directory -Force $hold | Out-Null
foreach ($rel in $Swapped) {
    $name = $rel -replace '\\', '__'
    Copy-Item (Join-Path $ProjectRoot $rel) (Join-Path $hold $name) -Force
    Copy-Item (Join-Path $Backup $rel) (Join-Path $ProjectRoot $rel) -Force
}
$proc = $null
try {
    $arguments = @('-u', 'pre_train.py', '--resume-run', $RunDir, '--episodes', "$Episodes",
        '--resume-checkpoint-interval', '20')
    $job = [PerEvResumeCpuJob]::Create([uint32]$CpuRateBasisPoints)
    $proc = Start-Process -FilePath $Python -ArgumentList $arguments -WorkingDirectory $ProjectRoot `
        -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    try { [PerEvResumeCpuJob]::Assign($job, $proc.Handle) } catch {
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        throw "Could not place the pretrain under the CPU cap: $_"
    }
    Write-Host "[resume AB per-EV local critic] pid=$($proc.Id) started $(Get-Date -Format o)"
    while (-not $proc.HasExited) {
        if ((Test-Path $stdout) -and (Select-String -Path $stdout -Pattern 'exact learner state restored' -Quiet)) { break }
        Start-Sleep -Seconds 5
    }
} finally {
    foreach ($rel in $Swapped) {
        $name = $rel -replace '\\', '__'
        Copy-Item (Join-Path $hold $name) (Join-Path $ProjectRoot $rel) -Force
    }
    Write-Host "[resume] current sources put back $(Get-Date -Format o)"
}
if ($proc -ne $null) {
    $proc.WaitForExit()
    Write-Host "[resume AB per-EV local critic] exit=$($proc.ExitCode) $(Get-Date -Format o)"
}
