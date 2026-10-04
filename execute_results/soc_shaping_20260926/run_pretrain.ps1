param(
    [ValidateSet('socshape', 'soclax', 'force')] [string] $Variant = 'socshape',
    [int] $Episodes = 2000,
    [int] $CpuRateBasisPoints = 2500
)
# AB (execute_results/ft_sweep/run_pretrain_AB.sh, archive
# prod_f500_dense8_AB_7station_20260922_022453) with only the station SoC
# shaping changed:
#   EVMA_LOCAL_SHAPING_REDUCTION=sum      per-EV progress is summed, not averaged
#   EVMA_LOCAL_SURPLUS_SHAPING_COEF=0.25  SoC above target is rewarded for shrinking
#   EVMA_LOCAL_SHAPING_CLIP=1.0           the 0.08 clip would bind once summed
# Variant soclax adds, on top of that:
#   EVMA_LOCAL_URGENCY_BASIS=laxity       urgency from the EV's slack before departure
#   EVMA_LOCAL_URGENCY_GAIN=3.0           weight 1 (slack >= 4 h) to 4 (no slack)
#   EVMA_LOCAL_DEPARTURE_REWARD_MODE=smooth  1.5 - 6 f - 12 f^2 for a shortfall fraction f
# Variant force keeps AB's rewards (none of the above) and instead:
#   EVMA_TRAIN_FORCE_CHARGING=1           the execution-time departure force floor
#                                         is applied in training and interim tests
#   EVMA_LOCAL_FORCED_PENALTY_PER_POINT=0.05  per SoC point the floor adds
# Same bank, episodes, checkpoint interval and environment variables as AB. The
# old bank is used as built and never rebuilt. preflight.py must pass before the
# trainer starts. Job Object hard CPU cap in basis points of 32 logical CPUs.
# Power settings are not touched.
$ErrorActionPreference = 'Stop'
$ProjectRoot = 'C:\Users\admin\Desktop\EVMALOCALUPDATE'
$RunRoot = Join-Path $ProjectRoot 'execute_results\soc_shaping_20260926'
$Python = 'C:\Users\admin\AppData\Local\Programs\Python\Python310\python.exe'

Add-Type -TypeDefinition @"
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
public static class SocShapingCpuJob {
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
# AB's variables, as in run_pretrain_AB.sh.
$env:EVMA_USE_CENTRAL_EV_RESIDUAL_ALLOCATOR = '0'
$env:EVMA_ACTOR_EV_COUNT = '1'
$env:EVMA_LOWER_BID_LOOKAHEAD_BLOCKS = '24'
$env:EVMA_GLOBAL_BALANCE_REWARD_MODE = 'bounded_absolute_error'
$env:EVMA_GLOBAL_BALANCE_REWARD_ERROR_SCALE_KW = '150'
$env:EVMA_GLOBAL_BALANCE_REWARD_LINEAR_TAIL_KW = '60'
$env:EVMA_LOCAL_FLEET_RESIDUAL_OBS = '1'
$env:EVMA_GRAD_CLIP_MAX = '5.0'
$env:EVMA_GRAD_CLIP_MAX_GLOBAL = '5.0'
$env:EVMA_ACTIVATION_SCENARIO_DIR = 'data/aemo/nem/processed_5min/dense8'
$env:EVMA_LOWER_TRAIN_RESEARCH_MINIMUM_BID_QUANTITY_KW = '500'
# The bank predates the current bank contract; use it as built.
$env:EVMA_LOWER_TRAIN_ACCEPT_BANK_AS_IS = '1'
$env:EVMA_LOWER_TRAIN_BUILD_BID_BANK = '0'
# The change under test.
foreach ($unset in 'EVMA_LOCAL_SHAPING_REDUCTION', 'EVMA_LOCAL_SURPLUS_SHAPING_COEF', 'EVMA_LOCAL_SHAPING_CLIP',
    'EVMA_LOCAL_URGENCY_BASIS', 'EVMA_LOCAL_URGENCY_GAIN', 'EVMA_LOCAL_DEPARTURE_REWARD_MODE',
    'EVMA_LOCAL_DEPARTURE_SMOOTH_LINEAR', 'EVMA_LOCAL_DEPARTURE_SMOOTH_QUADRATIC',
    'EVMA_TRAIN_FORCE_CHARGING', 'EVMA_TRAIN_FORCE_SLACK_KWH', 'EVMA_LOCAL_FORCED_PENALTY_PER_POINT') {
    Remove-Item "Env:$unset" -ErrorAction SilentlyContinue
}
if ($Variant -ne 'force') {
    $env:EVMA_LOCAL_SHAPING_REDUCTION = 'sum'
    $env:EVMA_LOCAL_SURPLUS_SHAPING_COEF = '0.25'
    $env:EVMA_LOCAL_SHAPING_CLIP = '1.0'
}
$preflightArgs = @()
$model = 'prod_f500_dense8_AB_socshape_7station'
$logStem = 'pretrain'
if ($Variant -eq 'soclax') {
    $env:EVMA_LOCAL_URGENCY_BASIS = 'laxity'
    $env:EVMA_LOCAL_URGENCY_GAIN = '3.0'
    $env:EVMA_LOCAL_DEPARTURE_REWARD_MODE = 'smooth'
    $env:EVMA_LOCAL_DEPARTURE_SMOOTH_LINEAR = '6.0'
    $env:EVMA_LOCAL_DEPARTURE_SMOOTH_QUADRATIC = '12.0'
    $preflightArgs = @('lax')
    $model = 'prod_f500_dense8_AB_soclax_7station'
    $logStem = 'pretrain_lax'
}
if ($Variant -eq 'force') {
    $env:EVMA_TRAIN_FORCE_CHARGING = '1'
    $env:EVMA_TRAIN_FORCE_SLACK_KWH = '0.1'
    $env:EVMA_LOCAL_FORCED_PENALTY_PER_POINT = '0.05'
    $preflightArgs = @('force')
    $model = 'prod_f500_dense8_AB_force_7station'
    $logStem = 'pretrain_force'
}
foreach ($unset in 'EVMA_Q_MIX_GLOBAL_WEIGHT', 'EVMA_GAMMA', 'EVMA_GAMMA_GLOBAL', 'EVMA_MIXER_B_MAX',
    'EVMA_WARMUP_STEPS', 'EVMA_TRAIN_UPDATES_PER_STEP', 'EVMA_OU_SIGMA', 'EVMA_OU_NOISE_GAIN',
    'EVMA_OU_NOISE_END_EPISODE', 'EVMA_EPSILON_END_EPISODE', 'EVMA_LR_ACTOR', 'EVMA_LR_CRITIC_LOCAL',
    'EVMA_LR_GLOBAL_CRITIC', 'EVMA_TAU', 'EVMA_TAU_GLOBAL', 'EVMA_LOCAL_REWARD_SCALE',
    'EVMA_MARL_ALGORITHM', 'EVMA_ACTIVATION_SIGNAL_SET') {
    Remove-Item "Env:$unset" -ErrorAction SilentlyContinue
}

& $Python (Join-Path $RunRoot 'preflight.py') @preflightArgs | Tee-Object -FilePath (Join-Path $RunRoot "preflight_$Variant.log")
if ($LASTEXITCODE -ne 0) { throw "preflight failed (exit $LASTEXITCODE); nothing was started" }

$arguments = @('-u', 'pre_train.py', '--episodes', "$Episodes",
    '--resume-checkpoint-interval', '20',
    '--model-name', $model,
    '--bank-dir', 'execute_results/bid_banks/train_25_7s_f500_dense8',
    '--test-bank-dir', 'execute_results/bid_banks/validation_5_7s_f500_dense8')
$stdout = Join-Path $RunRoot "$logStem.stdout.log"
$stderr = Join-Path $RunRoot "$logStem.stderr.log"
$job = [SocShapingCpuJob]::Create([uint32]$CpuRateBasisPoints)
$proc = Start-Process -FilePath $Python -ArgumentList $arguments -WorkingDirectory $ProjectRoot `
    -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
try { [SocShapingCpuJob]::Assign($job, $proc.Handle) } catch {
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    throw "Could not place the pretrain under the CPU cap: $_"
}
Write-Host "[pretrain AB+$Variant] pid=$($proc.Id) episodes=$Episodes started $(Get-Date -Format o)"
$proc.WaitForExit()
Write-Host "[pretrain AB+$Variant] exit=$($proc.ExitCode) $(Get-Date -Format o)"
