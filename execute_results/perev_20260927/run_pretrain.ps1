param(
    [int] $Episodes = 2000,
    [int] $CpuRateBasisPoints = 2500,
    [int] $AfterPid = 0
)
# The AB-on-AEMO-plan-deviation run (execute_results/newbank_AB_20260926,
# archive prod_aemoplan_AB_7station_20260926_210314) with one change:
#   EVMA_LOCAL_CRITIC_PER_EV=1   each station's local value is the sum of
#                                per-EV values, each EV bootstrapped from itself
# Same bank, rewards, actors, global critic, episodes and interim tests. The
# preflight (newbank_AB_20260926/preflight.py --per-ev) must pass before the
# trainer starts. With -AfterPid the launch waits until that process has exited,
# so at most one other MARL run shares the machine. Job Object hard CPU cap in
# basis points of 32 logical CPUs. Power settings are not touched.
$ErrorActionPreference = 'Stop'
$ProjectRoot = 'C:\Users\admin\Desktop\EVMALOCALUPDATE'
$RunRoot = Join-Path $ProjectRoot 'execute_results\perev_20260927'
$Python = 'C:\Users\admin\AppData\Local\Programs\Python\Python310\python.exe'

if ($AfterPid -gt 0) {
    Write-Host "[per-EV] waiting for pid $AfterPid to exit $(Get-Date -Format o)"
    while (Get-Process -Id $AfterPid -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 30 }
    Write-Host "[per-EV] pid $AfterPid exited $(Get-Date -Format o)"
}

Add-Type -TypeDefinition @"
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
public static class PerEvCpuJob {
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
# AB's variables, as in run_pretrain_AB.sh, without its dense8 and 500 kW overrides.
$env:EVMA_USE_CENTRAL_EV_RESIDUAL_ALLOCATOR = '0'
$env:EVMA_ACTOR_EV_COUNT = '1'
$env:EVMA_LOWER_BID_LOOKAHEAD_BLOCKS = '24'
$env:EVMA_GLOBAL_BALANCE_REWARD_MODE = 'bounded_absolute_error'
$env:EVMA_GLOBAL_BALANCE_REWARD_ERROR_SCALE_KW = '150'
$env:EVMA_GLOBAL_BALANCE_REWARD_LINEAR_TAIL_KW = '60'
$env:EVMA_LOCAL_FLEET_RESIDUAL_OBS = '1'
$env:EVMA_GRAD_CLIP_MAX = '5.0'
$env:EVMA_GRAD_CLIP_MAX_GLOBAL = '5.0'
# The bank, as newbank_AB_20260926.
$env:EVMA_ACTIVATION_SIGNAL_SET = 'aemo_plan_deviation'
$env:EVMA_LOWER_TRAIN_BUILD_BID_BANK = '0'
# The change under test.
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

& $Python (Join-Path $ProjectRoot 'execute_results\newbank_AB_20260926\preflight.py') --per-ev |
    Tee-Object -FilePath (Join-Path $RunRoot 'preflight.log')
if ($LASTEXITCODE -ne 0) { throw "preflight failed (exit $LASTEXITCODE); nothing was started" }

# The sources this run starts from, for an exact resume after later edits.
$backup = Join-Path $RunRoot 'resume_source_backup'
& robocopy $ProjectRoot $backup *.py /S /XD archive execute_results data .git __pycache__ /NFL /NDL /NJH /NJS /NP | Out-Null
if ($LASTEXITCODE -ge 8) { throw "source backup failed (robocopy $LASTEXITCODE)" }

$arguments = @('-u', 'pre_train.py', '--episodes', "$Episodes",
    '--resume-checkpoint-interval', '20',
    '--model-name', 'prod_aemoplan_AB_perev_7station',
    '--bank-dir', 'execute_results/bid_banks/train_25_minmedmax_3of128ev_128cmd_all_commands_aemo_plan_deviation',
    '--test-bank-dir', 'execute_results/bid_banks/validation_5_minmedmax_3of128ev_128cmd_all_commands_aemo_plan_deviation')
$stdout = Join-Path $RunRoot 'pretrain.stdout.log'
$stderr = Join-Path $RunRoot 'pretrain.stderr.log'
$job = [PerEvCpuJob]::Create([uint32]$CpuRateBasisPoints)
$proc = Start-Process -FilePath $Python -ArgumentList $arguments -WorkingDirectory $ProjectRoot `
    -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
try { [PerEvCpuJob]::Assign($job, $proc.Handle) } catch {
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    throw "Could not place the pretrain under the CPU cap: $_"
}
Write-Host "[pretrain AB per-EV local critic on aemo_plan_deviation bank] pid=$($proc.Id) episodes=$Episodes started $(Get-Date -Format o)"
$proc.WaitForExit()
Write-Host "[pretrain AB per-EV local critic on aemo_plan_deviation bank] exit=$($proc.ExitCode) $(Get-Date -Format o)"
