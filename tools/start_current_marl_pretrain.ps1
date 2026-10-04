param(
    [int]$Episodes = 0,
    [string]$ModelName = "direct_bid_128cmd_3ev_25d_evcount_12hbid_7station",
    [ValidateRange(0, 48)]
    [int]$BidLookaheadBlocks = 24,
    # Run directory of an interrupted pretrain. The resume guard checks the
    # environment this script sets, so a resume has to come through here rather
    # than through a bare python call.
    [string]$ResumeRun = "",
    # An exact state is 1.8 GB and takes about 4 s to write, so a short interval
    # costs under 1% of wall clock and decides how much a crash throws away.
    [int]$ResumeCheckpointInterval = 20
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $projectRoot

# Lock the run to the current 7-station, three-EV-count, all-128-command banks.
$env:EVMA_NUM_STATIONS = "7"
$env:EVMA_LOWER_TRAIN_USE_BID_BANK = "1"
$env:EVMA_LOWER_TRAIN_BUILD_BID_BANK = "0"
$env:EVMA_LOWER_TRAIN_UPPER_BID_ACTIVATION_SCENARIOS = "128"
$env:PYTHONUNBUFFERED = "1"

# Current MARL ablation: retain station EV count in the actor. The local
# critic's realized total EV power input is part of the active architecture.
$env:EVMA_ACTOR_EV_COUNT = "1"
$env:EVMA_LOWER_BID_CONTEXT_OBS = "1"
$env:EVMA_LOWER_BID_LOOKAHEAD_BLOCKS = [string]$BidLookaheadBlocks

$env:EVMA_LOWER_TRAIN_UPPER_BID_EV_SCENARIO_CANDIDATES = "128"
$trainBank = Join-Path $projectRoot "execute_results\bid_banks\train_25_minmedmax_3of128ev_128cmd_all_commands"
$validationBank = Join-Path $projectRoot "execute_results\bid_banks\validation_5_minmedmax_3of128ev_128cmd_all_commands"

$pythonArgs = @(
    "pre_train.py",
    "--episodes", $Episodes,
    "--model-name", $ModelName,
    "--bank-dir", $trainBank,
    "--test-bank-dir", $validationBank,
    "--resume-checkpoint-interval", $ResumeCheckpointInterval
)
if ($ResumeRun -ne "") {
    $pythonArgs += @("--resume-run", (Resolve-Path -LiteralPath $ResumeRun).Path)
}

python $pythonArgs

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
