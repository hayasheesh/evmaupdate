"""Adapt a pretrained lower-MARL policy to one fixed next-day bid.

This is the operational, day-ahead entry point.  It computes the specified
day's bid, restores the complete pretrained learner checkpoint, and trains on
many possible activation commands and fresh EV realizations for that day.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Normal run-button settings. If a pretrain exists under ``archive``, running
# ``python fine_tune.py`` discovers its latest complete learner checkpoint.
# Keep the operational adaptation day outside the 25 pretrain and five
# validation dates saved with the default warm start. 2024-12-04 is available
# in the demand archive but is not part of either bank in the current 25+5
# split.
DEFAULT_SERVICE_DAY = "2024-12-04"


def _default_warmstart() -> Path | None:
    """The pretrain run whose learner checkpoint was written most recently.

    This used to be a fixed list of three runs tried in order.  Both preferred
    entries have since been deleted, so it fell through to a five-station run
    that holds actor files and no `agent_state` bundle at all -- a warm start
    that cannot restore a critic or an optimizer, which is the whole point of
    warm-starting a fine-tune.  Looking for the bundle instead of for a name
    means the default follows whatever pretrain last produced, and a run that
    cannot serve as a warm start is never offered as one.
    """

    archive = PROJECT_ROOT / "archive"
    bundles = (
        sorted(archive.glob("*/results/TEST*/agent_state_ep*.pth"),
               key=lambda path: path.stat().st_mtime)
        if archive.is_dir() else []
    )
    # <run>/results/TEST<ep>/agent_state_ep<ep>.pth -> <run>.  The run root lets
    # find_latest_checkpoint pick the episode; pass an exact TEST directory to
    # --warmstart to pin one.
    runs = [bundle.parent.parent.parent for bundle in bundles]
    # A fine-tune writes bundles too, so newest-wins alone would hand the next
    # fine-tune a policy already specialised to somebody else's day -- and while
    # one is running, its own episode 50 outranks the pretrain it came from.
    # Only a pretrain leaves a resume manifest (training_resume.RESUME_MANIFEST
    # under its resume/ directory), which is the difference that matters here:
    # it was trained from scratch rather than warm-started.
    pretrains = [run for run in runs if (run / "resume" / "latest.json").is_file()]
    if pretrains:
        return pretrains[-1]
    # Older runs may predate the resume manifest; accept their complete bundle.
    if runs:
        return runs[-1]
    return None


DEFAULT_WARMSTART = _default_warmstart()
DEFAULT_EPISODES = 300
DEFAULT_FORECAST_SEED = 1_076_030
DEFAULT_NOISE_SCALE = 0.3
DEFAULT_COMMAND_SCENARIOS = int(os.environ.get(
    "EVMA_LOWER_TRAIN_UPPER_BID_ACTIVATION_SCENARIOS", "256"
))
DEFAULT_EVAL_COMMAND_SCENARIOS = max(512, 2 * DEFAULT_COMMAND_SCENARIOS)
DEFAULT_SELECTION_COMMAND_SCENARIOS = 96
DEFAULT_GRAPH_INTERVAL = 50


def _manifest_dates(path: Path) -> set[str]:
    """Read the service dates recorded in one copied bid-bank manifest."""

    if not path.is_file():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    settings = payload.get("settings") or {}
    dates = settings.get("selected_dates")
    if dates is None:
        dates = [entry.get("service_date") for entry in payload.get("entries") or []]
    return {str(value) for value in dates if value not in (None, "")}


def warmstart_seen_dates(warmstart: str | os.PathLike[str]) -> set[str]:
    """Return the pretrain and validation dates attached to a warm start."""

    checkpoint = Path(warmstart).expanduser().resolve()
    for root in (checkpoint, *checkpoint.parents):
        results = root / "results"
        train_manifest = results / "bid_bank_manifest.json"
        test_manifest = results / "bid_bank_test_manifest.json"
        if train_manifest.is_file() or test_manifest.is_file():
            return _manifest_dates(train_manifest) | _manifest_dates(test_manifest)
    raise FileNotFoundError(
        "fine-tune warm start has no copied train/test bid-bank manifests: "
        f"{checkpoint}"
    )


def require_unseen_finetune_day(day: str, warmstart: str | os.PathLike[str]) -> None:
    """Reject a fine-tune day used for pretrain updates or model validation."""

    seen = warmstart_seen_dates(warmstart)
    if str(day) in seen:
        raise ValueError(
            f"fine-tune day {day} occurs in the warm start's 25-day training or "
            "5-day validation bank; choose a held-out service day"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune a pretrained lower-MARL policy for one operating day.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--day",
        default=DEFAULT_SERVICE_DAY,
        help="Operating day in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--warmstart",
        default=(None if DEFAULT_WARMSTART is None else str(DEFAULT_WARMSTART)),
        help="Pretrain run or exact results/TEST* checkpoint directory.",
    )
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--forecast-seed", type=int, default=DEFAULT_FORECAST_SEED)
    parser.add_argument("--noise-scale", type=float, default=DEFAULT_NOISE_SCALE)
    parser.add_argument(
        "--command-scenarios",
        type=int,
        default=DEFAULT_COMMAND_SCENARIOS,
        help="Distinct historical commands used for adaptation.",
    )
    parser.add_argument(
        "--eval-command-scenarios",
        type=int,
        default=DEFAULT_EVAL_COMMAND_SCENARIOS,
        help="Disjoint holdout commands used for evaluation.",
    )
    parser.add_argument(
        "--selection-command-scenarios",
        type=int,
        default=DEFAULT_SELECTION_COMMAND_SCENARIOS,
        help="Fixed subset of holdout commands used to choose a checkpoint.",
    )
    parser.add_argument(
        "--graph-interval",
        type=int,
        default=DEFAULT_GRAPH_INTERVAL,
        help="Episode interval for detailed TEST plots.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Archive name prefix; default is operational_finetune_<day>.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    # A fine-tune is the same workload as a pretrain: hundreds of short
    # device waits an episode, decided by scheduling latency. It ran at
    # normal priority because this call lived in the other entry point.
    from tools.scheduling import hold_scheduler_priority

    hold_scheduler_priority("fine-tune")
    if args.warmstart is None:
        raise FileNotFoundError(
            "no complete pretrain checkpoint was found under archive; "
            "run pre_train.py first or pass --warmstart"
        )
    warmstart = Path(args.warmstart).expanduser().resolve()
    if not warmstart.exists():
        raise FileNotFoundError(f"warm-start checkpoint does not exist: {warmstart}")
    require_unseen_finetune_day(args.day, warmstart)
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive for operational fine-tuning")
    if (
        args.command_scenarios <= 0
        or args.eval_command_scenarios <= 0
        or args.selection_command_scenarios <= 0
    ):
        raise ValueError("command scenario counts must be positive")
    if args.selection_command_scenarios > args.eval_command_scenarios:
        raise ValueError("selection command scenarios cannot exceed final evaluation scenarios")
    if args.graph_interval <= 0:
        raise ValueError("--graph-interval must be positive")

    os.environ["EVMA_FINETUNE_ENABLE"] = "1"
    os.environ["EVMA_FINETUNE_WARMSTART_DIR"] = str(warmstart)
    os.environ["EVMA_FINETUNE_EPISODES"] = str(args.episodes)
    os.environ["EVMA_FINETUNE_NOISE_SCALE"] = str(args.noise_scale)
    os.environ["EVMA_FINETUNE_ACTIVATION_SCENARIOS"] = str(args.command_scenarios)
    os.environ["EVMA_FINETUNE_EVAL_ACTIVATION_SCENARIOS"] = str(
        args.eval_command_scenarios
    )
    os.environ["EVMA_FINETUNE_SELECTION_ACTIVATION_SCENARIOS"] = str(
        args.selection_command_scenarios
    )
    os.environ["EVMA_TRAIN_INTERIM_GRAPH_INTERVAL_EPISODES"] = str(
        args.graph_interval
    )
    os.environ["EVMA_LOWER_TRAIN_USE_BID_BANK"] = "0"

    print(
        "[fine-tune entry] "
        f"day={args.day} warmstart={warmstart} episodes={args.episodes} "
        f"commands={args.command_scenarios} eval_commands={args.eval_command_scenarios} "
        f"selection_commands={args.selection_command_scenarios} "
        f"graph_interval={args.graph_interval}",
        flush=True,
    )

    from training.run_after_day_ahead_bid import run_after_day_ahead_bid

    run_after_day_ahead_bid(
        day=args.day,
        split="all",
        episodes=args.episodes,
        model_name=args.model_name or f"operational_finetune_{args.day}",
        forecast_seed=args.forecast_seed,
        use_train_bid_bank=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
