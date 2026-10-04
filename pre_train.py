"""Train the reusable lower-MARL policy from the multi-day bid bank.

Run this offline when the shared pretrained checkpoint needs to be created or
updated, after tools/build_training_bid_bank.py has built the train and test
banks. The proposed system uses this fixed policy without day-specific
adaptation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pre-train one general lower-MARL policy on the multi-day bid bank."
    )
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument(
        "--model-name",
        default="direct_bid_boa_pretrain_7station",
    )
    parser.add_argument("--forecast-seed", type=int, default=73000)
    parser.add_argument("--train-days", type=int, default=25)
    parser.add_argument(
        "--bank-dir",
        default=None,
        help="Complete train-bank directory; default follows EnvConfig.py.",
    )
    parser.add_argument(
        "--test-bank-dir",
        default=None,
        help="Complete test-bank directory; default follows EnvConfig.py.",
    )
    parser.add_argument(
        "--resume-run",
        default=None,
        help=(
            "Exact pretrain run directory to continue in place. The saved bid-bank, "
            "normalization, source, replay, RNG, and optimizer fingerprints must match."
        ),
    )
    parser.add_argument(
        "--resume-checkpoint-interval",
        type=int,
        default=100,
        help="Rotate an exact resume checkpoint every N learned episodes; 0 disables periodic saves.",
    )
    parser.add_argument(
        "--request-stop",
        default=None,
        metavar="RUN_DIR",
        help="Ask a running pretrain to checkpoint and stop at the next episode boundary.",
    )
    return parser


from tools.scheduling import hold_scheduler_priority


def main() -> int:
    args = build_parser().parse_args()
    if args.request_stop:
        if args.resume_run:
            raise ValueError("--request-stop and --resume-run are mutually exclusive")
        from training.training_resume import request_stop

        marker = request_stop(args.request_stop)
        print(
            f"[resume] stop requested: {marker} "
            "(the trainer will save after its current episode)",
            flush=True,
        )
        return 0

    resume_run = None
    if args.resume_run:
        from training.training_resume import clear_stop_request, read_resume_manifest

        resume_run = Path(args.resume_run).expanduser().resolve()
        manifest = read_resume_manifest(resume_run)
        # Remove the marker that caused the previous clean stop before the new
        # process starts. A request arriving after this point remains visible
        # to the training loop, avoiding a startup race.
        clear_stop_request(resume_run)
        context = manifest.get("context") or {}
        if context.get("kind") != "lower_marl_bid_bank_pretrain":
            raise ValueError(f"unsupported resume run kind: {context.get('kind')!r}")
        saved_bank = Path(context["train_bid_bank"]["path"]).resolve()
        saved_test_bank = Path(context["test_bid_bank"]["path"]).resolve()
        if args.bank_dir and Path(args.bank_dir).expanduser().resolve() != saved_bank:
            raise ValueError("--bank-dir differs from the exact resume manifest")
        if args.test_bank_dir and Path(args.test_bank_dir).expanduser().resolve() != saved_test_bank:
            raise ValueError("--test-bank-dir differs from the exact resume manifest")
        args.bank_dir = str(saved_bank)
        args.test_bank_dir = str(saved_test_bank)
        args.model_name = str(context["model_name"])
        args.forecast_seed = int(context["forecast_seed"])
        args.train_days = int(context["train_split_count"])
        bank_manifest = json.loads(
            (saved_bank / "manifest.json").read_text(encoding="utf-8")
        )
        activation_source = (
            (bank_manifest.get("settings") or {}).get("activation_source_dir")
        )
        if activation_source:
            os.environ[
                "EVMA_LOWER_TRAIN_UPPER_BID_ACTIVATION_SOURCE_DIR"
            ] = str(activation_source)
        print(
            f"[resume] run={resume_run} "
            f"saved_episode={manifest['completed_training_episode']} "
            f"target_episode={args.episodes}",
            flush=True,
        )

    hold_scheduler_priority("pretrain")

    from training.run_after_day_ahead_bid import run_after_day_ahead_bid

    run_after_day_ahead_bid(
        episodes=args.episodes,
        model_name=args.model_name,
        forecast_seed=args.forecast_seed,
        train_split_count=args.train_days,
        bid_bank_dir=args.bank_dir,
        test_bid_bank_dir=args.test_bank_dir,
        resume_run_dir=str(resume_run) if resume_run is not None else None,
        resume_checkpoint_interval=max(0, int(args.resume_checkpoint_interval)),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
