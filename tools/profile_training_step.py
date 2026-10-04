"""Measure what one training episode actually waits on.

`update()` runs once per environment step and reads a handful of scalars back
from the GPU on the way -- losses, gradient norms, finiteness flags. Every one
of those reads blocks the CPU until the whole queued kernel stream has drained,
so the cost of a read is not the four bytes it copies but the pipeline it
empties. This counts those reads and times what is spent inside them, which is
the number the batching work needs in order to be worth doing.

The timing comes from wrapping `Tensor.cpu`/`.item`/`.tolist` rather than from a
profiler, because a sampling or tracing profiler charges the stall to whichever
Python frame happens to be on top and adds its own overhead to the Python side,
which is the side under suspicion. Wrapping measures the block directly.

Run it against a throwaway model name: it builds its own archive directory and
never touches a live run's state.
"""

from __future__ import annotations

import argparse
import collections
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Episodes to run. The first ones fill the replay buffer; only the "
             "episodes after the warmup exercise the update path.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1200,
        help="New transitions required before updates begin. One episode is "
             "about 1122 steps, so the default starts updating in episode 2.",
    )
    parser.add_argument(
        "--model-name",
        default="profile_sync_cost",
        help="Archive directory prefix. Keep it distinct from any live run.",
    )
    parser.add_argument("--bank-dir", default=None)
    parser.add_argument("--test-bank-dir", default=None)
    parser.add_argument(
        "--top",
        type=int,
        default=15,
        help="How many call sites to list.",
    )
    parser.add_argument(
        "--python-profile",
        action="store_true",
        help="Also run cProfile. It answers the other half of the question -- "
             "where the CPU goes when it is not blocked on the device -- at the "
             "cost of inflating every Python call, so read the two views "
             "separately rather than adding them up.",
    )
    return parser.parse_args()


class SyncMeter:
    """Count and time every device-to-host read, attributed to its call site."""

    def __init__(self) -> None:
        self.calls = collections.Counter()
        self.seconds = collections.Counter()
        self.total_calls = 0
        self.total_seconds = 0.0
        self._installed = []

    def _site(self):
        # Walk out of this file to the first frame that is not the wrapper, so
        # the cost lands on the line that asked for the value.
        frame = sys._getframe(2)
        while frame is not None and frame.f_code.co_filename == __file__:
            frame = frame.f_back
        if frame is None:
            return "<unknown>"
        path = Path(frame.f_code.co_filename)
        try:
            path = path.relative_to(PROJECT_ROOT)
        except ValueError:
            pass
        return f"{path}:{frame.f_lineno} ({frame.f_code.co_name})"

    def wrap(self, owner, name):
        original = getattr(owner, name)

        def wrapper(tensor, *args, **kwargs):
            # A host-side tensor costs nothing to read; only device reads stall.
            if not getattr(tensor, "is_cuda", False):
                return original(tensor, *args, **kwargs)
            start = time.perf_counter()
            try:
                return original(tensor, *args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                site = self._site()
                self.calls[site] += 1
                self.seconds[site] += elapsed
                self.total_calls += 1
                self.total_seconds += elapsed

        setattr(owner, name, wrapper)
        self._installed.append((owner, name, original))

    def restore(self) -> None:
        for owner, name, original in reversed(self._installed):
            setattr(owner, name, original)
        self._installed.clear()

    def reset(self) -> None:
        self.calls.clear()
        self.seconds.clear()
        self.total_calls = 0
        self.total_seconds = 0.0


def main() -> int:
    args = parse_args()

    os.environ["EVMA_NUM_STATIONS"] = os.environ.get("EVMA_NUM_STATIONS", "7")
    os.environ["EVMA_LOWER_TRAIN_USE_BID_BANK"] = "1"
    os.environ["EVMA_LOWER_TRAIN_BUILD_BID_BANK"] = "0"
    os.environ["EVMA_LOWER_TRAIN_UPPER_BID_ACTIVATION_SCENARIOS"] = "128"
    os.environ["EVMA_ACTOR_EV_COUNT"] = "1"
    os.environ["EVMA_LOWER_BID_CONTEXT_OBS"] = "1"
    os.environ.setdefault("EVMA_LOWER_BID_LOOKAHEAD_BLOCKS", "24")
    os.environ["EVMA_FINETUNE_ENABLE"] = "0"
    os.environ["EVMA_WARMUP_STEPS"] = str(int(args.warmup_steps))
    # The interim test is a separate cost and would sit inside the measured
    # window; push it past the end of this run.
    os.environ["EVMA_TRAIN_INTERIM_INTERVAL"] = str(10 * max(1, args.episodes))

    import torch

    if not torch.cuda.is_available():
        print("[profile] CUDA is unavailable; a device-to-host stall cannot be measured here.")
        return 1

    meter = SyncMeter()
    for name in ("cpu", "item", "tolist", "numpy"):
        if hasattr(torch.Tensor, name):
            meter.wrap(torch.Tensor, name)

    from training.run_after_day_ahead_bid import run_after_day_ahead_bid

    profiler = None
    if args.python_profile:
        import cProfile

        profiler = cProfile.Profile()

    started = time.perf_counter()
    try:
        if profiler is not None:
            profiler.enable()
        run_after_day_ahead_bid(
            episodes=int(args.episodes),
            model_name=str(args.model_name),
            use_train_bid_bank=True,
            bid_bank_dir=args.bank_dir,
            test_bid_bank_dir=args.test_bank_dir,
            resume_checkpoint_interval=0,
        )
    finally:
        if profiler is not None:
            profiler.disable()
        wall = time.perf_counter() - started
        meter.restore()

    print()
    print("=" * 78)
    print(f"[profile] wall clock            : {wall:8.1f} s  ({args.episodes} episodes incl. startup)")
    print(f"[profile] device-to-host reads  : {meter.total_calls:8d} calls")
    print(f"[profile] time blocked in them  : {meter.total_seconds:8.1f} s"
          f"  ({100.0 * meter.total_seconds / wall:.1f}% of wall clock)")
    print("=" * 78)
    print(f"{'seconds':>9} {'calls':>9} {'us/call':>9}  call site")
    for site, seconds in meter.seconds.most_common(int(args.top)):
        calls = meter.calls[site]
        print(f"{seconds:9.2f} {calls:9d} {1e6 * seconds / max(calls, 1):9.0f}  {site}")

    if profiler is not None:
        import io
        import pstats

        print()
        print("=" * 78)
        print("[profile] Python time, own cost excluding callees (cProfile inflates calls)")
        print("=" * 78)
        buffer = io.StringIO()
        stats = pstats.Stats(profiler, stream=buffer)
        stats.sort_stats("tottime").print_stats(int(args.top))
        print(buffer.getvalue())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
