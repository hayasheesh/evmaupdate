"""Set the baseline of blocks without bid width to 0 kW in a finished bid bank.

usage: zero_idle_bid_baseline.py BANK_DIR [BANK_DIR ...]

Banks built before training.bid_bank.zero_idle_baseline joined the builder keep
the bid LP's unconstrained baseline in blocks that carry no bid. For every day
this applies the same rule, rebuilds the day's training-episode info the way the
builder does, and rewrites the day's fixed bid, entry.json and results files.
The merged manifest gets the new entry summaries and the rule's setting. A bank
whose manifest is missing or incomplete is refused: its builder may still be
writing days.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def zero_bank(bank_dir: Path) -> None:
    from training.bid_bank import (
        IDLE_BASELINE_KW,
        _entry_summary,
        _write_json_atomic,
        load_fixed_bid,
        save_fixed_bid,
        zero_idle_baseline,
    )
    from training.lower_bid_training import build_fixed_upper_bid_training_episode
    from training.run_after_day_ahead_bid import write_bid_artifacts

    manifest_path = bank_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"no merged manifest (builder unfinished?): {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("complete"):
        raise ValueError(f"bank is incomplete: {manifest_path}")

    new_entries = {}
    for entry in manifest["entries"]:
        bid_path = bank_dir / entry["bid_path"]
        day_dir = bid_path.parent
        fixed_bid = load_fixed_bid(bid_path)
        changed = zero_idle_baseline(fixed_bid)
        _target, _tol, _arrival, info = build_fixed_upper_bid_training_episode(fixed_bid, 0)
        save_fixed_bid(bid_path, fixed_bid)
        write_bid_artifacts(day_dir, fixed_bid, info)
        updated = dict(entry, summary=_entry_summary(info))
        _write_json_atomic(day_dir / "entry.json", updated)
        new_entries[int(entry["index"])] = updated
        print(f"[zero-idle] {bank_dir.name} {entry['service_date']}: {changed} blocks set to "
              f"{IDLE_BASELINE_KW:g} kW, mean baseline {updated['summary']['bid_mean_baseline_kw']:.1f} kW",
              flush=True)

    manifest["entries"] = [new_entries[int(e["index"])] for e in manifest["entries"]]
    manifest["settings"] = dict(manifest.get("settings") or {}, idle_baseline_kw=IDLE_BASELINE_KW)
    _write_json_atomic(manifest_path, manifest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bank_dirs", nargs="+")
    args = parser.parse_args()
    for bank_dir in args.bank_dirs:
        zero_bank(Path(bank_dir).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
