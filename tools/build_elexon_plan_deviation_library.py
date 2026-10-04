"""Build the GB battery command library: instructed BOA level minus FPN.

Stages (all run by default):

select    Battery-like BM Units are ranked by the share of five-minute points
          under an acceptance, on every ``--sample-every``-th train day only,
          so the choice never sees validation or test days. Of the top
          ``--top``, units whose instructed points are mostly SO-flagged
          (share above ``--max-so-share``) are dropped: they are held for
          constraint management, not energy balancing.
download  Acceptances and Physical Notifications of the selected units for the
          whole period, in seven-day requests. Resumable per unit.
build     One 288-point command per unit and London calendar day, divided by
          the unit's fixed width max(generation, |demand|) capacity, unclipped.

Partitions are month-balanced: within each calendar month the first 60% of
days are train, the next 20% validation and the last 20% test, so every
partition holds the same months and so the same mix of seasons.

The Elexon Insights API needs no key. This never builds an upper-bid bank.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from market.command_waveforms import month_balanced_partitions  # noqa: E402
from market.elexon_plan_deviation import (  # noqa: E402
    SOURCE_TYPE, acceptance_in_force, day_activation, local_day_grid, storage_units, to_segments,
)

API = "https://data.elexon.co.uk/bmrs/api/v1"
DEFAULT_RAW = PROJECT_ROOT / "data" / "elexon" / "plan_deviation_raw"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "elexon" / "command_libraries" / "battery_plan_deviation_calendar_day"


def fetch(path: str, params, pause: float) -> list[dict]:
    url = f"{API}{path}?{urlencode(params, doseq=True)}"
    for attempt in range(5):
        try:
            request = Request(url, headers={"User-Agent": "EVMA-LOCAL research downloader/1.0"})
            with urlopen(request, timeout=180) as response:
                payload = json.load(response)
            time.sleep(pause)
            return payload["data"] if isinstance(payload, dict) else payload
        except Exception:
            if attempt == 4:
                raise
            time.sleep(5 * (attempt + 1))
    return []


def utc_bounds(day: date) -> tuple[str, str]:
    grid = local_day_grid(day)
    if grid is None:
        raise ValueError(f"{day} is a DST change day")
    return grid[0].strftime("%Y-%m-%dT%H:%MZ"), (grid[-1] + timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%MZ")


def select_units(days, partitions, units, *, sample_every, top, max_so_share, pause) -> dict:
    train_days = [d for d in days if partitions[d] == "train" and local_day_grid(d) is not None]
    sampled = train_days[::sample_every]
    names = sorted(units)
    instructed = {name: 0 for name in names}
    so_flagged = {name: 0 for name in names}
    points = 0
    for day in sampled:
        start, end = utc_bounds(day)
        rows = []
        for i in range(0, len(names), 40):
            rows += fetch("/datasets/BOALF", {"from": start, "to": end, "bmUnit": names[i:i + 40], "format": "json"}, pause)
        grid = local_day_grid(day)
        points += len(grid)
        by_unit: dict[str, list] = {}
        for row, segment in zip(rows, to_segments(rows, boa=True)):
            by_unit.setdefault(row["bmUnit"], []).append(segment)
        for name, segments in by_unit.items():
            if name not in instructed:
                continue
            chosen = [acceptance_in_force(segments, t) for t in grid]
            instructed[name] += sum(c is not None for c in chosen)
            so_flagged[name] += sum(c is not None and c.so_flag for c in chosen)
        print(f"[select] {day} sampled; units with any acceptance {len(by_unit)}", flush=True)
    share = {n: instructed[n] / points for n in names}
    so_share = {n: so_flagged[n] / instructed[n] if instructed[n] else 0.0 for n in names}
    ranking = sorted(names, key=lambda n: share[n], reverse=True)
    top_units = ranking[:top]
    return {
        "sampled_train_days": [d.isoformat() for d in sampled],
        "ranking": [{"unit": n, "share_under_acceptance": share[n], "so_flagged_share": so_share[n],
                     "width_mw": units[n]} for n in ranking],
        "top": top_units,
        "excluded_so_dominated": [n for n in top_units if so_share[n] > max_so_share],
        "selected": [n for n in top_units if so_share[n] <= max_so_share],
    }


def download_unit(unit: str, first: date, last: date, raw_dir: Path, pause: float) -> Path:
    target = raw_dir / f"{unit}.json"
    if target.exists():
        return target
    boa, pn = [], []
    cursor = first
    while cursor <= last:
        stop = min(cursor + timedelta(days=7), last + timedelta(days=1))
        params = {"bmUnit": unit, "from": f"{cursor}T00:00Z", "to": f"{stop}T00:00Z"}
        boa += fetch("/balancing/acceptances", params, pause)
        pn += [r for r in fetch("/balancing/physical", {**params, "dataset": "PN"}, pause) if r.get("dataset") == "PN"]
        cursor = stop
    key = lambda r: (r["timeFrom"], r["timeTo"], r["levelFrom"], r["levelTo"], r.get("acceptanceNumber"))
    payload = {
        "unit": unit, "from": first.isoformat(), "to": last.isoformat(),
        "acceptances": list({key(r): r for r in boa}.values()),
        "physical_notifications": list({key(r): r for r in pn}.values()),
    }
    partial = target.with_suffix(".json.part")
    partial.write_text(json.dumps(payload), encoding="utf-8")
    partial.replace(target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2025, 12, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 9, 14))
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--sample-every", type=int, default=7)
    parser.add_argument("--max-so-share", type=float, default=0.5)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pause", type=float, default=0.2)
    parser.add_argument("--stages", default="select,download,build")
    args = parser.parse_args()
    stages = set(args.stages.split(","))
    days = [args.start + timedelta(days=i) for i in range((args.end - args.start).days + 1)]
    partitions = month_balanced_partitions(days)
    args.raw_dir.mkdir(parents=True, exist_ok=True)

    reference = fetch("/reference/bmunits/all", {}, args.pause)
    units = storage_units(reference)
    selection_path = args.raw_dir / "selection.json"
    if "select" in stages:
        selection = select_units(
            days, partitions, units, sample_every=args.sample_every, top=args.top,
            max_so_share=args.max_so_share, pause=args.pause,
        )
        selection["candidate_rule"] = "fuel type OTHER or empty, generation and demand capacity within 0.5x-2x"
        selection["candidates"] = len(units)
        selection["max_so_flagged_share"] = args.max_so_share
        selection_path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
        print(f"[select] {len(units)} candidates; excluded {selection['excluded_so_dominated']}; "
              f"selected {selection['selected']}", flush=True)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))

    if "download" in stages:
        for unit in selection["selected"]:
            download_unit(unit, args.start - timedelta(days=1), args.end + timedelta(days=1), args.raw_dir, args.pause)
            print(f"[download] {unit}", flush=True)

    if "build" not in stages:
        return 0
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    reasons = Counter()
    beyond = 0
    so_steps = 0
    instructed_steps = 0
    for unit in selection["selected"]:
        raw = json.loads((args.raw_dir / f"{unit}.json").read_text(encoding="utf-8"))
        boa = to_segments(raw["acceptances"], boa=True)
        fpn = to_segments(raw["physical_notifications"], boa=False)
        for day in days:
            activation, reason = day_activation(
                unit=unit, day=day, boa=boa, fpn=fpn, width_mw=units.get(unit, 0.0), partition=partitions[day],
            )
            if activation is None:
                reasons[reason] += 1
                continue
            activation.to_csv(out / f"elexon_plan_{unit}_{day}.csv", index=False, float_format="%.12g")
            counts[partitions[day]] += 1
            beyond += int(np.sum(np.abs(activation["signed_activation_up_positive_raw"]) > 1.0))
            so_steps += int(activation["so_flag_in_force"].sum())
            instructed_steps += int(activation["acceptance_in_force"].sum())
        print(f"[build] {unit}", flush=True)
    total_steps = sum(counts.values()) * 288
    ranges: dict[str, dict[str, list[str]]] = {}
    for d in days:
        month = ranges.setdefault(d.strftime("%Y-%m"), {})
        span = month.setdefault(partitions[d], [d.isoformat(), d.isoformat()])
        span[1] = d.isoformat()
    metadata = {
        "build_complete": True,
        "bank_ready": all(counts[p] >= 128 for p in ("train", "validation", "test")),
        "regime": SOURCE_TYPE,
        "source_kind": "GB BM: level of the latest Bid-Offer Acceptance in force minus the unit's FPN",
        "plan": "Final Physical Notification, fixed at gate closure one hour before the settlement period",
        "sampling": "spot value at each five-minute point of the London calendar day; DST change days excluded",
        "reference": "zero deviation from FPN; zero when no acceptance is in force",
        "acceptances": "all, whatever the SO flag; units whose instructed points are mostly SO-flagged are not selected",
        "normalization": "one fixed width per unit: max(generationCapacity, |demandCapacity|) from Elexon reference data",
        "clipping": False,
        "interpolation_or_fill": "profiles are linear between Elexon spot points; nothing else is filled",
        "partition_policy": "whole London calendar dates; within each calendar month consecutive 60/20/20 blocks",
        "partition_date_ranges": ranges,
        "partition_counts": dict(counts),
        "excluded_unit_days": dict(reasons),
        "steps_beyond_unit_width": beyond,
        "share_of_steps_under_acceptance": instructed_steps / total_steps if total_steps else 0.0,
        "share_of_instructed_steps_so_flagged": so_steps / instructed_steps if instructed_steps else 0.0,
        "selection": {k: selection[k] for k in (
            "candidate_rule", "candidates", "max_so_flagged_share", "sampled_train_days",
            "top", "excluded_so_dominated", "selected",
        )},
        "unit_width_mw": {u: units.get(u) for u in selection["selected"]},
    }
    (out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[build] partition_counts={dict(counts)} excluded={dict(reasons)} bank_ready={metadata['bank_ready']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
