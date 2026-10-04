"""Activation scenarios sampled from explicit five-minute command files.

The bidder uses the dimensionless ``up_proxy/down_proxy`` utilization pair:
dimensionless command shape. ERCOT proxies use a fixed segment scale, not
instantaneous directional capability or an ERCOT award. Multiplying the
shape by this fleet's directional bids sets its simulated command amplitude.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from concurrent.futures import ThreadPoolExecutor
import glob
import hashlib
import os
from pathlib import Path
import re
import json
from typing import Iterable
import zlib

import numpy as np
import pandas as pd

from EnvConfig import ACTIVATION_SCENARIO_DIR, EPISODE_STEPS
from environment.readcsv import _load_and_normalize_demand_file
from market.bid_env import BASE_DOWN_KW, BASE_UP_KW


@dataclass(frozen=True)
class ActivationScenario:
    name: str
    up_proxy: np.ndarray
    down_proxy: np.ndarray
    weight: float = 1.0
    source: str = ""
    source_bmu: str = ""
    source_date: str = ""

    def as_dict(self) -> dict:
        payload = {
            "name": self.name,
            "up_proxy": np.asarray(self.up_proxy, dtype=np.float64),
            "down_proxy": np.asarray(self.down_proxy, dtype=np.float64),
            "weight": float(self.weight),
            "source": self.source,
        }
        if self.source_bmu:
            payload["source_bmu"] = self.source_bmu
        if self.source_date:
            payload["source_date"] = self.source_date
        return payload


def _parse_date(value) -> date | None:
    try:
        if isinstance(value, date):
            return value
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except Exception:
        try:
            return pd.to_datetime(value, errors="coerce").date()
        except Exception:
            return None


def _season(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


def _day_type(d: date | None) -> tuple[str, str]:
    if d is None:
        return ("unknown", "unknown")
    return (_season(int(d.month)), "weekend" if d.weekday() >= 5 else "weekday")


def stable_seed_for_context(*parts) -> int:
    text = "|".join(str(p) for p in parts)
    return int(zlib.crc32(text.encode("utf-8")) % (2**31))


def _load_proxy_shape_row(path: str) -> dict | None:
    try:
        # Scenario selection needs only the utilization and source fields.
        needed_columns = {
            "up_proxy",
            "down_proxy",
            "up_proxy_raw",
            "down_proxy_raw",
            "source_type",
            "source_date",
            "source_bmu",
            "scenario_partition",
            "segment_id",
        }
        raw = pd.read_csv(
            path,
            usecols=lambda name: str(name) in needed_columns,
        )
        raw_pair = {"up_proxy_raw", "down_proxy_raw"}
        legacy_pair = {"up_proxy", "down_proxy"}
        if raw_pair.issubset(raw.columns) or legacy_pair.issubset(raw.columns):
            up_name, down_name = (
                ("up_proxy_raw", "down_proxy_raw")
                if raw_pair.issubset(raw.columns)
                else ("up_proxy", "down_proxy")
            )
            up_proxy = pd.to_numeric(raw[up_name], errors="coerce").to_numpy(dtype=float)
            down_proxy = pd.to_numeric(raw[down_name], errors="coerce").to_numpy(dtype=float)
            if up_proxy.size < EPISODE_STEPS or down_proxy.size < EPISODE_STEPS:
                raise ValueError("explicit activation file has fewer than 288 rows")
            # Source libraries may retain values beyond one for QA. Saturate
            # only at this simulator-input boundary.
            up_proxy = np.clip(up_proxy[:EPISODE_STEPS], 0.0, 1.0)
            down_proxy = np.clip(down_proxy[:EPISODE_STEPS], 0.0, 1.0)
            source_type = str(
                raw.get("source_type", pd.Series(["explicit_5min"])).iloc[0]
            ).strip()
            if not source_type:
                source_type = "explicit_5min"
        else:
            series = _load_and_normalize_demand_file(path)[:EPISODE_STEPS]
            if series.size < EPISODE_STEPS:
                series = np.pad(series, (0, EPISODE_STEPS - series.size))
            up_proxy = np.clip(-series / max(BASE_UP_KW, 1e-6), 0.0, 1.0)
            down_proxy = np.clip(series / max(BASE_DOWN_KW, 1e-6), 0.0, 1.0)
            source_type = "local_5min"
    except Exception:
        return None
    both = (up_proxy > 0.0) & (down_proxy > 0.0)
    if np.any(both):
        keep_up = up_proxy >= down_proxy
        up_proxy = np.where(both & ~keep_up, 0.0, up_proxy)
        down_proxy = np.where(both & keep_up, 0.0, down_proxy)
    name = os.path.basename(path)
    label = name[:-4]
    if label.startswith("day_"):
        label = label[4:]
    source_date = None
    if "source_date" in raw.columns and not raw["source_date"].empty:
        source_date = str(raw["source_date"].iloc[0])
    if not source_date:
        match = re.search(r"\d{4}-\d{2}-\d{2}", label)
        source_date = match.group(0) if match else label
    d = _parse_date(source_date)
    season, kind = _day_type(d)
    source_bmu = ""
    if "source_bmu" in raw.columns and not raw["source_bmu"].empty:
        source_bmu = str(raw["source_bmu"].iloc[0])
    declared_partition = ""
    if "scenario_partition" in raw:
        partitions = raw["scenario_partition"].dropna().astype(str).unique()
        if len(partitions) != 1 or partitions[0] not in {"train", "validation", "test", "unassigned"}:
            raise ValueError(f"Invalid declared partition in {path}")
        declared_partition = partitions[0]
    return {
        "date": label,
        "path": os.path.abspath(path),
        "source_type": source_type,
        "source_date": source_date,
        "season": season,
        "day_kind": kind,
        "up_proxy": up_proxy,
        "down_proxy": down_proxy,
        "source_bmu": source_bmu,
        "declared_partition": declared_partition,
        "segment_ids": tuple(sorted(set(raw.get("segment_id", pd.Series(dtype=str)).dropna().astype(str)) - {""})),
    }


def _load_proxy_shape_library(directory: str | os.PathLike | None = None) -> pd.DataFrame:
    src_dir = str(directory or ACTIVATION_SCENARIO_DIR)
    metadata_path = Path(src_dir) / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("build_complete") is False:
            raise ValueError(f"Incomplete command library: {src_dir}")
    paths = sorted(glob.glob(os.path.join(src_dir, "*.csv")))
    # Independent files can be parsed concurrently. executor.map preserves the
    # sorted input order, so seeded scenario selection remains bit-for-bit
    # compatible with the former serial loader.
    workers = max(1, min(int(os.environ.get("EVMA_ACTIVATION_LOAD_WORKERS", "8")), 32))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        rows = [row for row in executor.map(_load_proxy_shape_row, paths) if row is not None]
    result = pd.DataFrame(rows)
    # A copied/mixed library must not silently share a segment across splits.
    ownership: dict[str, str] = {}
    for row in rows:
        for segment in row["segment_ids"]:
            previous = ownership.setdefault(segment, row["declared_partition"])
            if previous != row["declared_partition"]:
                raise ValueError(f"Online segment crosses declared partitions: {segment}")
    return result


_PROXY_LIBRARY_CACHE: dict[str, pd.DataFrame] = {}


def load_proxy_shape_library(directory: str | os.PathLike | None = None) -> pd.DataFrame:
    key = os.path.abspath(str(directory or ACTIVATION_SCENARIO_DIR))
    if key not in _PROXY_LIBRARY_CACHE:
        _PROXY_LIBRARY_CACHE[key] = _load_proxy_shape_library(key)
    return _PROXY_LIBRARY_CACHE[key]


def activation_library_signature(
    directory: str | os.PathLike | None = None,
) -> dict[str, object]:
    """Fingerprint the exact command library used to certify an upper bid.

    A bid bank must not silently reuse entries generated from another command
    source. Hash the actual CSV
    content, rather than just the directory name or modification time.
    """

    root = Path(directory or ACTIVATION_SCENARIO_DIR).resolve()
    paths = sorted(root.glob("*.csv")) if root.is_dir() else []
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
    source_types: set[str] = set()
    for path in paths:
        try:
            header = pd.read_csv(path, usecols=lambda name: name == "source_type", nrows=1)
            if "source_type" in header and not header.empty:
                source_types.add(str(header["source_type"].iloc[0]))
        except Exception:
            pass
    metadata: dict[str, object] = {}
    metadata_path = root / "metadata.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}
    mode = "+".join(sorted(source_types)) or "explicit_5min"
    regime = str(metadata.get("regime", "")).strip()
    if regime:
        mode = f"{mode}:{regime}"
    return {
        "activation_signal_mode": mode,
        "activation_signal_regime": regime or None,
        "activation_source_dir": str(root),
        "activation_library_file_count": len(paths),
        "activation_library_sha256": digest.hexdigest(),
    }


# Partition proportions. Two forecast and two feedback days for every holdout
# day, dealt in that repeating order along the difficulty ranking below.
_PARTITION_PATTERN = ("forecast", "feedback", "forecast", "feedback", "holdout")


def _day_difficulty_scores(shape_lib: pd.DataFrame, day_key: pd.Series) -> dict[str, float]:
    """Score each calendar day by how much tracking its commands demand.

    Four properties, averaged over the day's BM Unit files and then z-scored so
    none of them dominates by unit: how often an instruction is present, how
    deep it is, how long the longest unbroken instruction runs, and how often
    the direction reverses. All four make a day harder to follow, so their mean
    orders days from easiest to hardest.
    """

    per_day: dict[str, list[list[float]]] = {}
    for row_index in range(len(shape_lib)):
        up = np.asarray(shape_lib["up_proxy"].iloc[row_index], dtype=float)
        down = np.asarray(shape_lib["down_proxy"].iloc[row_index], dtype=float)
        signed = up - down
        active = np.abs(signed) > 1e-9
        current, longest = 0, 0
        for flag in active:
            current = current + 1 if flag else 0
            longest = max(longest, current)
        sign = np.sign(signed)
        nonzero = sign[sign != 0.0]
        reversals = (
            float(np.count_nonzero(np.diff(nonzero) != 0.0)) if nonzero.size > 1 else 0.0
        )
        per_day.setdefault(str(day_key.iloc[row_index]), []).append([
            float(active.mean()),
            float(np.abs(signed[active]).mean()) if np.any(active) else 0.0,
            float(longest),
            reversals,
        ])

    days = sorted(per_day)
    matrix = np.asarray([np.mean(per_day[day], axis=0) for day in days], dtype=float)
    spread = matrix.std(axis=0)
    spread[spread <= 1e-12] = 1.0
    normalized = (matrix - matrix.mean(axis=0)) / spread
    return {day: float(score) for day, score in zip(days, normalized.mean(axis=1))}


def _stratified_day_partitions(
    shape_lib: pd.DataFrame, day_key: pd.Series
) -> dict[str, set[str]]:
    """Split calendar days so every pool spans the same range of difficulty.

    Splitting on a hash of the date is only unbiased in expectation. With 31
    days it can produce a holdout whose commands are active 29% more of the
    time, run 28% longer unbroken and reverse 25% more often than the days the
    bid was built on -- so the bid is fitted to easy commands and judged on
    hard ones, and the out-of-sample failures partly measure that gap rather
    than the bid.

    Dealing the difficulty-ordered days into the pools makes each pool a
    systematic sample of the same distribution: every five adjacent days in the
    ranking contribute two, two and one. The pattern is rotated once per group,
    because a fixed order would hand the holdout the same slot every time --
    with holdout last that is the hardest day of each group, which made the
    imbalance worse than the hash split it replaced. Rotating moves the holdout
    slot through the whole group, so no pool sits systematically high or low.
    The hash still breaks ties, so the assignment stays deterministic.
    """

    scores = _day_difficulty_scores(shape_lib, day_key)
    ordered = sorted(
        scores,
        key=lambda day: (scores[day], stable_seed_for_context(day, "partition"), day),
    )
    width = len(_PARTITION_PATTERN)
    pools: dict[str, set[str]] = {name: set() for name in ("forecast", "feedback", "holdout")}
    for position, day in enumerate(ordered):
        rotated = (position + position // width) % width
        pools[_PARTITION_PATTERN[rotated]].add(day)
    return pools


_PARTITION_CACHE: dict[str, dict[str, set[str]]] = {}

# The pool the lower controller's rollout commands are drawn from: every
# command except the upper bid's design partition (train). In a library
# without declared partitions it is the whole library.
LOWER_CONTROL_POOL = "lower_control"
_LOWER_CONTROL_DECLARED = ("validation", "test")


def library_declares_partitions(directory: str | os.PathLike | None = None) -> bool:
    declared = load_proxy_shape_library(directory).get("declared_partition", pd.Series(dtype=str))
    return bool(declared.fillna("").ne("").any())


def lower_control_pool_label(directory: str | os.PathLike | None = None) -> str:
    """How LOWER_CONTROL_POOL resolves for this library, for run records."""

    if not library_declares_partitions(directory):
        return "all_historical"
    declared = set(load_proxy_shape_library(directory)["declared_partition"])
    if declared == {"unassigned"}:
        return "unassigned"
    return "+".join(_LOWER_CONTROL_DECLARED)


def _stratified_day_partitions_cached(
    directory, shape_lib: pd.DataFrame, day_key: pd.Series
) -> dict[str, set[str]]:
    """Cache the partition per command library.

    Scoring walks every step of every file, which is a second or two once the
    library covers a year. The library itself is already cached per directory
    and the partition is a pure function of it, so it is computed once.
    """

    key = os.path.abspath(str(directory or ACTIVATION_SCENARIO_DIR))
    if key not in _PARTITION_CACHE:
        _PARTITION_CACHE[key] = _stratified_day_partitions(shape_lib, day_key)
    return _PARTITION_CACHE[key]


def _row_source(row) -> str:
    """The ActivationScenario.source a library row becomes."""

    source_path = str(row.get("path", ""))
    source_name = os.path.basename(source_path) if source_path else str(row["date"])
    return f"{str(row.get('source_type', 'local_5min'))}:{source_name}"


def build_activation_scenario_set(
    service_date: str | date | None = None,
    n_scenarios: int = 8,
    seed: int = 0,
    proxy_shape_dir: str | os.PathLike | None = None,
    scenario_partition: str | None = None,
    require_unique: bool = False,
    exclude_sources: Iterable[str] | None = None,
) -> list[ActivationScenario]:
    """Randomly select complete 5-minute command files.

    Selection is uniform and without replacement while enough files exist.
    Legacy libraries use deterministic 40/40/20 date pools. A library that
    declares its split in ``scenario_partition`` keeps it: forecast=train,
    feedback=validation, holdout=test. The only draw across declared pools is
    LOWER_CONTROL_POOL, which takes every pool except train; in a library
    without declared partitions it is the whole library, drawn exactly as
    ``scenario_partition=None``. Files whose source is in ``exclude_sources``
    are removed from the pool before drawing.
    """
    shape_lib = load_proxy_shape_library(proxy_shape_dir)
    partition = str(scenario_partition or "").strip().lower()
    declared = shape_lib.get("declared_partition", pd.Series(dtype=str)).fillna("")
    is_declared = bool(declared.ne("").any())
    if partition == LOWER_CONTROL_POOL and not is_declared:
        partition, scenario_partition = "", None
    rng = np.random.default_rng(stable_seed_for_context(
        service_date,
        seed,
        "explicit_5min_random",
        scenario_partition or "all",
    ))
    if is_declared:
        if declared.eq("").any():
            raise ValueError("Do not mix libraries with declared partitions and unpartitioned libraries")
        aliases = {"forecast": "train", "feedback": "validation", "holdout": "test"}
        selected_partition = aliases.get(partition, partition)
        if not selected_partition:
            if set(declared) == {"unassigned"}:
                selected_partition = "unassigned"  # exploratory, not a claimed holdout
            else:
                raise ValueError("A library with declared partitions requires an explicit scenario_partition")
        if selected_partition == LOWER_CONTROL_POOL:
            pools = {"unassigned"} if set(declared) == {"unassigned"} else set(_LOWER_CONTROL_DECLARED)
            shape_lib = shape_lib[declared.isin(pools)].reset_index(drop=True)
        elif selected_partition not in {"train", "validation", "test", "unassigned"}:
            raise ValueError(f"unknown activation scenario partition: {scenario_partition!r}")
        else:
            shape_lib = shape_lib[declared == selected_partition].reset_index(drop=True)
    elif partition:
        if partition not in {"forecast", "feedback", "holdout"}:
            raise ValueError(f"unknown activation scenario partition: {scenario_partition!r}")
        ranked = shape_lib.copy()
        # Partition by calendar day, not by file. A single day can contribute
            # several files (one per source unit); every file for
        # that day must land in the same forecast/feedback/holdout pool, or
        # the same realized day would leak across disjoint partitions.
        if "source_date" in ranked.columns:
            day_key = ranked["source_date"].map(lambda value: str(value))
        else:
            day_key = ranked["date"].map(lambda value: str(value))
        day_slices = _stratified_day_partitions_cached(
            proxy_shape_dir, ranked, day_key
        )
        shape_lib = ranked[day_key.isin(day_slices[partition])].reset_index(drop=True)
    excluded = set(exclude_sources or ())
    if excluded:
        keep = [_row_source(shape_lib.iloc[i]) not in excluded for i in range(len(shape_lib))]
        shape_lib = shape_lib[keep].reset_index(drop=True)
    count = max(int(n_scenarios), 1)
    if shape_lib.empty:
        raise RuntimeError(
            f"Activation scenario partition {partition or 'all'} is empty in "
            f"{proxy_shape_dir or ACTIVATION_SCENARIO_DIR}."
        )
    if require_unique and len(shape_lib) < count:
        raise RuntimeError(
            f"Activation scenario partition {partition or 'all'} contains "
            f"{len(shape_lib)} unique files, but {count} distinct commands are required."
        )
    selections: list[np.ndarray] = []
    remaining = count
    while remaining > 0:
        permutation = rng.permutation(len(shape_lib))
        take = min(remaining, len(shape_lib))
        selections.append(permutation[:take])
        remaining -= take
    selected_indices = np.concatenate(selections)
    scenarios: list[ActivationScenario] = []
    for scenario_idx, row_idx in enumerate(np.asarray(selected_indices).reshape(-1)):
        row = shape_lib.iloc[int(row_idx)]
        up = np.asarray(row["up_proxy"], dtype=np.float64).reshape(-1)[:EPISODE_STEPS]
        down = np.asarray(row["down_proxy"], dtype=np.float64).reshape(-1)[:EPISODE_STEPS]
        source_type = str(row.get("source_type", "local_5min"))
        scenarios.append(ActivationScenario(
            name=f"{source_type}_{scenario_idx:02d}",
            up_proxy=np.clip(up, 0.0, 1.0),
            down_proxy=np.clip(down, 0.0, 1.0),
            source=_row_source(row),
            source_bmu=str(row.get("source_bmu", "")),
            source_date=str(row.get("source_date", "")),
        ))
    return scenarios


def scenarios_to_solver_payload(scenarios: Iterable[ActivationScenario]) -> list[dict]:
    return [scenario.as_dict() for scenario in scenarios]
