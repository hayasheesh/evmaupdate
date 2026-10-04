"""Lower-controller pretraining on a fixed multi-day bid bank.

The "forecast -> submitted bid -> lower MARL control" path: the submitted bids
come from a complete bid bank (tools/build_training_bid_bank.py), and the lower
controller is trained on them. Root-level ``pre_train.py`` is the entry point.

All heavy imports (config, EVEnv, train, bid builder) are deferred into the
function body so the root entry points can configure the run before EnvConfig
is imported and frozen.
"""

from __future__ import annotations

from collections import deque
import csv
from datetime import date, datetime
import json
import math
from pathlib import Path
import re
import shutil

import numpy as np

from environment.normalize import use_instruction_scale


PROJECT_ROOT = Path(__file__).resolve().parents[1]

_BANK_DAY_CLASS_ORDER = ("weekday", "saturday", "sunday_holiday")


def stratified_activation_index(episode_idx, day_count, scenario_count):
    """Traverse day x activation pairs while covering both marginals early.

    With 20 days and 24 scenarios the old day-major order showed warmup only
    scenario 0 (plus five instances of scenario 1).  This schedule exposes all
    20 days and all 24 scenario indices in the first 24 episodes, then covers
    every pair exactly once during the first 20 * 24 episodes.
    """

    day_count = max(int(day_count), 1)
    scenario_count = max(int(scenario_count), 1)
    ordinal = max(int(episode_idx) - 1, 0)
    pair_period = math.lcm(day_count, scenario_count)
    residue_classes = math.gcd(day_count, scenario_count)
    residue_offset = (ordinal // pair_period) % residue_classes
    return int((ordinal + residue_offset) % scenario_count)


def _safe_file_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return stem or "run"


def _payload_date(payload):
    return payload.get("date") if isinstance(payload, dict) else None


def _parse_service_date(value) -> date:
    return datetime.strptime(str(value), "%Y-%m-%d").date()


def weekday_class_for_date(service_date, *, is_holiday=None) -> str:
    """Classify a service date as weekday / saturday / sunday_holiday.

    Any holiday (regardless of weekday) joins the sunday_holiday class, so a
    holiday Monday is not mistaken for a working weekday. Uses the service
    market's calendar (EnvConfig.SERVICE_CALENDAR_COUNTRY) unless a custom
    predicate is injected (tests).
    """

    if is_holiday is None:
        from EnvConfig import SERVICE_CALENDAR_COUNTRY
        from environment.calendars import is_public_holiday

        def is_holiday(day):
            return is_public_holiday(day, SERVICE_CALENDAR_COUNTRY)
    day = service_date if isinstance(service_date, date) else _parse_service_date(service_date)
    if day.weekday() == 6 or bool(is_holiday(day)):
        return "sunday_holiday"
    if day.weekday() == 5:
        return "saturday"
    return "weekday"


def service_day_payloads(year: int | None = None) -> dict:
    """Every calendar day of the service year as a bank-day payload.

    A bank day needs only its date: the EV forecast follows from the date's
    day class, and the commands come from the command library. ``series`` is a
    zero placeholder, used only as the target when a bid has no command.
    """

    from EnvConfig import EPISODE_STEPS, SERVICE_YEAR

    year = int(SERVICE_YEAR if year is None else year)
    first = date(year, 1, 1)
    days = (date(year + 1, 1, 1) - first).days
    payloads = [
        {
            "series": np.zeros(EPISODE_STEPS, dtype=np.float32),
            "date": date.fromordinal(first.toordinal() + k).isoformat(),
            "path": "",
        }
        for k in range(days)
    ]
    return {"train": payloads, "test": []}


def _class_quotas(available: dict[str, int], count: int) -> dict[str, int]:
    """Split ``count`` over day classes in proportion to the available dates.

    Largest remainder, ties broken by _BANK_DAY_CLASS_ORDER; no class gets more
    dates than it has.
    """

    total = sum(available.values())
    if total <= 0 or count <= 0:
        return {cls: 0 for cls in _BANK_DAY_CLASS_ORDER}
    exact = {cls: count * available[cls] / total for cls in _BANK_DAY_CLASS_ORDER}
    quotas = {cls: min(int(math.floor(exact[cls])), available[cls]) for cls in _BANK_DAY_CLASS_ORDER}
    order = sorted(
        _BANK_DAY_CLASS_ORDER,
        key=lambda cls: (-(exact[cls] - math.floor(exact[cls])), _BANK_DAY_CLASS_ORDER.index(cls)),
    )
    while sum(quotas.values()) < count:
        progressed = False
        for cls in order:
            if sum(quotas.values()) >= count:
                break
            if quotas[cls] < available[cls]:
                quotas[cls] += 1
                progressed = True
        if not progressed:
            break
    return quotas


def stratified_bank_day_selection(
    payloads,
    *,
    train_count: int,
    test_count: int,
    is_holiday=None,
) -> tuple[list[dict], list[dict], dict]:
    """Deterministic train/test day selection over the available dates.

    Each selection takes weekday / saturday / sunday-or-holiday dates in
    proportion to how often the classes occur among the available dates
    (largest remainder), so the banks see the calendar's mix of day types.
    Within a class the months are taken in turn, spreading the days over the
    year. Test days are picked first, then train days from the rest. No RNG is
    used: the result is a pure function of the available date list.
    """

    train_count = int(train_count)
    test_count = int(test_count)
    dated: list[tuple[date, dict]] = []
    for payload in payloads:
        raw = _payload_date(payload)
        try:
            day = _parse_service_date(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"stratified bank day selection requires ISO dates, got {raw!r}"
            ) from exc
        dated.append((day, payload))
    dated.sort(key=lambda item: item[0])
    if len(dated) < train_count + test_count:
        raise RuntimeError(
            f"stratified bank day selection needs {train_count + test_count} dates "
            f"({train_count} train + {test_count} test) but only {len(dated)} are available"
        )

    # class -> month cycle + month -> chronological queue of payloads.
    cells: dict[str, dict[int, deque]] = {cls: {} for cls in _BANK_DAY_CLASS_ORDER}
    for day, payload in dated:
        cls = weekday_class_for_date(day, is_holiday=is_holiday)
        cells[cls].setdefault(day.month, deque()).append((day, payload))
    month_cycle = {
        cls: deque(sorted(months.keys())) for cls, months in cells.items()
    }
    available = {
        cls: sum(len(queue) for queue in months.values()) for cls, months in cells.items()
    }

    def take_from_class(cls: str):
        months = month_cycle[cls]
        for _ in range(len(months)):
            month = months[0]
            months.rotate(-1)
            queue = cells[cls][month]
            if queue:
                return queue.popleft()
        return None

    def proportional_pick(count: int) -> list[tuple[date, dict]]:
        remaining = {
            cls: sum(len(queue) for queue in months.values()) for cls, months in cells.items()
        }
        quotas = _class_quotas(remaining, count)
        picked: list[tuple[date, dict]] = []
        for cls in _BANK_DAY_CLASS_ORDER:
            for _ in range(quotas[cls]):
                item = take_from_class(cls)
                if item is not None:
                    picked.append(item)
        return picked

    test_picked = proportional_pick(test_count)
    train_picked = proportional_pick(train_count)
    if len(test_picked) != test_count or len(train_picked) != train_count:
        raise RuntimeError(
            "stratified bank day selection could not fill the requested counts: "
            f"train={len(train_picked)}/{train_count} test={len(test_picked)}/{test_count}"
        )
    train_picked.sort(key=lambda item: item[0])
    test_picked.sort(key=lambda item: item[0])

    def _class_counts(items: list[tuple[date, dict]]) -> dict[str, int]:
        counts = {cls: 0 for cls in _BANK_DAY_CLASS_ORDER}
        for day, _payload in items:
            counts[weekday_class_for_date(day, is_holiday=is_holiday)] += 1
        return counts

    def _cell_count(items: list[tuple[date, dict]]) -> int:
        return len({
            (weekday_class_for_date(day, is_holiday=is_holiday), day.month)
            for day, _payload in items
        })

    selection_info = {
        "mode": "stratified_proportional",
        "available_class_counts": dict(available),
        "train_dates": [day.isoformat() for day, _payload in train_picked],
        "test_dates": [day.isoformat() for day, _payload in test_picked],
        "train_class_counts": _class_counts(train_picked),
        "test_class_counts": _class_counts(test_picked),
        "train_covered_cells": _cell_count(train_picked),
        "test_covered_cells": _cell_count(test_picked),
        "available_dates": len(dated),
    }
    return (
        [payload for _day, payload in train_picked],
        [payload for _day, payload in test_picked],
        selection_info,
    )


def _write_upper_bid_trial_plot(results_dir: Path, trial_records: list[dict]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] could not import matplotlib for upper-bid trial plot: {exc}", flush=True)
        return
    if not trial_records:
        return

    def _f(row, key, default=0.0):
        try:
            return float(row.get(key, default))
        except Exception:
            return float(default)

    def _b(value) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("1", "true", "yes", "y")

    x = np.asarray([_f(r, "mean_up_kw") for r in trial_records], dtype=float)
    y = np.asarray([_f(r, "mean_down_kw") for r in trial_records], dtype=float)
    obj = np.asarray([
        _f(r, "objective_capacity_kw_block") for r in trial_records
    ], dtype=float)
    up_pass = np.asarray([_f(r, "up_pass_rate", 1.0) for r in trial_records], dtype=float)
    down_pass = np.asarray([_f(r, "down_pass_rate", 1.0) for r in trial_records], dtype=float)
    min_pass = np.minimum(up_pass, down_pass)
    certified = np.asarray([_b(r.get("certified", False)) for r in trial_records])
    accepted = np.asarray([str(r.get("decision", "")).lower() in ("accept", "certified", "seed") and _b(r.get("certified", False)) for r in trial_records])
    trial_axis = np.arange(len(trial_records), dtype=int)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    sc = axes[0].scatter(x, y, c=obj, cmap="viridis", s=42, alpha=0.82, linewidths=0)
    if np.any(certified):
        axes[0].scatter(x[certified], y[certified], facecolors="none", edgecolors="black", s=86, linewidths=1.1, label="certified")
    if np.any(accepted):
        axes[0].scatter(x[accepted], y[accepted], facecolors="none", edgecolors="red", s=130, linewidths=1.4, label="accepted/seed")
    axes[0].set_xlabel("mean up bid [kW]")
    axes[0].set_ylabel("mean down bid [kW]")
    axes[0].set_title("MILP H=1 Bid Trial Widths")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")
    fig.colorbar(sc, ax=axes[0], label="offered capacity [kW-block]")

    axes[1].plot(trial_axis, obj, color="tab:blue", linewidth=1.2, label="capacity")
    axes[1].scatter(trial_axis[certified], obj[certified], color="black", s=32, label="certified")
    axes[1].scatter(trial_axis[accepted], obj[accepted], facecolors="none", edgecolors="red", s=80, linewidths=1.2, label="accepted/seed")
    axes[1].set_xlabel("saved trial index")
    axes[1].set_ylabel("offered capacity [kW-block]")
    axes[1].grid(True, alpha=0.25)
    ax2 = axes[1].twinx()
    ax2.plot(trial_axis, min_pass * 100.0, color="tab:orange", linewidth=1.0, label="min direction pass")
    ax2.set_ylabel("min(up, down) pass [%]")
    axes[1].set_title("MILP Trial Capacity And Pass")
    lines1, labels1 = axes[1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1].legend(lines1 + lines2, labels1 + labels2, loc="best")

    fig.savefig(results_dir / "upper_bid_milp_trials.png", dpi=160)
    plt.close(fig)


def write_bid_artifacts(work_dir: Path, fixed_bid: dict, info: dict) -> None:
    results_dir = Path(work_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    for stale_name in ("upper_bid_milp_trials.csv", "upper_bid_milp_trials.png"):
        stale = results_dir / stale_name
        if stale.exists():
            stale.unlink()
    baseline = np.asarray(fixed_bid["baseline_plan"], dtype=float)
    up = np.asarray(fixed_bid["up_plan"], dtype=float)
    down = np.asarray(fixed_bid["down_plan"], dtype=float)
    submitted_up = np.asarray(fixed_bid.get("submitted_up_plan", up), dtype=float)
    submitted_down = np.asarray(fixed_bid.get("submitted_down_plan", down), dtype=float)
    awarded_fraction = np.asarray(
        fixed_bid.get("awarded_fraction_by_block", np.ones(len(up))),
        dtype=float,
    )
    with (results_dir / "submitted_day_ahead_bid.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "block",
            "baseline_kw",
            "submitted_up_kw",
            "submitted_down_kw",
            "awarded_up_kw",
            "awarded_down_kw",
            "awarded_fraction",
            "offered_up_plus_down_kw",
        ])
        for b in range(len(up)):
            writer.writerow([
                b,
                baseline[b],
                submitted_up[b],
                submitted_down[b],
                up[b],
                down[b],
                awarded_fraction[b],
                up[b] + down[b],
            ])
    feasibility = fixed_bid.get("bid_feasibility") or {}
    trial_records = list(((feasibility.get("notes") or {}).get("search_trial_records") or []))
    if trial_records:
        preferred = [
            "stage",
            "trial_no",
            "decision",
            "reason",
            "certified",
            "objective_capacity_kw_block",
            "offered_capacity_kw_block",
            "mean_up_kw",
            "mean_down_kw",
            "max_up_kw",
            "max_down_kw",
            "active_up_blocks",
            "active_down_blocks",
            "up_pass_rate",
            "down_pass_rate",
            "up_pass_lcb",
            "down_pass_lcb",
            "up_pass_count",
            "up_pass_total",
            "down_pass_count",
            "down_pass_total",
            "up_stay_rate",
            "down_stay_rate",
            "global_tracking_rate",
            "soc_hit_rate",
            "soc_hit_lcb",
            "soc_hit_count",
            "soc_hit_total",
            "up_level",
            "down_level",
            "candidate_family",
            "evfleet_policy_charge_fraction_mean",
            "evfleet_policy_departure_pressure_mean",
            "rl_score",
            "rl_sigma_mean",
            "rl_up_bin_mean",
            "rl_down_bin_mean",
            "direction",
            "block",
            "level",
            "trial_value_kw",
            "pruned_up_count",
            "pruned_down_count",
        ]
        keys = []
        for row in trial_records:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        fieldnames = [key for key in preferred if key in keys] + [key for key in keys if key not in preferred]
        with (results_dir / "upper_bid_milp_trials.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in trial_records:
                writer.writerow({key: row.get(key, "") for key in fieldnames})
        _write_upper_bid_trial_plot(results_dir, trial_records)
    serializable = dict(info)
    serializable["result"] = fixed_bid["result"].as_dict(prefix="awarded_bid")
    if fixed_bid.get("submitted_bid_result") is not None:
        serializable["submitted_bid_result"] = fixed_bid[
            "submitted_bid_result"
        ].as_dict(prefix="submitted_bid")
    serializable["award_metadata"] = fixed_bid.get("award_metadata") or {}
    if fixed_bid.get("bid_feasibility") is not None:
        serializable["bid_feasibility"] = fixed_bid.get("bid_feasibility")
    log_src = fixed_bid.get("upper_bid_progress_log")
    if log_src:
        serializable["upper_bid_progress_log"] = str(log_src)
        src_path = Path(str(log_src))
        if src_path.exists():
            shutil.copyfile(src_path, results_dir / "upper_bid_progress.log")
    def json_ready(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(key): json_ready(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_ready(item) for item in value]
        return value

    with (results_dir / "submitted_day_ahead_bid_summary.json").open("w", encoding="utf-8") as f:
        json.dump(json_ready(serializable), f, ensure_ascii=False, indent=2)


def write_bid_bank_artifacts(work_dir: Path, bid_bank, *, label: str = "train") -> None:
    """Record the exact precomputed bid bank used by one lower training run."""

    results_dir = Path(work_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if str(label) == "train" else f"_{_safe_file_stem(str(label))}"
    shutil.copyfile(
        bid_bank.root / "manifest.json",
        results_dir / f"bid_bank{suffix}_manifest.json",
    )
    rows = []
    for entry in bid_bank.entries:
        row = {
            "index": entry.get("index"),
            "service_date": entry.get("service_date"),
            "forecast_seed": entry.get("forecast_seed"),
        }
        row.update(dict(entry.get("summary") or {}))
        rows.append(row)
    if rows:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with (results_dir / f"bid_bank{suffix}_summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def run_after_day_ahead_bid(
    *,
    episodes: int,
    model_name: str,
    train_split_count: int,
    forecast_seed=None,
    bid_bank_dir=None,
    test_bid_bank_dir=None,
    resume_run_dir=None,
    resume_checkpoint_interval=100,
):
    """Train the lower controller on the complete train and test bid banks.

    Both banks must already exist and match the current upper-bid settings;
    tools/build_training_bid_bank.py builds them. ``episodes <= 0`` trains
    until stopped. Returns the training work_dir.
    """
    from EnvConfig import (
        LOWER_TRAIN_UPPER_BID_ACTIVATION_SOURCE_DIR,
        LOWER_TRAIN_UPPER_BID_SEED,
        LOWER_TRAIN_UPPER_BID_BANK_DIR,
        LOWER_TRAIN_UPPER_BID_TEST_BANK_COUNT,
        LOWER_TRAIN_UPPER_BID_TEST_BANK_DIR,
        LOWER_TRAIN_UPPER_BID_TEST_BANK_SEED_OFFSET,
    )
    from environment.EVEnv import EVEnv
    from environment.arrival_context import ArrivalScenarioSampler
    from training.bid_bank import (
        BidBank,
        derive_bid_bank_observation_normalization,
        manifest_missing_settings,
        manifest_settings_match,
    )
    from training.train import train
    from training.lower_bid_training import (
        build_fixed_upper_bid_training_episode,
        ev_population_signature,
        sample_random_historical_activation,
        upper_bid_bank_settings,
    )

    forecast_seed = LOWER_TRAIN_UPPER_BID_SEED if forecast_seed is None else int(forecast_seed)
    train_split_count = int(train_split_count)
    bid_bank_dir = Path(
        LOWER_TRAIN_UPPER_BID_BANK_DIR if bid_bank_dir is None else bid_bank_dir
    ).resolve()
    test_bid_bank_dir = Path(
        LOWER_TRAIN_UPPER_BID_TEST_BANK_DIR
        if test_bid_bank_dir is None
        else test_bid_bank_dir
    ).resolve()

    all_data = service_day_payloads()
    arrival_sampler = ArrivalScenarioSampler()
    common_bank_settings = {
        "arrival_model": arrival_sampler.settings_signature(),
        "ev_population": ev_population_signature(),
        **upper_bid_bank_settings(),
    }

    def require_complete_bank(
        path: Path,
        required_settings: dict,
        *,
        allow_independent_activation_library: bool = False,
    ) -> None:
        """Stop unless ``path`` is a complete bank built under these settings."""

        if not path.exists():
            raise FileNotFoundError(f"bid bank not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        compatible = manifest_settings_match(payload, required_settings)
        if not compatible and allow_independent_activation_library:
            activation_library_keys = {
                "activation_source_dir",
                "activation_library_file_count",
                "activation_library_sha256",
            }
            compatible = manifest_settings_match(
                payload,
                required_settings,
                ignored_keys=activation_library_keys,
            )
            if compatible:
                persisted_source = (payload.get("settings") or {}).get(
                    "activation_source_dir"
                )
                print(
                    "[bid-bank] held-out test bank keeps its independent "
                    f"activation library: {persisted_source}",
                    flush=True,
                )
        if not compatible:
            raise FileNotFoundError(
                f"bid bank was built under other upper-bid inputs: {path}"
            )
        if not bool(payload.get("complete", False)):
            raise FileNotFoundError(f"bid bank is incomplete: {path}")
        unrecorded = manifest_missing_settings(payload, required_settings)
        if unrecorded:
            print(
                f"[bid-bank] {path.parent} does not record {unrecorded}; "
                "it is used without checking them",
                flush=True,
            )

    test_bank_count = max(int(LOWER_TRAIN_UPPER_BID_TEST_BANK_COUNT), 1)
    selection_pool = (
        list(all_data.get("train") or []) + list(all_data.get("test") or [])
    )
    train_payloads, test_payloads, day_selection_info = (
        stratified_bank_day_selection(
            selection_pool,
            train_count=train_split_count,
            test_count=test_bank_count,
        )
    )
    day_selection_mode = str(day_selection_info["mode"])
    if not train_payloads:
        raise RuntimeError("no training days selected; cannot use a bid bank")
    if len(test_payloads) != test_bank_count:
        raise RuntimeError(
            "not enough held-out demand dates for the configured test bid bank"
        )
    print(
        f"[bid-bank] day selection mode={day_selection_mode} "
        f"train={len(train_payloads)} test={len(test_payloads)} "
        f"train_class_counts={day_selection_info.get('train_class_counts')} "
        f"test_class_counts={day_selection_info.get('test_class_counts')}",
        flush=True,
    )
    train_bank_settings = {
        "split": "train",
        "train_split_count": train_split_count,
        "base_forecast_seed": int(forecast_seed),
        "day_selection_mode": str(day_selection_mode),
        "selected_dates": [str(_payload_date(p)) for p in train_payloads],
        **common_bank_settings,
    }
    test_bank_settings = {
        "split": "test",
        "selected_days": len(test_payloads),
        "train_split_count": train_split_count,
        "base_forecast_seed": (
            int(forecast_seed)
            + int(LOWER_TRAIN_UPPER_BID_TEST_BANK_SEED_OFFSET)
        ),
        "day_selection_mode": str(day_selection_mode),
        "selected_dates": [str(_payload_date(p)) for p in test_payloads],
        **common_bank_settings,
    }
    require_complete_bank(bid_bank_dir / "manifest.json", train_bank_settings)
    bid_bank = BidBank(bid_bank_dir)
    train_payload_by_date = {
        str(payload.get("date")): payload for payload in train_payloads
    }
    bank_payloads = [
        train_payload_by_date[str(entry["service_date"])] for entry in bid_bank.entries
    ]

    require_complete_bank(
        test_bid_bank_dir / "manifest.json",
        test_bank_settings,
        allow_independent_activation_library=True,
    )
    test_bid_bank = BidBank(test_bid_bank_dir)
    test_payload_by_date = {
        str(payload.get("date")): payload for payload in test_payloads
    }
    test_bank_payloads = [
        test_payload_by_date[str(entry["service_date"])]
        for entry in test_bid_bank.entries
    ]

    print(
        f"[bid-bank] lower training uses {len(bid_bank)} dates from {bid_bank.root}",
        flush=True,
    )
    print(
        f"[bid-bank] interim test uses {len(test_bid_bank)} held-out dates "
        f"from {test_bid_bank.root}",
        flush=True,
    )
    observation_normalization_profile = (
        derive_bid_bank_observation_normalization(
            bid_bank,
            build_episode=build_fixed_upper_bid_training_episode,
            quantile=0.95,
        )
    )
    from training.training_resume import build_pretrain_resume_context

    resume_context = build_pretrain_resume_context(
        project_root=PROJECT_ROOT,
        model_name=str(model_name),
        forecast_seed=int(forecast_seed),
        train_split_count=train_split_count,
        bid_bank_dir=bid_bank.root,
        test_bid_bank_dir=test_bid_bank.root,
        observation_normalization_profile=observation_normalization_profile,
    )
    # The instruction is scaled by each day's own submitted envelope.  The
    # profile also supplies a fixed train-bank scale for that envelope, so
    # actor and critic can reconstruct physical kW without test leakage.
    print(
        "[normalization] train-bank p95 scales "
        f"demand={observation_normalization_profile['demand_scale_kw']:.3f}kW "
        f"instruction_scale="
        f"{observation_normalization_profile['instruction_scale_scale_kw']:.3f}kW",
        flush=True,
    )

    from Config import INTERIM_TEST_EPISODES
    from training.lower_bid_training import interim_test_command_sources

    # The interim tests draw the same few commands every time; training
    # never draws them, so the test curve is measured on unseen commands.
    interim_command_sources = interim_test_command_sources(
        list(test_bid_bank.entries),
        interim_test_episodes=int(INTERIM_TEST_EPISODES),
        library_dir=LOWER_TRAIN_UPPER_BID_ACTIVATION_SOURCE_DIR,
    )
    print(
        f"[bid-bank] interim-test commands excluded from training draws: "
        f"{sorted(interim_command_sources)}",
        flush=True,
    )

    def make_bank_episode_preparer(
        active_bank,
        *,
        command_stream: str,
        exclude_sources=None,
    ):
        """Build a bank preparer with a fresh draw outside the bid design pool."""

        def prepare(
            env: EVEnv,
            _episode_demand,
            episode_date,
            _arrival_sampler,
            episode_idx,
        ):
            entry = active_bank.entry_for_date(str(episode_date))
            fixed_bid = active_bank.load_entry(entry)
            day_cycle = max(int(episode_idx) - 1, 0) // max(len(active_bank), 1)
            activation_scenario = sample_random_historical_activation(
                fixed_bid,
                int(episode_idx),
                stream=command_stream,
                exclude_sources=exclude_sources,
            )
            target, tol, arrival_kwargs, info = (
                build_fixed_upper_bid_training_episode(
                    fixed_bid,
                    activation_scenario=activation_scenario,
                )
            )
            use_instruction_scale(
                arrival_kwargs.get("instruction_scale_kw", 1.0)
            )
            env.reset(
                net_demand_series=target,
                tol_narrow_series=tol,
                tracking_enabled_series=arrival_kwargs.get(
                    "tracking_enabled_series"
                ),
                market_context_series=arrival_kwargs.get("market_context_series"),
                arrival_probabilities_by_station=arrival_kwargs.get(
                    "arrival_probabilities_by_station"
                ),
                service_date=arrival_kwargs.get("service_date"),
                baseline_series=arrival_kwargs.get("baseline_series"),
            )
            info.update({
                "bid_bank_index": int(entry["index"]),
                "bid_bank_day_cycle": int(day_cycle),
                "bid_bank_dir": str(active_bank.root),
                "lower_command_sampling_pool": activation_scenario.get(
                    "lower_command_sampling_pool"
                ),
                "lower_command_stream": str(command_stream),
            })
            return info

        return prepare

    bank_episode_preparer = make_bank_episode_preparer(
        bid_bank,
        command_stream="train",
        exclude_sources=interim_command_sources,
    )
    test_bank_episode_preparer = make_bank_episode_preparer(
        test_bid_bank,
        command_stream="validation",
    )

    num_episodes = int(episodes) if int(episodes) > 0 else None

    def _bank_interim_hook(_trained_agent, _training_ep, working_dir):
        write_bid_bank_artifacts(Path(working_dir), bid_bank)
        write_bid_bank_artifacts(
            Path(working_dir), test_bid_bank, label="test"
        )

    agent, all_rewards, perf, ep_data, work_dir = train(
        num_episodes=num_episodes,
        model_name=str(model_name),
        demand_data_override=bank_payloads,
        arrival_sampler_override=arrival_sampler,
        episode_preparer=bank_episode_preparer,
        test_demand_data_override=test_bank_payloads,
        test_episode_preparer=test_bank_episode_preparer,
        interim_eval_fn=_bank_interim_hook,
        balanced_demand_sampling=True,
        observation_normalization_profile=observation_normalization_profile,
        resume_run_dir=resume_run_dir,
        resume_context=resume_context,
        resume_checkpoint_interval=int(resume_checkpoint_interval),
    )
    del agent, all_rewards, perf, ep_data
    write_bid_bank_artifacts(Path(work_dir), bid_bank)
    write_bid_bank_artifacts(Path(work_dir), test_bid_bank, label="test")
    print(f"[done] work_dir={work_dir}", flush=True)
    return work_dir
