"""Config-driven driver: one fixed day-ahead bid, then lower-controller training.

This is the shared "forecast -> submitted bid -> lower MARL control" callable.
Use root-level ``pre_train.py`` for multi-day pretraining. The archived
``legacy/finetune/fine_tune.py`` reproduces the retired one-day adaptation.
No env vars or CLI flags are required.

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


def _select_payload(all_data: dict, split: str, day, index: int):
    pools = []
    if split == "all":
        pools.extend(all_data.get("train", []))
        pools.extend(all_data.get("test", []))
    else:
        pools.extend(all_data.get(split, []))
    if not pools:
        raise RuntimeError(f"empty demand split: {split}")
    if day:
        matches = [p for p in pools if str(_payload_date(p)) == str(day)]
        if not matches:
            available = ", ".join(str(_payload_date(p)) for p in pools[:12])
            raise ValueError(f"day {day!r} not found in split={split}; first available: {available}")
        return matches[0]
    idx = int(index)
    if idx < 0:
        idx = len(pools) + idx
    if idx < 0 or idx >= len(pools):
        raise IndexError(f"index {index} out of range for split={split} size={len(pools)}")
    return pools[idx]


def _coerce_series(payload, episode_steps: int):
    if isinstance(payload, dict):
        series = payload.get("series")
        date = payload.get("date")
    else:
        series = payload
        date = None
    arr = np.asarray(series, dtype=np.float32).reshape(-1)
    if arr.size >= episode_steps:
        return arr[:episode_steps], date
    return np.pad(arr, (0, episode_steps - arr.size)), date


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


def _checkpoint_episodes(run_dir) -> list[int]:
    """Every episode the fine-tune saved a full learner bundle for."""

    root = Path(run_dir) / "results"
    if not root.is_dir():
        return []
    episodes = []
    for bundle in root.glob("TEST*/agent_state_ep*.pth"):
        match = re.search(r"agent_state_ep(\d+)\.pth$", bundle.name)
        if match:
            episodes.append(int(match.group(1)))
    return sorted(set(episodes))


def _spread(values: list, limit: int) -> list:
    """At most `limit` entries, evenly spaced, endpoints kept.

    Taking the last few would only find a late peak. A fine-tune that peaks
    early and then degrades is exactly the case this is here to catch.
    """

    if limit <= 0 or len(values) <= limit:
        return list(values)
    if limit == 1:
        return [values[-1]]
    step = (len(values) - 1) / (limit - 1)
    picked = {int(round(i * step)) for i in range(limit)}
    return [values[i] for i in sorted(picked)]


def _score(summary: dict) -> tuple[float, float]:
    """Rank by market tracking, then by departure SoC.

    Both come from the `system` pipeline, which is what the day actually
    delivers. The pre-correction view is recorded separately: the central
    allocator and the battery can carry a worse controller to the same headline,
    and the selection should not be the only place that would notice.
    """

    if not summary:
        return (float("-inf"), float("-inf"))
    return (
        float(summary.get("global_tracking_rate", float("-inf"))),
        float(summary.get("soc_hit_rate", float("-inf"))),
    )


def select_finetuned_model(
    agent,
    fixed_bid,
    *,
    evaluate_fn,
    finetune_dir,
    warmstart_dir,
    warmstart_episode,
    out_dir,
    n_seeds,
    base_seed,
    max_candidates,
    max_activation_scenarios=None,
):
    """Adopt the best checkpoint the fine-tune produced, or the warm start.

    Scored on EV realizations the final evaluation does not use, so the figure
    that picks the model is not the figure that reports it. Selection uses a
    fixed subset of the held-out command bank; the final report uses the whole
    held-out bank.

    Leaves the winner loaded in `agent` and returns the table it chose from.
    """

    from training.agent_checkpoint import initialize_agent_from_checkpoint

    warmstart = {
        "tag": "warmstart",
        "dir": str(warmstart_dir),
        "episode": int(warmstart_episode),
    }
    candidates = [warmstart]
    for episode in _spread(_checkpoint_episodes(finetune_dir), int(max_candidates)):
        candidates.append({
            "tag": f"ft_ep{episode}",
            "dir": str(Path(finetune_dir) / "results" / f"TEST{episode}"),
            "episode": int(episode),
        })

    selection_bid = fixed_bid
    scenarios = list(fixed_bid.get("activation_scenario_payload") or [])
    if max_activation_scenarios is not None and scenarios:
        count = min(max(1, int(max_activation_scenarios)), len(scenarios))
        selection_bid = dict(fixed_bid)
        selection_bid["activation_scenario_payload"] = scenarios[:count]
        selection_bid["activation_scenarios"] = int(count)

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    for candidate in candidates:
        initialize_agent_from_checkpoint(
            agent, candidate["dir"], episode=candidate["episode"]
        )
        candidate["system"] = evaluate_fn(
            agent,
            selection_bid,
            n_seeds=int(n_seeds),
            base_seed=int(base_seed),
            out_dir=str(out_root / candidate["tag"]),
        )
        candidate["score"] = _score(candidate["system"])
        print(
            f"[fine-tune] candidate {candidate['tag']}: "
            f"tracking {candidate['score'][0] * 100:.2f}%  "
            f"SoC {candidate['score'][1] * 100:.2f}%",
            flush=True,
        )

    best = max(candidates, key=lambda c: c["score"])
    initialize_agent_from_checkpoint(agent, best["dir"], episode=best["episode"])

    # Does the headline hold because the controller improved, or because the
    # central allocator and the battery absorbed more? Only the two endpoints
    # need this, and only to answer that question.
    masking = {}
    for candidate in (warmstart, best):
        if candidate["tag"] in masking:
            continue
        initialize_agent_from_checkpoint(
            agent, candidate["dir"], episode=candidate["episode"]
        )
        masking[candidate["tag"]] = evaluate_fn(
            agent,
            selection_bid,
            n_seeds=int(n_seeds),
            base_seed=int(base_seed),
            out_dir=str(out_root / f"{candidate['tag']}_marl_only"),
            evaluation_pipeline="marl_force",
        )
    initialize_agent_from_checkpoint(agent, best["dir"], episode=best["episode"])

    before = float(masking.get("warmstart", {}).get("global_tracking_rate", float("nan")))
    after = float(masking.get(best["tag"], {}).get("global_tracking_rate", float("nan")))
    hidden = bool(
        best["tag"] != "warmstart"
        and after == after
        and before == before
        and after < before
    )
    if hidden:
        print(
            f"[fine-tune] WARNING {best['tag']} wins after correction but the MARL "
            f"layer alone got worse: {before * 100:.2f}% -> {after * 100:.2f}%",
            flush=True,
        )
    if best["tag"] == "warmstart":
        print("[fine-tune] no candidate beat the warm start; keeping it", flush=True)

    return {
        "adopted": best["tag"],
        "adopted_dir": best["dir"],
        "adopted_episode": best["episode"],
        "selection_base_seed": int(base_seed),
        "selection_seeds": int(n_seeds),
        "selection_activation_scenarios": int(
            selection_bid.get("activation_scenarios", len(scenarios) or 1)
        ),
        "marl_only_tracking_before": before,
        "marl_only_tracking_after": after,
        "correction_masked_a_regression": hidden,
        "candidates": [
            {
                "tag": c["tag"],
                "episode": c["episode"],
                "global_tracking_rate": c["score"][0],
                "soc_hit_rate": c["score"][1],
            }
            for c in candidates
        ],
    }


def run_finetune_stage(
    *,
    work_dir,
    test_bid_bank,
    test_bank_payloads,
    arrival_sampler,
    episode_preparer,
    warmstart_dir=None,
    episodes: int = 300,
    noise_scale: float = 0.3,
    warmup_steps: int | None = None,
    eval_seeds: int = 5,
    base_eval_seed: int = 910_000,
    model_name: str = "model",
    observation_normalization_profile=None,
    select_model: bool = True,
    selection_seeds: int = 3,
    selection_base_seed: int = 610_000,
    selection_max_candidates: int = 6,
    selection_activation_scenarios: int = 96,
    train_fn=None,
    evaluate_fn=None,
) -> list[dict]:
    """Per-test-day fine-tune plus paired finetuned/zeroshot precision evaluation.

    For every test-bank day: warm-start from the pretrain checkpoint
    (`warmstart_dir` or `work_dir`), fine-tune `episodes` episodes on that day
    using the exact persisted test-bank bid, then evaluate physical precision for the fine-tuned agent
    under <work_dir>/finetune/<date>/finetuned/ and for the untouched
    warm-start policy under <work_dir>/finetune/<date>/zeroshot/.

    Only one agent instance lives at a time: the warm-start weights are
    reloaded in place for the zero-shot pass, so both evaluations run with
    the same architecture and the same per-day evaluation seed (identical
    EV/command realizations). `train_fn`/`evaluate_fn` are injectable for
    unit tests.
    """

    import gc

    from training.agent_checkpoint import (
        find_latest_checkpoint,
        initialize_agent_from_checkpoint,
    )

    if train_fn is None:
        from training.train import train as train_fn
    if evaluate_fn is None:
        from training.evaluate_controller_precision import (
            evaluate_controller_precision as evaluate_fn,
        )

    warmstart_root = str(warmstart_dir) if warmstart_dir else str(work_dir)
    checkpoint_dir, checkpoint_episode = find_latest_checkpoint(warmstart_root)
    print(
        f"[fine-tune] warm start checkpoint: {checkpoint_dir} ep{checkpoint_episode} "
        f"episodes={int(episodes)} noise_scale={float(noise_scale):.3f}",
        flush=True,
    )

    stage_dir = Path(work_dir) / "finetune"
    stage_dir.mkdir(parents=True, exist_ok=True)
    stage_rows: list[dict] = []
    for day_index, payload in enumerate(test_bank_payloads):
        service_date = str(_payload_date(payload))
        day_dir = stage_dir / _safe_file_stem(service_date)
        day_dir.mkdir(parents=True, exist_ok=True)
        # Deterministic per-day evaluation seed (existing seed + prime idiom);
        # shared by the finetuned and zeroshot passes so they are comparable.
        eval_seed = int(base_eval_seed) + 1009 * int(day_index)
        row: dict = {
            "date": service_date,
            "day_index": int(day_index),
            "warmstart_checkpoint_dir": str(checkpoint_dir),
            "warmstart_checkpoint_episode": int(checkpoint_episode),
            "finetune_episodes": int(episodes),
            "noise_scale": float(noise_scale),
            "model_selection": bool(select_model),
            "eval_base_seed": int(eval_seed),
            "eval_seeds": int(eval_seeds),
        }
        agent = None
        try:
            fixed_bid = test_bid_bank.load_date(service_date)
            print(
                f"[fine-tune] {service_date} ({day_index + 1}/{len(test_bank_payloads)}) "
                f"training {int(episodes)} episodes",
                flush=True,
            )
            agent, _rewards, _perf, _ep_data, _ft_dir = train_fn(
                num_episodes=int(episodes),
                model_name=f"{model_name}_ft_{_safe_file_stem(service_date)}",
                working_dir=str(day_dir / "training"),
                demand_data_override=[payload],
                arrival_sampler_override=arrival_sampler,
                episode_preparer=episode_preparer,
                test_demand_data_override=[payload],
                test_episode_preparer=episode_preparer,
                initial_agent_checkpoint=str(checkpoint_dir),
                exploration_noise_scale=float(noise_scale),
                observation_normalization_profile=observation_normalization_profile,
                warmup_steps=(None if warmup_steps is None else int(warmup_steps)),
            )
            del _rewards, _perf, _ep_data
            # Choose before reporting. Without this the final weights were kept
            # whatever they were, so a fine-tune that ended below its own warm
            # start still became the day's controller.
            if select_model:
                row["selection"] = select_finetuned_model(
                    agent,
                    fixed_bid,
                    evaluate_fn=evaluate_fn,
                    finetune_dir=_ft_dir,
                    warmstart_dir=checkpoint_dir,
                    warmstart_episode=int(checkpoint_episode),
                    out_dir=day_dir / "selection",
                    n_seeds=int(selection_seeds),
                    base_seed=int(selection_base_seed),
                    max_candidates=int(selection_max_candidates),
                    max_activation_scenarios=int(selection_activation_scenarios),
                )
            row["finetuned"] = evaluate_fn(
                agent,
                fixed_bid,
                n_seeds=int(eval_seeds),
                base_seed=int(eval_seed),
                out_dir=str(day_dir / "finetuned"),
            )
            # Zero-shot comparison: restore the untouched warm-start weights
            # into the same instance, then settle under identical conditions.
            initialize_agent_from_checkpoint(
                agent, str(checkpoint_dir), episode=int(checkpoint_episode)
            )
            row["zeroshot"] = evaluate_fn(
                agent,
                fixed_bid,
                n_seeds=int(eval_seeds),
                base_seed=int(eval_seed),
                out_dir=str(day_dir / "zeroshot"),
            )
        except Exception as exc:  # noqa: BLE001 - keep remaining days running
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[fine-tune] {service_date} failed: {row['error']}", flush=True)
        finally:
            if agent is not None:
                del agent
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        with (day_dir / "finetune_day_summary.json").open("w", encoding="utf-8") as f:
            json.dump(row, f, ensure_ascii=False, indent=2, default=str)
        stage_rows.append(row)

    with (stage_dir / "finetune_stage_summary.json").open("w", encoding="utf-8") as f:
        json.dump({
            "warmstart_checkpoint_dir": str(checkpoint_dir),
            "warmstart_checkpoint_episode": int(checkpoint_episode),
            "finetune_episodes": int(episodes),
            "noise_scale": float(noise_scale),
            "days": stage_rows,
        }, f, ensure_ascii=False, indent=2, default=str)
    return stage_rows


def run_after_day_ahead_bid(
    *,
    day=None,
    split=None,
    index=None,
    episodes=None,
    model_name=None,
    forecast_seed=None,
    train_split_count=None,
    use_train_bid_bank=None,
    bid_bank_dir=None,
    test_bid_bank_dir=None,
    resume_run_dir=None,
    resume_checkpoint_interval=100,
    dry_run=False,
):
    """Build one fixed day-ahead bid and train the lower controller on it.

    Every keyword defaults to its EnvConfig `LOWER_TRAIN_UPPER_BID_*` value when
    left as None, so `run_after_day_ahead_bid()` with no args is a pure
    config-driven run. Returns the training work_dir (or the dry-run dir).
    """
    from Config import EPISODE_STEPS
    from EnvConfig import (
        LOWER_TRAIN_UPPER_BID_ACTIVATION_SOURCE_DIR,
        LOWER_TRAIN_UPPER_BID_DAY,
        LOWER_TRAIN_UPPER_BID_SPLIT,
        LOWER_TRAIN_UPPER_BID_INDEX,
        LOWER_TRAIN_UPPER_BID_EPISODES,
        LOWER_TRAIN_UPPER_BID_MODEL_NAME,
        LOWER_TRAIN_UPPER_BID_SEED,
        LOWER_TRAIN_UPPER_BID_TRAIN_SPLIT_COUNT,
        LOWER_TRAIN_UPPER_BID_EVAL_CONTROLLER_PRECISION,
        LOWER_TRAIN_UPPER_BID_EVAL_SEEDS,
        LOWER_TRAIN_UPPER_BID_USE_TRAIN_BANK,
        LOWER_TRAIN_UPPER_BID_BANK_BUILD_MISSING,
        LOWER_TRAIN_ACCEPT_BANK_AS_IS,
        LOWER_TRAIN_UPPER_BID_BANK_DIR,
        LOWER_TRAIN_UPPER_BID_TEST_BANK_COUNT,
        LOWER_TRAIN_UPPER_BID_TEST_BANK_DIR,
        LOWER_TRAIN_UPPER_BID_TEST_BANK_SEED_OFFSET,
        FINETUNE_ENABLE,
        FINETUNE_ACTIVATION_SCENARIOS,
        FINETUNE_EVAL_ACTIVATION_SCENARIOS,
        FINETUNE_EPISODES,
        FINETUNE_NOISE_SCALE,
        FINETUNE_WARMUP_STEPS,
        FINETUNE_WARMSTART_DIR,
        FINETUNE_SELECT_ENABLE,
        FINETUNE_SELECTION_SEEDS,
        FINETUNE_SELECTION_BASE_SEED,
        FINETUNE_SELECTION_ACTIVATION_SCENARIOS,
        FINETUNE_SELECTION_MAX_CANDIDATES,
    )
    from environment.readcsv import load_multiple_demand_files_with_labels
    from environment.EVEnv import EVEnv
    from environment.arrival_context import ArrivalScenarioSampler
    from training.train import train
    from training.lower_bid_training import (
        build_fixed_upper_bid_for_day,
        build_fixed_upper_bid_training_episode,
        _activation_scenarios_for_day,
        sample_random_historical_activation,
        set_upper_bid_progress_log,
        upper_bid_bank_settings,
    )

    # Resolve each argument against its config default.
    day = LOWER_TRAIN_UPPER_BID_DAY if day is None else (day or None)
    split = (LOWER_TRAIN_UPPER_BID_SPLIT if split is None else split)
    index = LOWER_TRAIN_UPPER_BID_INDEX if index is None else int(index)
    episodes = LOWER_TRAIN_UPPER_BID_EPISODES if episodes is None else int(episodes)
    model_name = LOWER_TRAIN_UPPER_BID_MODEL_NAME if model_name is None else str(model_name)
    forecast_seed = LOWER_TRAIN_UPPER_BID_SEED if forecast_seed is None else int(forecast_seed)
    train_split_count = (
        LOWER_TRAIN_UPPER_BID_TRAIN_SPLIT_COUNT if train_split_count is None else int(train_split_count)
    )
    use_train_bid_bank = (
        bool(LOWER_TRAIN_UPPER_BID_USE_TRAIN_BANK)
        if use_train_bid_bank is None else bool(use_train_bid_bank)
    )
    bid_bank_dir = Path(
        LOWER_TRAIN_UPPER_BID_BANK_DIR if bid_bank_dir is None else bid_bank_dir
    ).resolve()
    test_bid_bank_dir = Path(
        LOWER_TRAIN_UPPER_BID_TEST_BANK_DIR
        if test_bid_bank_dir is None
        else test_bid_bank_dir
    ).resolve()

    all_data = (
        service_day_payloads()
        if use_train_bid_bank
        else load_multiple_demand_files_with_labels(train_split=int(train_split_count))
    )
    if use_train_bid_bank:
        from training.bid_bank import (
            BidBank,
            build_training_bid_bank,
            derive_bid_bank_observation_normalization,
            manifest_settings_match,
        )

        arrival_sampler = ArrivalScenarioSampler()
        common_bank_settings = {
            "arrival_model": arrival_sampler.settings_signature(),
            **upper_bid_bank_settings(),
        }

        def inspect_bank_manifest(
            path: Path,
            required_settings: dict,
            *,
            allow_independent_activation_library: bool = False,
        ) -> tuple[bool, bool]:
            """Return (needs_build, incompatible_upper_bid_inputs)."""

            if not path.exists():
                return True, False
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                return True, False
            compatible = manifest_settings_match(payload, required_settings)
            complete = bool(payload.get("complete", False))
            if not compatible and complete and bool(LOWER_TRAIN_ACCEPT_BANK_AS_IS):
                recorded = payload.get("settings") or {}
                differing = sorted(
                    key for key, value in required_settings.items()
                    if recorded.get(key) != value
                )
                print(
                    f"[bid-bank] using {path.parent} as built; its settings differ "
                    f"from the current code in {differing}",
                    flush=True,
                )
                return False, False
            if not compatible and complete and allow_independent_activation_library:
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
            return (not complete) or (not compatible), not compatible

        test_bank_count = max(int(LOWER_TRAIN_UPPER_BID_TEST_BANK_COUNT), 1)
        selection_pool = (
            list(all_data.get("train") or []) + list(all_data.get("test") or [])
        )
        train_payloads, test_payloads, day_selection_info = (
            stratified_bank_day_selection(
                selection_pool,
                train_count=int(train_split_count),
                test_count=test_bank_count,
            )
        )
        day_selection_mode = str(day_selection_info["mode"])
        if not train_payloads:
            raise RuntimeError("training demand split is empty; cannot build bid bank")
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
            "train_split_count": int(train_split_count),
            "base_forecast_seed": int(forecast_seed),
            "day_selection_mode": str(day_selection_mode),
            "selected_dates": [str(_payload_date(p)) for p in train_payloads],
            **common_bank_settings,
        }
        test_bank_settings = {
            "split": "test",
            "selected_days": len(test_payloads),
            "train_split_count": int(train_split_count),
            "base_forecast_seed": (
                int(forecast_seed)
                + int(LOWER_TRAIN_UPPER_BID_TEST_BANK_SEED_OFFSET)
            ),
            "day_selection_mode": str(day_selection_mode),
            "selected_dates": [str(_payload_date(p)) for p in test_payloads],
            **common_bank_settings,
        }
        manifest_path = bid_bank_dir / "manifest.json"
        bank_needs_build, bank_incompatible = inspect_bank_manifest(
            manifest_path, train_bank_settings
        )
        if bank_needs_build:
            if not bool(LOWER_TRAIN_UPPER_BID_BANK_BUILD_MISSING):
                reason = "incompatible upper-bid inputs" if bank_incompatible else "incomplete"
                raise FileNotFoundError(
                    f"compatible complete bid bank not found ({reason}): {manifest_path}"
                )
            print(f"[bid-bank] building/resuming bank at {bid_bank_dir}", flush=True)
            build_training_bid_bank(
                train_payloads,
                bid_bank_dir,
                base_forecast_seed=int(forecast_seed),
                build_fixed_bid=build_fixed_upper_bid_for_day,
                build_episode=build_fixed_upper_bid_training_episode,
                arrival_sampler=arrival_sampler,
                set_progress_log=set_upper_bid_progress_log,
                artifact_writer=write_bid_artifacts,
                settings=train_bank_settings,
                overwrite=bool(bank_incompatible),
            )
        bid_bank = BidBank(bid_bank_dir)
        train_payload_by_date = {
            str(payload.get("date")): payload for payload in train_payloads
        }
        bank_payloads = [
            train_payload_by_date[str(entry["service_date"])] for entry in bid_bank.entries
        ]

        test_manifest_path = test_bid_bank_dir / "manifest.json"
        test_bank_needs_build, test_bank_incompatible = inspect_bank_manifest(
            test_manifest_path,
            test_bank_settings,
            allow_independent_activation_library=True,
        )
        if test_bank_needs_build:
            if not bool(LOWER_TRAIN_UPPER_BID_BANK_BUILD_MISSING):
                reason = (
                    "incompatible upper-bid inputs"
                    if test_bank_incompatible
                    else "incomplete"
                )
                raise FileNotFoundError(
                    f"compatible complete test bid bank not found ({reason}): "
                    f"{test_manifest_path}"
                )
            print(
                f"[bid-bank] building/resuming test bank at {test_bid_bank_dir}",
                flush=True,
            )
            build_training_bid_bank(
                test_payloads,
                test_bid_bank_dir,
                base_forecast_seed=(
                    int(forecast_seed)
                    + int(LOWER_TRAIN_UPPER_BID_TEST_BANK_SEED_OFFSET)
                ),
                build_fixed_bid=build_fixed_upper_bid_for_day,
                build_episode=build_fixed_upper_bid_training_episode,
                arrival_sampler=arrival_sampler,
                set_progress_log=set_upper_bid_progress_log,
                artifact_writer=write_bid_artifacts,
                settings=test_bank_settings,
                overwrite=bool(test_bank_incompatible),
            )
        test_bid_bank = BidBank(test_bid_bank_dir)
        test_payload_by_date = {
            str(payload.get("date")): payload for payload in test_payloads
        }
        test_bank_payloads = [
            test_payload_by_date[str(entry["service_date"])]
            for entry in test_bid_bank.entries
        ]
        sampler = arrival_sampler

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
            train_split_count=int(train_split_count),
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
                arrival_probs = arrival_kwargs.get("arrival_probabilities_by_station")
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
                    arrival_probabilities_by_station=arrival_probs,
                    day_context=arrival_kwargs.get("day_context"),
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

        if dry_run:
            out_dir = PROJECT_ROOT / "execute_results" / "after_bid_bank_dry_run"
            write_bid_bank_artifacts(out_dir, bid_bank)
            write_bid_bank_artifacts(out_dir, test_bid_bank, label="test")
            print(f"[dry-run] verified bid bank under {bid_bank.root}", flush=True)
            return out_dir

        num_episodes = int(episodes) if int(episodes) > 0 else None

        def _bank_interim_hook(_trained_agent, _training_ep, working_dir):
            write_bid_bank_artifacts(Path(working_dir), bid_bank)
            write_bid_bank_artifacts(
                Path(working_dir), test_bid_bank, label="test"
            )

        finetune_kwargs = dict(
            test_bid_bank=test_bid_bank,
            test_bank_payloads=test_bank_payloads,
            arrival_sampler=sampler,
            episode_preparer=test_bank_episode_preparer,
            episodes=int(FINETUNE_EPISODES),
            noise_scale=float(FINETUNE_NOISE_SCALE),
            warmup_steps=int(FINETUNE_WARMUP_STEPS),
            eval_seeds=int(LOWER_TRAIN_UPPER_BID_EVAL_SEEDS),
            model_name=str(model_name),
            observation_normalization_profile=observation_normalization_profile,
            select_model=bool(FINETUNE_SELECT_ENABLE),
            selection_seeds=int(FINETUNE_SELECTION_SEEDS),
            selection_base_seed=int(FINETUNE_SELECTION_BASE_SEED),
            selection_max_candidates=int(FINETUNE_SELECTION_MAX_CANDIDATES),
            selection_activation_scenarios=int(
                FINETUNE_SELECTION_ACTIVATION_SCENARIOS
            ),
        )

        if bool(FINETUNE_ENABLE) and str(FINETUNE_WARMSTART_DIR or "").strip():
            # Reuse an existing pretrained checkpoint dir: skip pretrain and
            # go straight to the per-day fine-tune stage.
            from training.train import create_model_directory

            work_dir = create_model_directory(f"{model_name}_finetune_only")
            write_bid_bank_artifacts(Path(work_dir), bid_bank)
            write_bid_bank_artifacts(Path(work_dir), test_bid_bank, label="test")
            print(
                "[fine-tune] skipping pretrain; warm start from "
                f"{FINETUNE_WARMSTART_DIR}",
                flush=True,
            )
            run_finetune_stage(
                work_dir=work_dir,
                warmstart_dir=str(FINETUNE_WARMSTART_DIR),
                **finetune_kwargs,
            )
            print(f"[done] work_dir={work_dir}", flush=True)
            return work_dir

        agent, all_rewards, perf, ep_data, work_dir = train(
            num_episodes=num_episodes,
            model_name=str(model_name),
            demand_data_override=bank_payloads,
            arrival_sampler_override=sampler,
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
        if bool(FINETUNE_ENABLE):
            run_finetune_stage(
                work_dir=work_dir,
                warmstart_dir=None,
                **finetune_kwargs,
            )
        print(f"[done] work_dir={work_dir}", flush=True)
        return work_dir

    payload = _select_payload(all_data, split, day, index)
    base_series, service_date = _coerce_series(payload, int(EPISODE_STEPS))
    if service_date is None:
        service_date = day or f"{split}_idx{index}"

    sampler = ArrivalScenarioSampler()
    scenario = sampler.scenario_for_day(service_date)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_stem = _safe_file_stem(f"{model_name}_{service_date}_seed{forecast_seed}_{timestamp}")
    upper_log_path = PROJECT_ROOT / "execute_results" / "upper_bid_progress" / f"{log_stem}.log"
    set_upper_bid_progress_log(upper_log_path, reset=True)
    print(f"[upper-bid] progress_log={upper_log_path}", flush=True)

    from EnvConfig import LOWER_TRAIN_BID_SCENARIO_WORKERS

    fixed_bid = build_fixed_upper_bid_for_day(
        base_series,
        service_date,
        arrival_scenario=scenario,
        forecast_seed=int(forecast_seed),
        scenario_workers=max(1, int(LOWER_TRAIN_BID_SCENARIO_WORKERS)),
    )
    fixed_bid["upper_bid_progress_log"] = str(upper_log_path)
    _target, _tol, _arrival_kwargs, bid_info = build_fixed_upper_bid_training_episode(fixed_bid)
    print(
        "[submitted bid] "
        f"day={service_date} seed={forecast_seed} "
        f"activation={bid_info.get('activation_mode')}:{bid_info.get('activation_scenarios')} "
        f"bid={bid_info['bid_status']} feasible={bid_info['bid_feasible']} "
        f"capacity={bid_info['bid_objective_capacity_kw_block']:,.0f}kW-block "
        f"base={bid_info['bid_mean_baseline_kw']:.1f} "
        f"up={bid_info['bid_mean_up_kw']:.1f}/{bid_info['bid_max_up_kw']:.1f} "
        f"down={bid_info['bid_mean_down_kw']:.1f}/{bid_info['bid_max_down_kw']:.1f} "
        f"target=[{bid_info['target_min_kw']:.1f},{bid_info['target_max_kw']:.1f}]",
        flush=True,
    )
    feasibility = fixed_bid.get("bid_feasibility") or {}
    if feasibility:
        certification = feasibility.get("joint_certification") or {}
        validation = certification.get("validation") or {}
        feedback = certification.get("feedback") or {}
        print(
            "[blockwise bid] "
            f"up/down blocks="
            f"{int(feasibility.get('submitted_up_participating_blocks', 0))}/"
            f"{int(feasibility.get('submitted_down_participating_blocks', 0))} "
            f"up=[{float(feasibility.get('submitted_up_nonzero_min_kw', 0.0)):.1f},"
            f"{float(feasibility.get('submitted_up_nonzero_max_kw', 0.0)):.1f}]kW "
            f"down=[{float(feasibility.get('submitted_down_nonzero_min_kw', 0.0)):.1f},"
            f"{float(feasibility.get('submitted_down_nonzero_max_kw', 0.0)):.1f}]kW "
            f"passII up/down="
            f"{float(validation.get('min_up_step_pass_rate', 0.0))*100:.1f}/"
            f"{float(validation.get('min_down_step_pass_rate', 0.0))*100:.1f}% "
            f"feedback_complete={bool(feedback.get('complete', False))}",
            flush=True,
        )
    else:
        print(f"[feasibility] mode={fixed_bid.get('feasibility_mode')}", flush=True)
    if dry_run:
        out_dir = PROJECT_ROOT / "execute_results" / "after_day_ahead_bid_dry_run"
        out_dir.mkdir(parents=True, exist_ok=True)
        write_bid_artifacts(out_dir, fixed_bid, bid_info)
        print(f"[dry-run] wrote bid artifacts under {out_dir / 'results'}", flush=True)
        return out_dir

    single_day_finetune = bool(
        FINETUNE_ENABLE and str(FINETUNE_WARMSTART_DIR or "").strip()
    )
    training_fixed_bid = fixed_bid
    evaluation_fixed_bid = fixed_bid
    if single_day_finetune:
        # The submitted bid is already fixed.  The lower controller therefore
        # needs many possible next-day commands, not another bid optimization.
        #
        # Adaptation uses the commands the bid builder already attached, which
        # are the feedback partition: drawn disjoint from the forecast commands
        # the bid LP was certified against, and checked for overlap where they
        # were drawn.  Re-drawing from forecast here trained the controller on
        # the very commands a physical solution is known to exist for, which
        # flatters tracking and is not what the problem statement specifies.
        train_activations = list(
            fixed_bid.get("activation_scenario_payload") or []
        )
        train_partition = str(
            fixed_bid.get("training_command_partition") or "feedback"
        )
        train_activation_mode = str(
            fixed_bid.get("training_activation_mode")
            or fixed_bid.get("activation_mode")
            or ""
        )
        # --command-scenarios still bounds how many of them are used.
        wanted = max(1, int(FINETUNE_ACTIVATION_SCENARIOS))
        if train_activations and wanted < len(train_activations):
            train_activations = train_activations[:wanted]
        if not train_activations:
            raise RuntimeError(
                "the submitted bid carries no lower-training commands; "
                "expected the feedback partition attached at bid build time"
            )
        eval_activations, eval_activation_mode = _activation_scenarios_for_day(
            service_date,
            int(forecast_seed),
            n_scenarios=max(1, int(FINETUNE_EVAL_ACTIVATION_SCENARIOS)),
            scenario_partition="holdout",
        )
        training_fixed_bid = dict(fixed_bid)
        training_fixed_bid.update({
            "activation_scenario_payload": train_activations,
            "activation_scenarios": int(len(train_activations)),
            "activation_mode": train_activation_mode,
            "activation_scenario_partition": train_partition,
        })
        evaluation_fixed_bid = dict(fixed_bid)
        evaluation_fixed_bid.update({
            "activation_scenario_payload": eval_activations,
            "activation_scenarios": int(len(eval_activations)),
            "activation_mode": eval_activation_mode,
            "activation_scenario_partition": "holdout",
        })
        print(
            "[fine-tune] command scenarios "
            f"train={len(train_activations)} ({train_partition} partition) "
            f"eval={len(eval_activations)} (disjoint holdout partition)",
            flush=True,
        )

    def fixed_bid_episode_preparer(env: EVEnv, _episode_demand, _episode_date, _arrival_sampler, _episode_idx):
        target, tol, arrival_kwargs, info = build_fixed_upper_bid_training_episode(
            training_fixed_bid, _episode_idx
        )
        use_instruction_scale(arrival_kwargs.get("instruction_scale_kw", 1.0))
        env.reset(
            net_demand_series=target,
            tol_narrow_series=tol,
            tracking_enabled_series=arrival_kwargs.get("tracking_enabled_series"),
            market_context_series=arrival_kwargs.get("market_context_series"),
            arrival_probabilities_by_station=arrival_kwargs.get("arrival_probabilities_by_station"),
            day_context=arrival_kwargs.get("day_context"),
            service_date=arrival_kwargs.get("service_date"),
            baseline_series=arrival_kwargs.get("baseline_series"),
        )
        return info

    def fixed_bid_test_episode_preparer(env: EVEnv, _episode_demand, _episode_date, _arrival_sampler, _episode_idx):
        target, tol, arrival_kwargs, info = build_fixed_upper_bid_training_episode(
            evaluation_fixed_bid, _episode_idx
        )
        use_instruction_scale(arrival_kwargs.get("instruction_scale_kw", 1.0))
        env.reset(
            net_demand_series=target,
            tol_narrow_series=tol,
            tracking_enabled_series=arrival_kwargs.get("tracking_enabled_series"),
            market_context_series=arrival_kwargs.get("market_context_series"),
            arrival_probabilities_by_station=arrival_kwargs.get("arrival_probabilities_by_station"),
            day_context=arrival_kwargs.get("day_context"),
            service_date=arrival_kwargs.get("service_date"),
            baseline_series=arrival_kwargs.get("baseline_series"),
        )
        return info

    # episodes <= 0 => no cap: train until manually stopped (Ctrl+C). Physical
    # precision is then reported at each interim interval via the hook below, since
    # train() never returns in that mode.
    num_episodes = int(episodes) if int(episodes) > 0 else None

    # Operational single-day fine-tune: the bid above is now fixed, so restore
    # the complete pretrained learner state and keep its train-bank observation
    # normalization. The ordinary one-day path remains a from-scratch run.
    initial_agent_checkpoint = None
    exploration_noise_scale = None
    observation_normalization_profile = None
    warmstart_checkpoint_episode = None
    finetune_warmup_steps = None
    if single_day_finetune:
        from environment.normalize import load_observation_normalization_for_archive
        from training.agent_checkpoint import find_latest_checkpoint

        initial_agent_checkpoint, warmstart_checkpoint_episode = (
            find_latest_checkpoint(FINETUNE_WARMSTART_DIR)
        )
        observation_normalization_profile = (
            load_observation_normalization_for_archive(FINETUNE_WARMSTART_DIR)
        )
        if observation_normalization_profile is None:
            raise FileNotFoundError(
                "single-day fine-tune requires the pretrained observation "
                "normalization profile under <run>/input: "
                f"{FINETUNE_WARMSTART_DIR}"
            )
        num_episodes = max(1, int(FINETUNE_EPISODES))
        exploration_noise_scale = float(FINETUNE_NOISE_SCALE)
        finetune_warmup_steps = int(FINETUNE_WARMUP_STEPS)
        print(
            "[fine-tune] operational single-day mode "
            f"day={service_date} checkpoint={initial_agent_checkpoint} "
            f"ep{warmstart_checkpoint_episode} episodes={num_episodes} "
            f"noise_scale={exploration_noise_scale:.3f}",
            flush=True,
        )

    evaluate_controller_precision = None
    if LOWER_TRAIN_UPPER_BID_EVAL_CONTROLLER_PRECISION:
        from training.evaluate_controller_precision import evaluate_controller_precision

    def _interim_hook(trained_agent, training_ep, working_dir):
        # Persist the (fixed) submitted bid alongside the run, then settle the
        # current controller's actual tracking under fresh EV realizations from
        # the same day-ahead arrival-probability forecast used for training.
        # Operational fine-tune already runs the normal lightweight one-day
        # checkpoint test. Its full holdout-command x seed settlement is reserved
        # for the final paired comparison; repeating that every 20 episodes
        # would consume most of the pre-operation training window.
        write_bid_artifacts(Path(working_dir), fixed_bid, bid_info)
        if evaluate_controller_precision is not None and not single_day_finetune:
            evaluate_controller_precision(
                trained_agent,
                fixed_bid,
                n_seeds=int(LOWER_TRAIN_UPPER_BID_EVAL_SEEDS),
                out_dir=working_dir,
            )

    agent, all_rewards, perf, ep_data, work_dir = train(
        num_episodes=num_episodes,
        model_name=str(model_name),
        demand_data_override=[payload],
        arrival_sampler_override=sampler,
        episode_preparer=fixed_bid_episode_preparer,
        test_demand_data_override=[payload],
        test_episode_preparer=fixed_bid_test_episode_preparer,
        interim_eval_fn=_interim_hook,
        initial_agent_checkpoint=initial_agent_checkpoint,
        exploration_noise_scale=exploration_noise_scale,
        observation_normalization_profile=observation_normalization_profile,
        warmup_steps=finetune_warmup_steps,
    )
    # Reached only for a bounded run (num_episodes > 0); unlimited runs exit via
    # Ctrl+C after the interim hook has already reported the latest precision.
    del all_rewards, perf, ep_data
    write_bid_artifacts(Path(work_dir), fixed_bid, bid_info)
    if evaluate_controller_precision is not None:
        if single_day_finetune:
            from training.agent_checkpoint import initialize_agent_from_checkpoint

            paired_seed = 910_000
            # The bank path chooses between the checkpoints a fine-tune
            # produced; this path did not, so a run that ended below its own
            # warm start still became the day's controller. Same function, same
            # held-out selection seeds, so "no adaptation" is reachable here too.
            selection = None
            if bool(FINETUNE_SELECT_ENABLE):
                selection = select_finetuned_model(
                    agent,
                    evaluation_fixed_bid,
                    evaluate_fn=evaluate_controller_precision,
                    finetune_dir=work_dir,
                    warmstart_dir=initial_agent_checkpoint,
                    warmstart_episode=int(warmstart_checkpoint_episode),
                    out_dir=Path(work_dir) / "selection",
                    n_seeds=int(FINETUNE_SELECTION_SEEDS),
                    base_seed=int(FINETUNE_SELECTION_BASE_SEED),
                    max_candidates=int(FINETUNE_SELECTION_MAX_CANDIDATES),
                    max_activation_scenarios=int(
                        FINETUNE_SELECTION_ACTIVATION_SCENARIOS
                    ),
                )
            finetuned_summary = evaluate_controller_precision(
                agent,
                evaluation_fixed_bid,
                n_seeds=int(LOWER_TRAIN_UPPER_BID_EVAL_SEEDS),
                base_seed=paired_seed,
                out_dir=str(Path(work_dir) / "finetuned"),
            )
            initialize_agent_from_checkpoint(
                agent,
                str(initial_agent_checkpoint),
                episode=int(warmstart_checkpoint_episode),
            )
            zeroshot_summary = evaluate_controller_precision(
                agent,
                evaluation_fixed_bid,
                n_seeds=int(LOWER_TRAIN_UPPER_BID_EVAL_SEEDS),
                base_seed=paired_seed,
                out_dir=str(Path(work_dir) / "zeroshot"),
            )
            summary_path = Path(work_dir) / "results" / "single_day_finetune_summary.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with summary_path.open("w", encoding="utf-8") as f:
                json.dump({
                    "service_date": str(service_date),
                    "warmstart_checkpoint_dir": str(initial_agent_checkpoint),
                    "warmstart_checkpoint_episode": int(warmstart_checkpoint_episode),
                    "selection": selection,
                    "finetune_episodes": int(num_episodes),
                    "noise_scale": float(exploration_noise_scale),
                    "train_activation_scenarios": int(
                        training_fixed_bid.get("activation_scenarios", 0)
                    ),
                    "train_activation_partition": training_fixed_bid.get(
                        "activation_scenario_partition"
                    ),
                    "eval_activation_scenarios": int(
                        evaluation_fixed_bid.get("activation_scenarios", 0)
                    ),
                    "eval_activation_partition": evaluation_fixed_bid.get(
                        "activation_scenario_partition"
                    ),
                    "eval_base_seed": int(paired_seed),
                    "finetuned": finetuned_summary,
                    "zeroshot": zeroshot_summary,
                }, f, ensure_ascii=False, indent=2, default=str)
        else:
            evaluate_controller_precision(
                agent,
                fixed_bid,
                n_seeds=int(LOWER_TRAIN_UPPER_BID_EVAL_SEEDS),
                out_dir=work_dir,
            )
    del agent
    print(f"[done] work_dir={work_dir}", flush=True)
    return work_dir
