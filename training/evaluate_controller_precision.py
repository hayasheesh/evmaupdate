"""Physical precision evaluation for a submitted day-ahead bid.

This module rolls out a lower controller under fresh EV realizations and
activation scenarios, then evaluates each 30-minute block. Assessment I,
Assessment II and EV SoC are evaluated without any monetary input. A co-located BESS contributes only
when the selected evaluation pipeline actually includes it, and its sustained
30-minute capability is added explicitly rather than bypassing Assessment I.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from market.bid_participation import (
    masked_pass_rate,
    participation_by_block,
    participation_by_step,
)


EVALUATION_PIPELINES = {
    # Proposed operational system. Station-level control stays decentralised;
    # the PCC battery absorbs only the remaining aggregate error.
    "system": {
        "force_layer": True,
        "central_allocator": False,
        "residual_bess": True,
        "response_source": "pcc",
        "use_actor": True,
    },
    # C4: the force layer is part of the intended MARL controller.  The two
    # downstream rescue layers are excluded so its trajectory is independent.
    "marl_force": {
        "force_layer": True,
        "central_allocator": False,
        "residual_bess": False,
        "response_source": "ev",
        "use_actor": True,
    },
    # The two rungs between marl_force and system, so the ladder from the bare
    # actor to the deployed hierarchy can be read one layer at a time.
    "marl_raw": {
        "force_layer": False,
        "central_allocator": False,
        "residual_bess": False,
        "response_source": "ev",
        "use_actor": True,
    },
    # 完全分散の EV 制御に、連系点の BESS だけを足した構成。中央残差配分は
    # 局あたり11スカラーを要求するが、BESS は合計電力スカラー1個しか見ない。
    "marl_force_bess": {
        "force_layer": True,
        "central_allocator": False,
        "residual_bess": True,
        "response_source": "pcc",
        "use_actor": True,
    },
    # Central-observation rule baseline. It receives station envelopes and
    # directly assigns station targets; no learned actor contributes actions.
    "rule_based_central": {
        "force_layer": True,
        "central_allocator": True,
        "residual_bess": False,
        "response_source": "ev",
        "use_actor": False,
    },
}


def _assessment_i_unmet_ratio(
    awarded_kw: float, supply_capacity_kw_per_step
) -> tuple[float, float]:
    """Return worst-step 30-minute capability deficit for one direction."""

    supply = np.asarray(supply_capacity_kw_per_step, dtype=float).reshape(-1)
    if supply.size == 0 or np.any(~np.isfinite(supply)):
        raise ValueError("Assessment-I supply trace must be finite and non-empty")
    supply_30 = float(np.min(supply))
    awarded = max(float(awarded_kw), 0.0)
    if awarded <= 0.0:
        return 0.0, supply_30
    return float(np.clip((awarded - supply_30) / awarded, 0.0, 1.0)), supply_30


def evaluation_pipeline_settings(name: str) -> dict[str, object]:
    try:
        return dict(EVALUATION_PIPELINES[str(name)])
    except KeyError as exc:
        valid = ", ".join(sorted(EVALUATION_PIPELINES))
        raise ValueError(f"unknown evaluation_pipeline {name!r}; valid: {valid}") from exc


def _nanmean_or_nan(values) -> float:
    arr = np.asarray(list(values), dtype=float).reshape(-1)
    finite = np.isfinite(arr)
    return float(np.mean(arr[finite])) if np.any(finite) else float("nan")


def _block_series(block_values: np.ndarray, steps_per_block: int, n_steps: int) -> np.ndarray:
    arr = np.asarray(block_values, dtype=float).reshape(-1)
    out = np.repeat(arr, int(steps_per_block))
    if out.size < int(n_steps):
        out = np.pad(out, (0, int(n_steps) - out.size), mode="edge")
    return out[: int(n_steps)]


def directional_block_tracking(
    dispatch_kw_per_step,
    response_kw_per_step,
    *,
    baseline_kw: float,
    tolerance_kw_per_step,
    stay_rate_threshold: float = 0.90,
    eps: float = 1e-6,
) -> dict[str, float | int | bool]:
    """Classify and score one block's up, down, and idle instructions.

    EV up regulation lowers grid import relative to the baseline, while down
    regulation raises it. Empty direction masks pass vacuously and are marked
    unassessed so callers do not count them as active direction blocks.
    """

    dispatch = np.asarray(dispatch_kw_per_step, dtype=float).reshape(-1)
    response = np.asarray(response_kw_per_step, dtype=float).reshape(-1)
    tolerance = np.asarray(tolerance_kw_per_step, dtype=float).reshape(-1)
    if not (dispatch.shape == response.shape == tolerance.shape):
        raise ValueError(
            "dispatch/response/tolerance shape mismatch: "
            f"{dispatch.shape}, {response.shape}, {tolerance.shape}"
        )

    delta = dispatch - float(baseline_kw)
    passed = np.abs(dispatch - response) <= tolerance + float(eps)
    masks = {
        "up": delta < -float(eps),
        "down": delta > float(eps),
    }
    masks["idle"] = ~(masks["up"] | masks["down"])

    result: dict[str, float | int | bool] = {}
    for direction, mask in masks.items():
        assessed_steps = int(np.count_nonzero(mask))
        stay_rate = float(np.mean(passed[mask])) if assessed_steps else 1.0
        result[f"{direction}_assessed"] = bool(assessed_steps)
        result[f"{direction}_active_steps"] = assessed_steps
        result[f"{direction}_passed_steps"] = int(np.count_nonzero(passed & mask))
        result[f"{direction}_stay_rate"] = stay_rate
        result[f"{direction}_pass_II"] = bool(
            stay_rate + 1e-12 >= float(stay_rate_threshold)
        )
    return result


def direction_step_metrics(
    dispatch_kw_per_step,
    response_kw_per_step,
    baseline_kw_per_step,
    tolerance_kw_per_step,
    participation_step_mask,
    *,
    steps_per_block: int,
    eps: float = 1e-6,
) -> dict[str, float | list[int]]:
    """Classify every participating step by command-delta sign and score it.

    Mirrors the ``up_step_pass_rate`` / ``down_step_pass_rate`` /
    ``idle_step_pass_rate`` semantics of ``validate_joint_solution``
    (market/physical_lp_bidding/joint_validation.py): a step is up-active
    when ``dispatch - baseline < -eps``, down-active when ``> eps``, and idle
    otherwise. Only steps inside a participating block (``bid_participation.
    participation_by_step``) are assessed; steps outside participation are
    ignored entirely so they cannot inflate or dilute any direction's rate.

    A pure, numpy-only function (no env dependency) so it is unit-testable
    on its own.

    Returns a dict with, for each of "up" / "down" / "idle":
      - ``{direction}_active_steps``   assessed step count
      - ``{direction}_step_pass_rate`` mean pass rate over assessed steps
        (vacuously 1.0 if none assessed)
      - ``{direction}_step_failed_blocks`` sorted list of block indices that
        contain at least one missed step of that direction (a block may
        appear under more than one direction).
    """

    dispatch = np.asarray(dispatch_kw_per_step, dtype=float).reshape(-1)
    response = np.asarray(response_kw_per_step, dtype=float).reshape(-1)
    baseline = np.asarray(baseline_kw_per_step, dtype=float).reshape(-1)
    tolerance = np.asarray(tolerance_kw_per_step, dtype=float).reshape(-1)
    participation = np.asarray(participation_step_mask, dtype=bool).reshape(-1)
    n = dispatch.shape[0]
    if not (
        response.shape[0] == baseline.shape[0] == tolerance.shape[0]
        == participation.shape[0] == n
    ):
        raise ValueError(
            "dispatch/response/baseline/tolerance/participation length mismatch: "
            f"{dispatch.shape}, {response.shape}, {baseline.shape}, "
            f"{tolerance.shape}, {participation.shape}"
        )

    delta = dispatch - baseline
    passed = np.abs(dispatch - response) <= tolerance + float(eps)

    masks = {
        "up": participation & (delta < -float(eps)),
        "down": participation & (delta > float(eps)),
    }
    masks["idle"] = participation & ~(masks["up"] | masks["down"])

    steps_per_block = int(steps_per_block)
    block_index = (
        np.arange(n) // steps_per_block if steps_per_block > 0 else np.zeros(n, dtype=int)
    )

    result: dict[str, float | list[int]] = {}
    for direction, mask in masks.items():
        assessed_steps = int(np.count_nonzero(mask))
        result[f"{direction}_active_steps"] = assessed_steps
        result[f"{direction}_step_pass_rate"] = (
            float(np.mean(passed[mask])) if assessed_steps else 1.0
        )
        missed = mask & ~passed
        result[f"{direction}_step_failed_blocks"] = sorted(
            {int(b) for b in block_index[missed]}
        )
    return result


def _write_visual_artifacts(results_dir: Path, payload: dict) -> None:
    """Write representative controller-precision plots and CSVs."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"realized_s{int(payload['scenario']):02d}_seed{int(payload['seed']):02d}"
    steps = np.arange(1, int(payload["n_steps"]) + 1)
    blocks = np.asarray(payload["blocks"], dtype=int)
    target = np.asarray(payload["dispatch_kw"], dtype=float)
    response = np.asarray(payload["response_kw"], dtype=float)
    ev_response = np.asarray(payload.get("ev_response_kw", response), dtype=float)
    controller_pre_system_response = np.asarray(
        payload.get("controller_pre_system_response_kw", ev_response), dtype=float
    )
    bess_power = np.asarray(payload.get("bess_power_kw", np.zeros_like(response)), dtype=float)
    bess_soc = np.asarray(payload.get("bess_soc_pct", np.zeros_like(response)), dtype=float)
    baseline_step = np.asarray(payload["baseline_step_kw"], dtype=float)
    tol = np.asarray(payload["tol_kw"], dtype=float)
    regulation = target - baseline_step
    response_delta = response - baseline_step
    sup_c = np.asarray(payload["supply_charge_kw"], dtype=float)
    sup_d = np.asarray(payload["supply_discharge_kw"], dtype=float)
    forced = np.asarray(payload["forced_count"], dtype=float)
    arrivals = np.asarray(payload["arrivals_by_station"], dtype=float)
    station_powers = np.asarray(payload["station_powers_kw"], dtype=float)

    step_csv = results_dir / f"{prefix}_step_trace.csv"
    station_cols = [f"station_{i + 1}_kw" for i in range(station_powers.shape[1])]
    arrival_cols = [f"arrivals_station_{i + 1}" for i in range(arrivals.shape[1])]
    with step_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "step",
            "block",
            "baseline_kw",
            "dispatch_kw",
            "response_kw",
            "ev_response_kw",
            "controller_pre_system_response_kw",
            "bess_power_kw",
            "bess_soc_pct",
            "regulation_kw",
            "response_delta_kw",
            "tol_kw",
            "supply_charge_kw",
            "supply_discharge_kw",
            "forced_count",
            "arrivals_total",
            *station_cols,
            *arrival_cols,
        ])
        for i in range(len(steps)):
            writer.writerow([
                int(steps[i]),
                int(blocks[i]),
                float(baseline_step[i]),
                float(target[i]),
                float(response[i]),
                float(ev_response[i]),
                float(controller_pre_system_response[i]),
                float(bess_power[i]),
                float(bess_soc[i]),
                float(regulation[i]),
                float(response_delta[i]),
                float(tol[i]),
                float(sup_c[i]),
                float(sup_d[i]),
                float(forced[i]),
                int(arrivals[i].sum()) if arrivals.size else 0,
                *[float(x) for x in station_powers[i, :]],
                *[int(x) for x in arrivals[i, :]],
            ])

    block_rows = list(payload["block_rows"])
    block_csv = results_dir / f"{prefix}_precision_by_block.csv"
    with block_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(block_rows[0].keys()) if block_rows else ["block"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in block_rows:
            writer.writerow(row)

    bid_csv = results_dir / "submitted_bid_profile.csv"
    with bid_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "block", "baseline_kw", "up_kw", "down_kw", "up_plus_down_kw"
        ])
        for b, base, up, down in zip(
            payload["block_index"],
            payload["baseline_plan_kw"],
            payload["up_plan_kw"],
            payload["down_plan_kw"],
        ):
            writer.writerow([
                int(b), float(base), float(up), float(down), float(up + down)
            ])

    fig, axes = plt.subplots(3, 1, figsize=(18, 14), sharex=True)
    axes[0].step(steps, target, where="mid", color="black", linewidth=2.4, label="dispatch target")
    axes[0].plot(steps, response, color="tab:purple", linewidth=2.0, label="EV response")
    axes[0].plot(steps, baseline_step, color="tab:gray", linewidth=1.8, linestyle="--", label="baseline")
    axes[0].fill_between(steps, target - tol, target + tol, color="tab:gray", alpha=0.18, label="assessment II band")
    axes[0].set_ylabel("Power [kW]")
    axes[0].legend(loc="upper right", ncol=2, fontsize=10)
    axes[0].grid(alpha=0.3)

    axes[1].step(steps, regulation, where="mid", color="tab:blue", linewidth=2.0, label="dispatch - baseline")
    axes[1].plot(steps, response_delta, color="tab:orange", linewidth=1.8, label="response - baseline")
    axes[1].fill_between(steps, regulation - tol, regulation + tol, color="tab:gray", alpha=0.18)
    axes[1].axhline(0, color="black", linewidth=1.0, alpha=0.4)
    axes[1].set_ylabel("Regulation [kW]")
    axes[1].legend(loc="upper right", fontsize=10)
    axes[1].grid(alpha=0.3)

    mismatch = target - response
    axes[2].bar(steps, mismatch, color=np.where(mismatch >= 0, "tab:red", "tab:blue"), alpha=0.65)
    axes[2].axhline(0, color="black", linewidth=1.0, alpha=0.5)
    axes[2].set_xlabel("5-min step")
    axes[2].set_ylabel("Target - response [kW]")
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(results_dir / f"{prefix}_dispatch_tracking.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    block_idx = np.asarray(payload["block_index"], dtype=int)
    baseline_plan = np.asarray(payload["baseline_plan_kw"], dtype=float)
    up_plan = np.asarray(payload["up_plan_kw"], dtype=float)
    down_plan = np.asarray(payload["down_plan_kw"], dtype=float)
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    axes[0].plot(block_idx, baseline_plan, color="tab:gray", linewidth=2.4)
    axes[0].axhline(0, color="black", linewidth=1.0, alpha=0.4)
    axes[0].set_ylabel("Baseline [kW]")
    axes[0].grid(alpha=0.3)
    axes[1].bar(block_idx - 0.18, up_plan, width=0.36, color="tab:blue", alpha=0.75, label="up")
    axes[1].bar(block_idx + 0.18, down_plan, width=0.36, color="tab:green", alpha=0.75, label="down")
    axes[1].set_ylabel("Awarded width [kW]")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[2].plot(
        block_idx, up_plan + down_plan, color="tab:purple", linewidth=2.4
    )
    axes[2].set_xlabel("30-min block")
    axes[2].set_ylabel("Up + down [kW]")
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(results_dir / "submitted_bid_profile.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    if block_rows:
        failed_steps = np.asarray([
            r["precision_failed_steps"] for r in block_rows
        ], dtype=float)
        up_stay = np.asarray([r["up_stay_rate"] for r in block_rows], dtype=float) * 100.0
        down_stay = np.asarray([r["down_stay_rate"] for r in block_rows], dtype=float) * 100.0
        up_pass = np.asarray([r["up_pass_II"] for r in block_rows], dtype=float) * 100.0
        down_pass = np.asarray([r["down_pass_II"] for r in block_rows], dtype=float) * 100.0
        fig, axes = plt.subplots(3, 1, figsize=(16, 13), sharex=True)
        axes[0].bar(block_idx, failed_steps, color="tab:purple", alpha=0.75)
        axes[0].set_ylabel("Failed 5-min steps")
        axes[0].grid(alpha=0.3)
        axes[1].plot(block_idx, up_stay, color="tab:blue", linewidth=2.2, label="up stay")
        axes[1].plot(block_idx, down_stay, color="tab:green", linewidth=2.2, label="down stay")
        axes[1].axhline(90.0, color="black", linestyle="--", linewidth=1.6, label="90%")
        axes[1].set_ylim(0, 105)
        axes[1].set_ylabel("Stay rate [%]")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        axes[2].step(block_idx, up_pass, where="mid", color="tab:blue", linewidth=2.0, label="up pass II")
        axes[2].step(block_idx, down_pass, where="mid", color="tab:green", linewidth=2.0, label="down pass II")
        axes[2].set_ylim(-5, 105)
        axes[2].set_xlabel("30-min block")
        axes[2].set_ylabel("Pass II")
        axes[2].legend()
        axes[2].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(results_dir / f"{prefix}_precision_by_block.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(18, 7))
    pos_bottom = np.zeros_like(steps, dtype=float)
    neg_bottom = np.zeros_like(steps, dtype=float)
    colors = plt.cm.tab20(np.linspace(0, 1, max(station_powers.shape[1], 1)))
    for st in range(station_powers.shape[1]):
        y = station_powers[:, st]
        pos = np.clip(y, 0, None)
        neg = np.clip(y, None, 0)
        ax.bar(steps, pos, bottom=pos_bottom, width=0.9, color=colors[st], alpha=0.75, label=f"station {st + 1}")
        ax.bar(steps, neg, bottom=neg_bottom, width=0.9, color=colors[st], alpha=0.75)
        pos_bottom += pos
        neg_bottom += neg
    ax.plot(
        steps,
        controller_pre_system_response,
        color="tab:gray",
        linewidth=1.2,
        label="controller pre-system",
    )
    ax.plot(steps, ev_response, color="black", linewidth=1.5, label="EV total")
    ax.plot(steps, response, color="tab:red", linewidth=2.0, label="assessed response")
    ax.set_xlabel("5-min step")
    ax.set_ylabel("Station power [kW]")
    ax.legend(ncol=3, fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(results_dir / f"{prefix}_station_stack.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _write_precision_distribution(results_dir: Path, rows: list[dict]) -> None:
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenario = np.asarray([r["scenario"] for r in rows], dtype=int)
    failed = np.asarray([
        r["failed_tracking_block_count"] for r in rows
    ], dtype=float)
    up_pass = np.asarray([r["up_pass_rate"] for r in rows], dtype=float) * 100.0
    down_pass = np.asarray([r["down_pass_rate"] for r in rows], dtype=float) * 100.0
    x = np.arange(len(rows))
    labels = [f"s{int(s)}" for s in scenario]
    fig, axes = plt.subplots(2, 1, figsize=(max(12, len(rows) * 0.55), 9), sharex=True)
    axes[0].bar(x, failed, color="tab:purple", alpha=0.75)
    axes[0].set_ylabel("Failed 30-min blocks")
    axes[0].grid(alpha=0.3)
    axes[1].plot(x, up_pass, color="tab:blue", marker="o", label="up pass")
    axes[1].plot(x, down_pass, color="tab:green", marker="o", label="down pass")
    axes[1].axhline(90.0, color="black", linestyle="--", linewidth=1.4)
    axes[1].set_ylabel("Pass II [%]")
    axes[1].set_xlabel("scenario/seed rollout")
    axes[1].set_ylim(0, 105)
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=60, ha="right")
    fig.tight_layout()
    fig.savefig(results_dir / "controller_precision_distribution.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _evaluate_controller_precision_impl(
    agent,
    fixed_bid: dict,
    n_seeds: int = 5,
    base_seed: int = 910_000,
    force_slack_kwh: float = 0.1,
    ignore_assessment_I: bool = False,
    evaluation_pipeline: str = "system",
    out_dir=None,
    visualize: bool = True,
) -> dict:
    """Roll out `agent` over all activation scenarios x `n_seeds` realizations."""

    from Config import EPISODE_STEPS
    from environment.EVEnv import EVEnv
    from environment.normalize import normalize_observation, use_instruction_scale
    from market.bid_env import N_BLOCKS, STEPS_PER_BLOCK
    from environment.physical_capability import (
        aggregate_physical_capability,
        step_dispatch_envelope,
    )
    from training.system_controller import apply_force_charging
    from tools.evaluator import set_env_seed
    from training.lower_bid_training import build_fixed_upper_bid_training_episode

    baseline = np.asarray(fixed_bid["baseline_plan"], dtype=float).reshape(-1)[:N_BLOCKS]
    up_plan = np.asarray(fixed_bid["up_plan"], dtype=float).reshape(-1)[:N_BLOCKS]
    down_plan = np.asarray(fixed_bid["down_plan"], dtype=float).reshape(-1)[:N_BLOCKS]
    if fixed_bid.get("bid_objective") != "total_ev_regulation_capacity_kw_block":
        raise ValueError("controller precision evaluation requires a capacity-only bid")
    arrival_probs = fixed_bid.get("arrival_probabilities_by_station")
    day_context = fixed_bid.get("day_context")
    service_date = fixed_bid.get("service_date")
    n_scen = max(int(fixed_bid.get("activation_scenarios", 0)), 1)
    offered_capacity_kw_block = float(np.sum(up_plan + down_plan))

    env = EVEnv()
    pipeline = evaluation_pipeline_settings(evaluation_pipeline)
    # The force layer is the pipeline's own setting below; the training-time
    # floor (EVMA_TRAIN_FORCE_CHARGING) would otherwise reach marl_raw too.
    env.apply_train_force_floor = False
    if pipeline["central_allocator"] is not None:
        env.use_central_ev_residual_allocator = bool(pipeline["central_allocator"])
    if pipeline["residual_bess"] is not None:
        env.use_residual_bess = bool(pipeline["residual_bess"])

    rows: list[dict] = []

    all_block_rows: list[dict] = []
    visual_payload = None
    candidate_up_precision_pass = np.ones(N_BLOCKS, dtype=bool)
    candidate_down_precision_pass = np.ones(N_BLOCKS, dtype=bool)
    candidate_up_assessment_i_pass = np.ones(N_BLOCKS, dtype=bool)
    candidate_down_assessment_i_pass = np.ones(N_BLOCKS, dtype=bool)
    candidate_soc_passed = True
    candidate_robust_up_failed_blocks = 0
    candidate_robust_down_failed_blocks = 0
    candidate_robust_joint_failed_blocks = 0
    for s in range(n_scen):
        target, tol, _arr, _info = build_fixed_upper_bid_training_episode(fixed_bid, s)
        tracking_enabled = np.asarray(
            _arr.get("tracking_enabled_series", np.ones(EPISODE_STEPS, dtype=bool)),
            dtype=bool,
        ).reshape(-1)[:EPISODE_STEPS]
        for k in range(int(n_seeds)):
            want_visual = bool(visualize and out_dir is not None and visual_payload is None)
            realized_seed = int(base_seed) + 1000 * s + k
            set_env_seed(realized_seed)
            use_instruction_scale(_arr.get("instruction_scale_kw", 1.0))
            env.reset(
                net_demand_series=target,
                tol_narrow_series=tol,
                tracking_enabled_series=tracking_enabled,
                market_context_series=_arr.get("market_context_series"),
                arrival_probabilities_by_station=arrival_probs,
                day_context=day_context,
                service_date=service_date,
                baseline_series=_arr.get("baseline_series"),
            )
            if bool(pipeline.get("use_actor", True)):
                agent.update_active_evs(env)
            disp = np.zeros(EPISODE_STEPS, dtype=float)
            resp = np.zeros(EPISODE_STEPS, dtype=float)
            ev_resp = np.zeros(EPISODE_STEPS, dtype=float)
            controller_pre_system_resp = np.zeros(EPISODE_STEPS, dtype=float)
            bess_power = np.zeros(EPISODE_STEPS, dtype=float)
            bess_soc = np.zeros(EPISODE_STEPS, dtype=float)
            sup_c = np.zeros(EPISODE_STEPS, dtype=float)
            sup_d = np.zeros(EPISODE_STEPS, dtype=float)
            station_powers = np.zeros((EPISODE_STEPS, env.num_stations), dtype=float)
            arrivals_by_station = np.zeros((EPISODE_STEPS, env.num_stations), dtype=float)
            forced_counts = np.zeros(EPISODE_STEPS, dtype=float)
            forced_kw_per_step = np.zeros(EPISODE_STEPS, dtype=float)
            step_env_min = np.zeros(EPISODE_STEPS, dtype=float)
            step_env_max = np.zeros(EPISODE_STEPS, dtype=float)
            step_env_obligated_min = np.zeros(EPISODE_STEPS, dtype=float)
            for t in range(EPISODE_STEPS):
                obs = env.begin_step()
                # Assessment I: the EV fleet's sustained power over this
                # 30-minute block, from the state at the block's first step.
                if t % STEPS_PER_BLOCK == 0:
                    block_end_step = min(t + STEPS_PER_BLOCK, EPISODE_STEPS)
                    block_charge_kw, block_discharge_kw = aggregate_physical_capability(
                        env, block_end_step
                    )
                sup_c[t] = block_charge_kw
                sup_d[t] = block_discharge_kw
                # The one-step reachable band, for the diagnostic that asks
                # whether a missed step was unreachable or merely missed.
                (
                    step_env_min[t],
                    step_env_max[t],
                    step_env_obligated_min[t],
                ) = step_dispatch_envelope(
                    env,
                    force_slack_kwh=(
                        force_slack_kwh if bool(pipeline["force_layer"]) else 0.0
                    ),
                )
                if bool(pipeline.get("use_actor", True)):
                    agent.update_active_evs(env)
                    obs = normalize_observation(obs)
                    actions = agent.act(obs, env=env, noise=False)
                else:
                    actions = env.soc.new_zeros(
                        (env.num_stations, env.max_ev_per_station)
                    )
                if bool(pipeline["force_layer"]):
                    actions, _forced, _ = apply_force_charging(
                        actions, env, slack_kwh=force_slack_kwh
                    )
                    forced_kw_per_step[t] = float(
                        getattr(apply_force_charging, "last_forced_kw", 0.0)
                    )
                else:
                    _forced = 0
                    forced_kw_per_step[t] = 0.0
                _o, _lr, _gr, _d, sinfo = env.apply_action(actions)
                disp[t] = float(env.current_net_demand)
                ev_resp[t] = float(sinfo.get("total_ev_transport", 0.0))
                # EVEnv calls its input action "raw actor".  In the C4
                # research pipeline that input already includes the mandatory
                # force-charging layer, so expose the unambiguous experiment
                # name here: controller output before central/BESS assistance.
                controller_pre_system_resp[t] = float(
                    sinfo.get("raw_actor_total_power_kw", ev_resp[t])
                )
                bess_power[t] = float(sinfo.get("bess_power_kw", 0.0))
                bess_soc[t] = float(sinfo.get("bess_soc_pct", 0.0))
                if pipeline["response_source"] == "pcc":
                    resp[t] = float(sinfo.get("pcc_power_kw", ev_resp[t]))
                else:
                    resp[t] = ev_resp[t]
                forced_counts[t] = float(_forced)
                if "station_powers" in sinfo:
                    station_powers[t, :] = np.asarray(sinfo["station_powers"], dtype=float).reshape(-1)[: env.num_stations]
                if "arrivals_by_station" in sinfo:
                    arrivals_by_station[t, :] = np.asarray(
                        sinfo["arrivals_by_station"], dtype=float
                    ).reshape(-1)[: env.num_stations]

            env_metrics = env.get_metrics()
            tol_arr = np.asarray(tol, dtype=float).reshape(-1)[:EPISODE_STEPS]
            assessment_tol_arr = tol_arr.copy()
            step_passed = np.abs(disp - resp) <= assessment_tol_arr + 1e-6
            # Was the instruction reachable at all?  Three indicators of the
            # same quantity: the sustained Assessment-I figure the bid is
            # certified against, the one-step physical band, and the one-step
            # band once the departure guarantee has claimed its share.
            _band = assessment_tol_arr + 1e-6
            _assessed = np.asarray(tracking_enabled, dtype=bool).reshape(-1)[:EPISODE_STEPS]
            _unreach_sustained = np.maximum(disp - sup_c, -sup_d - disp) > _band
            _unreach_step = np.maximum(disp - step_env_max, step_env_min - disp) > _band
            _unreach_obligated = (
                np.maximum(disp - step_env_max, step_env_obligated_min - disp) > _band
            )
            _missed = _assessed & ~step_passed
            # Block-level counterfactual: the market settles per 30-minute
            # block, so a share of missed steps does not say how many
            # blocks a perfect allocation would have saved. Count blocks
            # that are failed and hold at least one unreachable step
            # separately from blocks failed only at reachable steps.
            band_frac_for_rows = float(fixed_bid.get("assessment_band_fraction", 0.10) or 0.10)
            _blk = np.arange(EPISODE_STEPS) // STEPS_PER_BLOCK
            _blk_assessed = np.zeros(N_BLOCKS, dtype=bool)
            _blk_failed = np.zeros(N_BLOCKS, dtype=bool)
            _blk_has_unreach = np.zeros(N_BLOCKS, dtype=bool)
            for _b in range(N_BLOCKS):
                _sel = _blk == _b
                if not bool(_assessed[_sel].any()):
                    continue
                _blk_assessed[_b] = True
                _blk_failed[_b] = bool(_missed[_sel].any())
                _blk_has_unreach[_b] = bool((_assessed & _unreach_obligated)[_sel].any())
            _blk_saveable = _blk_failed & ~_blk_has_unreach
            central_step_passed = np.abs(disp - ev_resp) <= assessment_tol_arr + 1e-6
            controller_pre_system_step_passed = (
                np.abs(disp - controller_pre_system_resp)
                <= assessment_tol_arr + 1e-6
            )
            global_tracking_rate = masked_pass_rate(step_passed, tracking_enabled)
            controller_pre_system_tracking_rate = masked_pass_rate(
                controller_pre_system_step_passed, tracking_enabled
            )
            central_tracking_rate = masked_pass_rate(central_step_passed, tracking_enabled)
            # EVEnv scores departures on the actions it was handed. With the
            # central allocator those are not the executed ones on the last
            # step, so that pipeline counts the SoC each EV actually left with.
            if bool(pipeline["central_allocator"]):
                soc_miss_rate = float(env_metrics.get("central_soc_miss_rate", 0.0))
                departing_evs_soc_met = int(env_metrics.get("central_departing_evs_soc_met", 0))
            else:
                soc_miss_rate = float(env_metrics.get("soc_miss_rate", 0.0))
                departing_evs_soc_met = int(env_metrics.get("departing_evs_soc_met", 0))
            soc_hit_rate = 1.0 - soc_miss_rate / 100.0
            candidate_soc_passed = bool(
                candidate_soc_passed
                and int(env_metrics.get("departing_evs", 0)) == departing_evs_soc_met
            )
            forced_total = int(np.sum(forced_counts))
            forced_active_steps = int(np.count_nonzero(forced_counts > 0.0))
            # Shield footprint (checklist 0-1): the aggregate error band is
            # 0.1 * bid width, so a handful of EVs simultaneously forced to
            # max charge can consume it outright. Report the concurrency
            # distribution and the absolute forced power, not just totals.
            forced_concurrent_max = int(np.max(forced_counts)) if forced_counts.size else 0
            forced_concurrent_p95 = (
                float(np.percentile(forced_counts, 95.0)) if forced_counts.size else 0.0
            )
            forced_kw_max = float(np.max(forced_kw_per_step)) if forced_kw_per_step.size else 0.0
            forced_kw_mean_active = (
                float(np.mean(forced_kw_per_step[forced_counts > 0.0]))
                if forced_active_steps > 0
                else 0.0
            )
            # Forced power relative to the assessment band actually in force.
            _tol_pos = assessment_tol_arr
            _band_ok = _tol_pos > 1e-9
            forced_kw_to_band_max = (
                float(np.max(forced_kw_per_step[_band_ok] / _tol_pos[_band_ok]))
                if np.any(_band_ok)
                else 0.0
            )
            # Does the shield coincide with tracking failure?
            _assessed = np.asarray(tracking_enabled, dtype=bool)
            _forced_mask = (forced_counts > 0.0) & _assessed
            _clean_mask = (forced_counts <= 0.0) & _assessed
            forced_step_tracking_rate = (
                float(np.mean(step_passed[_forced_mask])) if np.any(_forced_mask) else float("nan")
            )
            unforced_step_tracking_rate = (
                float(np.mean(step_passed[_clean_mask])) if np.any(_clean_mask) else float("nan")
            )

            baseline_step_kw = _block_series(baseline, STEPS_PER_BLOCK, EPISODE_STEPS)
            participation_step_mask = participation_by_step(
                up_plan,
                down_plan,
                steps_per_block=STEPS_PER_BLOCK,
                steps=EPISODE_STEPS,
            )
            step_direction = direction_step_metrics(
                disp,
                resp,
                baseline_step_kw,
                assessment_tol_arr,
                participation_step_mask,
                steps_per_block=STEPS_PER_BLOCK,
            )

            up_pass_flags: list[float] = []
            down_pass_flags: list[float] = []
            idle_pass_flags: list[float] = []
            up_stay: list[float] = []
            down_stay: list[float] = []
            idle_stay: list[float] = []
            block_rows: list[dict] = []
            failed_up_blocks: list[int] = []
            failed_down_blocks: list[int] = []
            failed_idle_blocks: list[int] = []
            failed_tracking_blocks: list[int] = []
            for b in range(N_BLOCKS):
                s0 = b * STEPS_PER_BLOCK
                s1 = s0 + STEPS_PER_BLOCK
                base_b = float(baseline[b])
                up_unmet_i, up_supply_30 = _assessment_i_unmet_ratio(
                    float(up_plan[b]),
                    np.maximum(sup_d[s0:s1] + base_b, 0.0),
                )
                down_unmet_i, down_supply_30 = _assessment_i_unmet_ratio(
                    float(down_plan[b]),
                    np.maximum(sup_c[s0:s1] - base_b, 0.0),
                )
                participating = bool(up_plan[b] > 1e-6 or down_plan[b] > 1e-6)
                direct_stay_rate = (
                    float(np.mean(step_passed[s0:s1])) if participating else 1.0
                )
                direct_failed_steps = int(
                    np.count_nonzero(~step_passed[s0:s1])
                ) if participating else 0
                direct_pass_II = bool(direct_stay_rate >= 0.90)
                directional = directional_block_tracking(
                    disp[s0:s1],
                    resp[s0:s1],
                    baseline_kw=base_b,
                    tolerance_kw_per_step=assessment_tol_arr[s0:s1],
                )
                if up_plan[b] > 1e-6:
                    up_pass = bool(direct_pass_II)
                    candidate_up_precision_pass[b] &= up_pass
                    candidate_up_assessment_i_pass[b] &= bool(
                        ignore_assessment_I or up_unmet_i <= 1e-12
                    )
                    up_pass_flags.append(1.0 if up_pass else 0.0)
                    up_stay.append(float(direct_stay_rate))
                    if not up_pass:
                        failed_up_blocks.append(int(b))
                if down_plan[b] > 1e-6:
                    down_pass = bool(direct_pass_II)
                    candidate_down_precision_pass[b] &= down_pass
                    candidate_down_assessment_i_pass[b] &= bool(
                        ignore_assessment_I or down_unmet_i <= 1e-12
                    )
                    down_pass_flags.append(1.0 if down_pass else 0.0)
                    down_stay.append(float(direct_stay_rate))
                    if not down_pass:
                        failed_down_blocks.append(int(b))
                if participating and bool(directional["idle_assessed"]):
                    idle_pass = bool(directional["idle_pass_II"])
                    idle_pass_flags.append(1.0 if idle_pass else 0.0)
                    idle_stay.append(float(directional["idle_stay_rate"]))
                    if not idle_pass:
                        failed_idle_blocks.append(int(b))
                block_rows.append({
                    "block": int(b),
                    "baseline_kw": float(base_b),
                    "up_kw": float(up_plan[b]),
                    "down_kw": float(down_plan[b]),
                    "offered_capacity_kw": float(up_plan[b] + down_plan[b]),
                    "precision_stay_rate": float(direct_stay_rate),
                    "precision_failed_steps": int(direct_failed_steps),
                    "precision_pass_II": int(bool(direct_pass_II)),
                    "up_stay_rate": float(directional["up_stay_rate"]),
                    "down_stay_rate": float(directional["down_stay_rate"]),
                    "idle_stay_rate": float(directional["idle_stay_rate"]),
                    "up_pass_II": int(bool(direct_pass_II) if up_plan[b] > 1e-6 else True),
                    "down_pass_II": int(bool(direct_pass_II) if down_plan[b] > 1e-6 else True),
                    "idle_pass_II": int(bool(directional["idle_pass_II"])),
                    "active_up_pass_II": int(bool(directional["up_pass_II"])),
                    "active_down_pass_II": int(bool(directional["down_pass_II"])),
                    "up_active_steps": int(directional["up_active_steps"]),
                    "down_active_steps": int(directional["down_active_steps"]),
                    "idle_active_steps": int(directional["idle_active_steps"]),
                    "up_assessment_i_unmet_ratio": float(up_unmet_i),
                    "down_assessment_i_unmet_ratio": float(down_unmet_i),
                    "up_supply_capacity_30min_kw": float(up_supply_30),
                    "down_supply_capacity_30min_kw": float(down_supply_30),
                    "participating": int(participating),
                    "scenario": int(s),
                    "seed": int(k),
                    "service_date": str(service_date),
                    "band_up_kw": float(up_plan[b]) * band_frac_for_rows,
                    "band_down_kw": float(down_plan[b]) * band_frac_for_rows,
                    "band_idle_kw": float(up_plan[b] + down_plan[b]) * band_frac_for_rows,
                    "utilization": float(
                        np.mean(np.abs(disp[s0:s1] - float(base_b)))
                        / max(float(up_plan[b] + down_plan[b]), 1e-9)
                    ),
                    "has_unreachable_step": int(_blk_has_unreach[b]),
                })
                all_block_rows.append(block_rows[-1])
                if participating and not direct_pass_II:
                    failed_tracking_blocks.append(int(b))

            if want_visual:
                visual_payload = {
                    "scenario": int(s),
                    "seed": int(k),
                    "realized_seed": int(realized_seed),
                    "n_steps": int(EPISODE_STEPS),
                    "blocks": np.arange(EPISODE_STEPS, dtype=int) // int(STEPS_PER_BLOCK),
                    "block_index": np.arange(N_BLOCKS, dtype=int),
                    "baseline_plan_kw": baseline.copy(),
                    "up_plan_kw": up_plan.copy(),
                    "down_plan_kw": down_plan.copy(),
                    "baseline_step_kw": baseline_step_kw,
                    "dispatch_kw": disp.copy(),
                    "response_kw": resp.copy(),
                    "ev_response_kw": ev_resp.copy(),
                    "controller_pre_system_response_kw": (
                        controller_pre_system_resp.copy()
                    ),
                    "bess_power_kw": bess_power.copy(),
                    "bess_soc_pct": bess_soc.copy(),
                    "tol_kw": assessment_tol_arr.copy(),
                    "tracking_enabled": tracking_enabled.copy(),
                    "supply_charge_kw": sup_c.copy(),
                    "supply_discharge_kw": sup_d.copy(),
                    "station_powers_kw": station_powers.copy(),
                    "arrivals_by_station": arrivals_by_station.copy(),
                    "forced_count": forced_counts.copy(),
                    "block_rows": block_rows,
                }

            rollout_failed_up = set(failed_up_blocks)
            rollout_failed_down = set(failed_down_blocks)
            candidate_robust_up_failed_blocks = max(
                candidate_robust_up_failed_blocks,
                len(rollout_failed_up),
            )
            candidate_robust_down_failed_blocks = max(
                candidate_robust_down_failed_blocks,
                len(rollout_failed_down),
            )
            candidate_robust_joint_failed_blocks = max(
                candidate_robust_joint_failed_blocks,
                len(rollout_failed_up | rollout_failed_down),
            )

            rows.append({
                "scenario": s,
                "seed": k,
                "realized_seed": realized_seed,
                "service_date": service_date,
                "evaluation_pipeline": str(evaluation_pipeline),
                "offered_capacity_kw_block": offered_capacity_kw_block,
                "up_pass_rate": float(np.mean(up_pass_flags)) if up_pass_flags else 1.0,
                "down_pass_rate": float(np.mean(down_pass_flags)) if down_pass_flags else 1.0,
                "idle_pass_rate": float(np.mean(idle_pass_flags)) if idle_pass_flags else 1.0,
                "up_stay_mean": float(np.mean(up_stay)) if up_stay else 1.0,
                "down_stay_mean": float(np.mean(down_stay)) if down_stay else 1.0,
                "idle_stay_mean": float(np.mean(idle_stay)) if idle_stay else 1.0,
                "global_tracking_rate": global_tracking_rate,
                "market_tracking_rate": global_tracking_rate,
                "assessed_steps": int(_assessed.sum()),
                "missed_steps": int(_missed.sum()),
                "unreachable_sustained": int((_assessed & _unreach_sustained).sum()),
                "unreachable_step": int((_assessed & _unreach_step).sum()),
                "unreachable_obligated": int((_assessed & _unreach_obligated).sum()),
                "missed_unreachable_sustained": int((_missed & _unreach_sustained).sum()),
                "missed_unreachable_step": int((_missed & _unreach_step).sum()),
                "missed_unreachable_obligated": int((_missed & _unreach_obligated).sum()),
                "missed_though_reachable": int((_missed & ~_unreach_obligated).sum()),
                "blocks_assessed": int(_blk_assessed.sum()),
                "blocks_failed": int(_blk_failed.sum()),
                "blocks_failed_with_unreachable": int(_blk_has_unreach.sum()),
                "blocks_failed_all_reachable": int(_blk_saveable.sum()),
                "controller_pre_system_tracking_rate": (
                    controller_pre_system_tracking_rate
                ),
                "central_tracking_rate": central_tracking_rate,
                "controller_pre_system_mae_kw": float(np.mean(np.abs(
                    disp[tracking_enabled]
                    - controller_pre_system_resp[tracking_enabled]
                ))) if np.any(tracking_enabled) else 0.0,
                "pre_bess_mae_kw": float(np.mean(np.abs(disp[tracking_enabled] - ev_resp[tracking_enabled]))) if np.any(tracking_enabled) else 0.0,
                "post_bess_mae_kw": float(np.mean(np.abs(disp[tracking_enabled] - resp[tracking_enabled]))) if np.any(tracking_enabled) else 0.0,
                "bess_final_soc_pct": float(bess_soc[-1]) if bess_soc.size else 0.0,
                "bess_max_abs_power_kw": float(np.max(np.abs(bess_power))) if bess_power.size else 0.0,
                "bess_throughput_kwh": float(env_metrics.get("bess_throughput_kwh", 0.0)),
                "bess_power_limit_hits": int(env_metrics.get("bess_power_limit_hits", 0)),
                "bess_energy_limit_hits": int(env_metrics.get("bess_energy_limit_hits", 0)),
                "central_absolute_correction_kwh": float(
                    env_metrics.get("central_absolute_correction_kwh", 0.0)
                ),
                "central_corrected_ev_steps": int(
                    env_metrics.get("central_corrected_ev_steps", 0)
                ),
                "central_correction_active_steps": int(
                    env_metrics.get("central_correction_active_steps", 0)
                ),
                "soc_hit_rate": soc_hit_rate,
                "participating_blocks": int(np.count_nonzero(participation_by_block(up_plan, down_plan))),
                "failed_up_blocks": ";".join(map(str, failed_up_blocks)),
                "failed_down_blocks": ";".join(map(str, failed_down_blocks)),
                "failed_idle_blocks": ";".join(map(str, failed_idle_blocks)),
                "failed_tracking_blocks": ";".join(map(str, failed_tracking_blocks)),
                "failed_up_block_count": int(len(failed_up_blocks)),
                "failed_down_block_count": int(len(failed_down_blocks)),
                "failed_idle_block_count": int(len(failed_idle_blocks)),
                "failed_tracking_block_count": int(len(failed_tracking_blocks)),
                "up_step_pass_rate": float(step_direction["up_step_pass_rate"]),
                "down_step_pass_rate": float(step_direction["down_step_pass_rate"]),
                "idle_step_pass_rate": float(step_direction["idle_step_pass_rate"]),
                "up_active_step_count": int(step_direction["up_active_steps"]),
                "down_active_step_count": int(step_direction["down_active_steps"]),
                "idle_active_step_count": int(step_direction["idle_active_steps"]),
                "up_step_failed_blocks": ";".join(map(str, step_direction["up_step_failed_blocks"])),
                "down_step_failed_blocks": ";".join(map(str, step_direction["down_step_failed_blocks"])),
                "idle_step_failed_blocks": ";".join(map(str, step_direction["idle_step_failed_blocks"])),
                "up_step_failed_block_count": int(len(step_direction["up_step_failed_blocks"])),
                "down_step_failed_block_count": int(len(step_direction["down_step_failed_blocks"])),
                "idle_step_failed_block_count": int(len(step_direction["idle_step_failed_blocks"])),
                "departing_evs": int(env_metrics.get("departing_evs", 0)),
                "departing_evs_soc_met": departing_evs_soc_met,
                "forced_ev_step_overrides": forced_total,
                "forced_active_steps": forced_active_steps,
                "forced_concurrent_max": forced_concurrent_max,
                "forced_concurrent_p95": forced_concurrent_p95,
                "forced_kw_max": forced_kw_max,
                "forced_kw_mean_active": forced_kw_mean_active,
                "forced_kw_to_band_max": forced_kw_to_band_max,
                "forced_step_tracking_rate": forced_step_tracking_rate,
                "unforced_step_tracking_rate": unforced_step_tracking_rate,
            })

    summary = {
        "evaluation_pipeline": str(evaluation_pipeline),
        "assessment_i_ignored": bool(ignore_assessment_I),
        "offered_capacity_kw_block": offered_capacity_kw_block,
        "mean_offered_capacity_kw": float(offered_capacity_kw_block / N_BLOCKS),
        "up_pass_rate": float(np.mean([r["up_pass_rate"] for r in rows])) if rows else 1.0,
        "down_pass_rate": float(np.mean([r["down_pass_rate"] for r in rows])) if rows else 1.0,
        "idle_pass_rate": float(np.mean([r["idle_pass_rate"] for r in rows])) if rows else 1.0,
        "up_stay_mean": float(np.mean([r["up_stay_mean"] for r in rows])) if rows else 1.0,
        "down_stay_mean": float(np.mean([r["down_stay_mean"] for r in rows])) if rows else 1.0,
        "idle_stay_mean": float(np.mean([r["idle_stay_mean"] for r in rows])) if rows else 1.0,
        "global_tracking_rate": float(np.mean([r["global_tracking_rate"] for r in rows])) if rows else 1.0,
        "global_tracking_rate_min": float(np.min([r["global_tracking_rate"] for r in rows])) if rows else 1.0,
        "controller_pre_system_tracking_rate": float(np.mean([
            r["controller_pre_system_tracking_rate"] for r in rows
        ])) if rows else 1.0,
        "central_ev_tracking_rate": float(np.mean([r["central_tracking_rate"] for r in rows])) if rows else 1.0,
        "pre_bess_mae_kw": float(np.mean([r["pre_bess_mae_kw"] for r in rows])) if rows else 0.0,
        "post_bess_mae_kw": float(np.mean([r["post_bess_mae_kw"] for r in rows])) if rows else 0.0,
        "bess_max_abs_power_kw": float(max((r["bess_max_abs_power_kw"] for r in rows), default=0.0)),
        "bess_final_soc_pct": float(np.mean([
            r["bess_final_soc_pct"] for r in rows
        ])) if rows else 0.0,
        "bess_throughput_kwh_mean": float(np.mean([
            r["bess_throughput_kwh"] for r in rows
        ])) if rows else 0.0,
        "bess_power_limit_hits_mean": float(np.mean([
            r["bess_power_limit_hits"] for r in rows
        ])) if rows else 0.0,
        "bess_energy_limit_hits_mean": float(np.mean([
            r["bess_energy_limit_hits"] for r in rows
        ])) if rows else 0.0,
        "central_absolute_correction_kwh_mean": float(np.mean([
            r["central_absolute_correction_kwh"] for r in rows
        ])) if rows else 0.0,
        "central_corrected_ev_steps_mean": float(np.mean([
            r["central_corrected_ev_steps"] for r in rows
        ])) if rows else 0.0,
        "central_correction_active_steps_mean": float(np.mean([
            r["central_correction_active_steps"] for r in rows
        ])) if rows else 0.0,
        "participating_blocks": int(np.count_nonzero(participation_by_block(up_plan, down_plan))),
        "failed_up_blocks_total": int(sum(r["failed_up_block_count"] for r in rows)),
        "failed_down_blocks_total": int(sum(r["failed_down_block_count"] for r in rows)),
        "failed_idle_blocks_total": int(sum(r["failed_idle_block_count"] for r in rows)),
        "failed_tracking_blocks_total": int(sum(r["failed_tracking_block_count"] for r in rows)),
        "up_step_pass_rate": float(np.mean([r["up_step_pass_rate"] for r in rows])) if rows else 1.0,
        "down_step_pass_rate": float(np.mean([r["down_step_pass_rate"] for r in rows])) if rows else 1.0,
        "idle_step_pass_rate": float(np.mean([r["idle_step_pass_rate"] for r in rows])) if rows else 1.0,
        "up_step_pass_rate_min": float(np.min([r["up_step_pass_rate"] for r in rows])) if rows else 1.0,
        "down_step_pass_rate_min": float(np.min([r["down_step_pass_rate"] for r in rows])) if rows else 1.0,
        "idle_step_pass_rate_min": float(np.min([r["idle_step_pass_rate"] for r in rows])) if rows else 1.0,
        "up_step_failed_blocks_total": int(sum(r["up_step_failed_block_count"] for r in rows)),
        "down_step_failed_blocks_total": int(sum(r["down_step_failed_block_count"] for r in rows)),
        "idle_step_failed_blocks_total": int(sum(r["idle_step_failed_block_count"] for r in rows)),
        "soc_hit_rate": (
            float(sum(r["departing_evs_soc_met"] for r in rows))
            / float(sum(r["departing_evs"] for r in rows))
            if rows and sum(r["departing_evs"] for r in rows) > 0
            else 1.0
        ),
        "forced_ev_step_overrides": int(sum(r["forced_ev_step_overrides"] for r in rows)),
        "forced_active_steps": int(sum(r["forced_active_steps"] for r in rows)),
        "forced_concurrent_max": int(max((r["forced_concurrent_max"] for r in rows), default=0)),
        "forced_concurrent_p95_mean": float(
            np.mean([r["forced_concurrent_p95"] for r in rows]) if rows else 0.0
        ),
        "forced_kw_max": float(max((r["forced_kw_max"] for r in rows), default=0.0)),
        "forced_kw_mean_active": float(
            np.mean([r["forced_kw_mean_active"] for r in rows]) if rows else 0.0
        ),
        "forced_kw_to_band_max": float(max((r["forced_kw_to_band_max"] for r in rows), default=0.0)),
        "forced_step_tracking_rate": float(
            _nanmean_or_nan(r["forced_step_tracking_rate"] for r in rows)
            if rows else float("nan")
        ),
        "unforced_step_tracking_rate": float(
            _nanmean_or_nan(r["unforced_step_tracking_rate"] for r in rows)
            if rows else float("nan")
        ),
        "n_scenarios": n_scen,
        "n_seeds": int(n_seeds),
        # Robust candidate contract for the monthly DP.  A block passes only
        # when it passes every activation x realized-EV rollout evaluated here.
        # The baseline remains part of each candidate; it is never reconstructed
        # from the accepted directional widths after evaluation.
        "candidate_evaluation": {
            "aggregation": "all_evaluated_rollouts",
            "baseline_kw": baseline.astype(float).tolist(),
            "up_kw": up_plan.astype(float).tolist(),
            "down_kw": down_plan.astype(float).tolist(),
            "up_precision_pass": candidate_up_precision_pass.astype(bool).tolist(),
            "down_precision_pass": candidate_down_precision_pass.astype(bool).tolist(),
            "up_assessment_i_pass": candidate_up_assessment_i_pass.astype(bool).tolist(),
            "down_assessment_i_pass": candidate_down_assessment_i_pass.astype(bool).tolist(),
            "assessment_i_passed": bool(
                np.all(candidate_up_assessment_i_pass)
                and np.all(candidate_down_assessment_i_pass)
            ),
            "soc_passed": bool(candidate_soc_passed),
            "rollouts": int(n_scen * int(n_seeds)),
            "offered_capacity_kw_block": offered_capacity_kw_block,
            "mean_offered_capacity_kw": float(
                offered_capacity_kw_block / N_BLOCKS
            ),
            # Robust count uses the worst count in one rollout.  It does not
            # union block identities across mutually exclusive EV/command
            # scenarios, which would over-count the monthly budget.
            "robust_up_failed_blocks": int(
                candidate_robust_up_failed_blocks
            ),
            "robust_down_failed_blocks": int(
                candidate_robust_down_failed_blocks
            ),
            "robust_joint_failed_blocks": int(
                candidate_robust_joint_failed_blocks
            ),
        },
    }

    if out_dir is not None:
        results_dir = Path(out_dir) / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        with (results_dir / "controller_precision_by_scenario.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["scenario", "seed"])
            w.writeheader()
            for r in rows:
                w.writerow(r)
        with (results_dir / "controller_precision_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        _write_precision_distribution(results_dir, rows)
        if all_block_rows:
            with (results_dir / "controller_precision_by_block.csv").open(
                "w", newline="", encoding="utf-8"
            ) as f:
                w = csv.DictWriter(f, fieldnames=list(all_block_rows[0].keys()))
                w.writeheader()
                for r in all_block_rows:
                    w.writerow(r)
        if visual_payload is not None:
            _write_visual_artifacts(results_dir, visual_payload)

    print(
        "[controller precision] "
        f"capacity={summary['mean_offered_capacity_kw']:,.1f}kW  "
        f"passII up/down={summary['up_pass_rate']*100:.0f}/{summary['down_pass_rate']*100:.0f}%  "
        f"stay up/down={summary['up_stay_mean']*100:.0f}/{summary['down_stay_mean']*100:.0f}%  "
        f"global_market={summary['global_tracking_rate']*100:.1f}% "
        f"local={summary['soc_hit_rate']*100:.1f}% "
        f"force={summary['forced_ev_step_overrides']} "
        f"({summary['n_scenarios']} scen x {summary['n_seeds']} seeds)",
        flush=True,
    )
    return summary


def evaluate_controller_precision(
    agent,
    fixed_bid: dict,
    n_seeds: int = 5,
    base_seed: int = 910_000,
    force_slack_kwh: float = 0.1,
    ignore_assessment_I: bool = False,
    evaluation_pipeline: str = "system",
    out_dir=None,
    visualize: bool = True,
) -> dict:
    """Evaluate a controller without leaving the training agent in test mode."""

    previous_test_mode = bool(getattr(agent, "test_mode", False))
    exploration_state = {
        name: getattr(agent, name)
        for name in ("epsilon", "ou_noise_scale")
        if hasattr(agent, name)
    }
    agent.set_test_mode(True)
    try:
        return _evaluate_controller_precision_impl(
            agent,
            fixed_bid,
            n_seeds=n_seeds,
            base_seed=base_seed,
            force_slack_kwh=force_slack_kwh,
            ignore_assessment_I=ignore_assessment_I,
            evaluation_pipeline=evaluation_pipeline,
            out_dir=out_dir,
            visualize=visualize,
        )
    finally:
        agent.set_test_mode(previous_test_mode)
        for name, value in exploration_state.items():
            setattr(agent, name, value)
