"""Evaluate the distributed MARL controller or the central rule baseline.

The submitted bids come from a persistent bid bank.  Command traces are drawn
from the disjoint holdout partition, minus every command the model's pretrain
drew (the lower controller trains on all partitions but the bid design one),
and EV realizations use independent seeds. The script must configure the
observation layout before importing EVEnv.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--episode", type=int, default=None)
    parser.add_argument(
        "--bid-bank-dir",
        default=str(
            PROJECT_ROOT
            / "execute_results"
            / "bid_banks"
            / "validation_5_minmedmax_3of128ev_128cmd_all_commands"
        ),
    )
    parser.add_argument("--command-scenarios", type=int, default=24)
    parser.add_argument("--ev-seeds", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=1_422_090)
    parser.add_argument("--force-slack-kwh", type=float, default=0.1)
    parser.add_argument(
        "--pipeline",
        choices=("marl_raw", "marl_force", "marl_force_bess", "rule_based_central"),
        default="marl_force_bess",
        help="Controller stack to evaluate. The default is the proposed system.",
    )
    parser.add_argument("--max-days", type=int, default=0)
    parser.add_argument(
        "--include-training-commands",
        action="store_true",
        help="Keep holdout commands the model's pretrain drew. By default they are excluded.",
    )
    parser.add_argument("--output-dir", required=True)
    return parser


def _commands_seen_in_training(model_dir: Path) -> set[str]:
    """Rebuild the rollout commands the model's bank pretrain drew."""

    from EnvConfig import LOWER_TRAIN_UPPER_BID_ACTIVATION_SOURCE_DIR
    from training.bid_bank import BidBank
    from training.lower_bid_training import lower_commands_seen_in_pretrain

    manifest_path = model_dir / "resume" / "latest.json"
    if not manifest_path.exists():
        raise SystemExit(
            f"{manifest_path} is missing, so the commands this model trained on cannot be "
            "rebuilt; pass --include-training-commands to evaluate without excluding them"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    context = manifest["context"]
    if context.get("kind") != "lower_marl_bid_bank_pretrain":
        raise SystemExit(f"not a bank pretrain run: {context.get('kind')!r}")
    runtime = context["runtime"]
    library = Path(runtime["EnvConfig"]["LOWER_TRAIN_UPPER_BID_ACTIVATION_SOURCE_DIR"]).resolve()
    if library != Path(LOWER_TRAIN_UPPER_BID_ACTIVATION_SOURCE_DIR).resolve():
        raise SystemExit(
            f"the model trained on commands from {library}, but this evaluation draws from "
            f"{LOWER_TRAIN_UPPER_BID_ACTIVATION_SOURCE_DIR}; set EVMA_ACTIVATION_SIGNAL_SET to match"
        )
    return lower_commands_seen_in_pretrain(
        list(BidBank(context["train_bid_bank"]["path"]).entries),
        list(BidBank(context["test_bid_bank"]["path"]).entries),
        environment_episodes=int(manifest["completed_environment_episodes"]),
        interim_test_episodes=int(runtime["Config"]["INTERIM_TEST_EPISODES"]),
        library_dir=library,
    )


def _configure_environment() -> None:
    os.environ["EVMA_LOWER_BID_CONTEXT_OBS"] = "1"
    os.environ["EVMA_USE_CENTRAL_EV_RESIDUAL_ALLOCATOR"] = "0"
    os.environ["EVMA_USE_RESIDUAL_BESS"] = "1"


def _weighted_soc(rows) -> float:
    departing = float(rows["departing_evs"].sum())
    if departing <= 0.0:
        return 1.0
    return float(rows["departing_evs_soc_met"].sum()) / departing


def _scale_submitted_bid(fixed_bid: dict) -> dict:
    """約定量を EVMA_EVAL_BID_SCALE 倍に縮める。

    目標も帯も up_plan/down_plan から再計算されるので、この2本を縮めれば
    両方が整合して縮む。縮めた結果が最低入札量を割るブロックは、規定上
    参加できないのでゼロにする。解き直しではないが、能力 X を出せるなら
    sX も出せるので実行可能性は保たれる。
    """
    import numpy as np

    raw = os.environ.get("EVMA_EVAL_BID_SCALE", "1.0").strip()
    try:
        scale = float(raw)
    except ValueError:
        raise SystemExit(f"EVMA_EVAL_BID_SCALE が数値でない: {raw!r}")
    if scale == 1.0:
        return fixed_bid
    if not (0.0 < scale <= 1.0):
        raise SystemExit(f"EVMA_EVAL_BID_SCALE は (0, 1] の範囲: {scale}")

    from EnvConfig import LOWER_TRAIN_RESEARCH_MINIMUM_BID_QUANTITY_KW as MIN_BID

    q = float(MIN_BID)
    out = dict(fixed_bid)
    kept = 0
    dropped = 0
    for key in ("up_plan", "down_plan"):
        plan = np.asarray(out[key], dtype=float) * scale
        below = (plan > 1e-8) & (plan < q)
        dropped += int(below.sum())
        plan[below] = 0.0
        kept += int((plan > 1e-8).sum())
        out[key] = plan
    print(
        f"[eval] 入札を {scale:.2f} 倍に縮めた: 参加方向ブロック {kept}, "
        f"最低入札量 {q:.0f} kW を割って落ちたもの {dropped}",
        flush=True,
    )
    return out


def _drop_bid_blocks(fixed_bid: dict) -> dict:
    """EVMA_EVAL_DROP_BLOCKS_JSON に挙げたブロックを入札から外す。

    JSON は {"YYYY-MM-DD": [block, ...]}。上げも下げもゼロのブロックは
    その30分コマ全体が評価の対象外になるので、落としたブロックの EV は
    追従を気にせず充電できる。残るブロックが楽になるかを測るための穴。
    """
    import json

    import numpy as np

    path = os.environ.get("EVMA_EVAL_DROP_BLOCKS_JSON", "").strip()
    if not path:
        return fixed_bid
    with open(path, encoding="utf-8") as fh:
        table = json.load(fh)
    day = str(fixed_bid.get("service_date"))
    blocks = [int(b) for b in table.get(day, [])]
    if not blocks:
        print(f"[eval] {day}: 外すブロックなし", flush=True)
        return fixed_bid
    out = dict(fixed_bid)
    for key in ("up_plan", "down_plan"):
        plan = np.asarray(out[key], dtype=float).copy()
        plan[blocks] = 0.0
        out[key] = plan
    print(f"[eval] {day}: {len(blocks)} ブロックを入札から外した {blocks}", flush=True)
    return out


def _override_band_fraction(fixed_bid: dict) -> dict:
    """許容帯を EVMA_EVAL_BAND_FRACTION に差し替える。

    約定量はそのままで帯幅だけを変える。帯は目標と一緒に
    _bid_to_target_tol_from_activation で再計算されるので、この値を
    置き換えれば評価側の合否判定にもそのまま効く。規定は 10%。
    """
    raw = os.environ.get("EVMA_EVAL_BAND_FRACTION", "").strip()
    if not raw:
        return fixed_bid
    frac = float(raw)
    if not (0.0 < frac <= 1.0):
        raise SystemExit(f"EVMA_EVAL_BAND_FRACTION は (0, 1]: {frac}")
    out = dict(fixed_bid)
    before = float(out.get("assessment_band_fraction", 0.1))
    out["assessment_band_fraction"] = frac
    print(f"[eval] 帯幅 {before:.3f} -> {frac:.3f}", flush=True)
    return out


def _apply_min_award(fixed_bid: dict) -> dict:
    """EVMA_EVAL_MIN_AWARD_KW 未満の方向約定をゼロにする。

    帯下限（LOWER_CONTROLLER_MIN_TRACKING_BAND_KW）を上げたときに落ちる
    ブロックを、入札を解き直さずに近似する。解き直しの効果は含まれない。
    """
    import numpy as np

    raw = os.environ.get("EVMA_EVAL_MIN_AWARD_KW", "").strip()
    if not raw:
        return fixed_bid
    floor = float(raw)
    out = dict(fixed_bid)
    dropped = kept = 0
    for key in ("up_plan", "down_plan"):
        plan = np.asarray(out[key], dtype=float).copy()
        low = (plan > 1e-8) & (plan < floor)
        dropped += int(low.sum())
        plan[low] = 0.0
        kept += int((plan > 1e-8).sum())
        out[key] = plan
    print(f"[eval] {fixed_bid.get('service_date')}: 約定 {floor:.0f} kW 未満を落とした "
          f"落とし {dropped} / 残り {kept}", flush=True)
    return out


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command_scenarios <= 0 or args.ev_seeds <= 0:
        raise ValueError("command scenarios and EV seeds must be positive")
    _configure_environment()

    import numpy as np
    import pandas as pd
    import torch

    from environment.EVEnv import EVEnv
    from environment.normalize import (
        load_observation_normalization_for_archive,
        normalize_observation,
        use_instruction_scale,
    )
    from tools.evaluator import set_env_seed
    from training.system_controller import (
        build_agent,
        choose_best_episode,
        find_model_path_and_episode,
        validate_actor_checkpoint_compatibility,
    )
    from training.bid_bank import BidBank
    from training.evaluate_controller_precision import evaluate_controller_precision
    from training.lower_bid_training import (
        _activation_scenarios_for_day,
        build_fixed_upper_bid_training_episode,
    )

    model_dir = Path(args.model_dir).expanduser().resolve()
    bid_bank_dir = Path(args.bid_bank_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    profile = load_observation_normalization_for_archive(model_dir)
    bank = BidBank(bid_bank_dir)
    entries = list(bank.entries)
    if args.max_days > 0:
        entries = entries[: int(args.max_days)]
    if not entries:
        raise ValueError("no bid-bank entries selected")
    seen_in_training: set[str] = (
        set() if args.include_training_commands else _commands_seen_in_training(model_dir)
    )
    print(f"[eval] excluding {len(seen_in_training)} commands the pretrain drew", flush=True)

    first_bid = dict(bank.load_entry(entries[0]))
    first_payloads, _ = _activation_scenarios_for_day(
        first_bid.get("service_date"),
        int(first_bid.get("forecast_seed", 0)) + 512_209,
        n_scenarios=1,
        scenario_partition="holdout",
        exclude_sources=seen_in_training,
    )
    first_bid["activation_scenario_payload"] = list(first_payloads)
    first_bid["activation_scenarios"] = len(first_payloads)
    target, tol, arrival, _ = build_fixed_upper_bid_training_episode(first_bid, 0)
    set_env_seed(int(args.base_seed))
    use_instruction_scale(arrival.get("instruction_scale_kw", 1.0))
    bootstrap_env = EVEnv()
    bootstrap_env.reset(
        net_demand_series=target,
        tol_narrow_series=tol,
        tracking_enabled_series=arrival.get("tracking_enabled_series"),
        market_context_series=arrival.get("market_context_series"),
        arrival_probabilities_by_station=arrival.get("arrival_probabilities_by_station"),
        day_context=arrival.get("day_context"),
        service_date=arrival.get("service_date"),
        baseline_series=arrival.get("baseline_series"),
    )
    # Exercise normalization once before loading so a layout/profile mismatch
    # fails before the expensive rollouts start.
    normalized = normalize_observation(bootstrap_env.begin_step())
    agent = build_agent(bootstrap_env)
    if int(normalized.shape[1]) != int(agent.s_dim):
        raise ValueError(
            f"observation/agent mismatch: {normalized.shape[1]} != {agent.s_dim}"
        )

    selected_episode = (
        int(args.episode)
        if args.episode is not None
        else int(choose_best_episode(str(model_dir)))
    )
    checkpoint_dir, selected_episode = find_model_path_and_episode(
        str(model_dir), selected_episode
    )
    validate_actor_checkpoint_compatibility(
        agent, checkpoint_dir, selected_episode, expected_stations=bootstrap_env.num_stations
    )
    map_location = None if torch.cuda.is_available() else "cpu"
    agent.load_actors(checkpoint_dir, selected_episode, map_location=map_location)

    day_summaries: list[dict] = []
    rollout_frames = []
    for day_index, entry in enumerate(entries):
        fixed_bid = dict(bank.load_entry(entry))
        fixed_bid = _scale_submitted_bid(fixed_bid)
        fixed_bid = _drop_bid_blocks(fixed_bid)
        fixed_bid = _override_band_fraction(fixed_bid)
        fixed_bid = _apply_min_award(fixed_bid)
        command_seed = int(fixed_bid.get("forecast_seed", 0)) + 512_209
        payloads, activation_mode = _activation_scenarios_for_day(
            fixed_bid.get("service_date"),
            command_seed,
            n_scenarios=int(args.command_scenarios),
            scenario_partition="holdout",
            exclude_sources=seen_in_training,
        )
        fixed_bid["activation_scenario_payload"] = list(payloads)
        fixed_bid["activation_scenarios"] = len(payloads)
        fixed_bid["activation_mode"] = activation_mode
        day_dir = output_dir / f"day_{day_index:03d}_{entry['service_date']}"
        summary = evaluate_controller_precision(
            agent,
            fixed_bid,
            n_seeds=int(args.ev_seeds),
            base_seed=int(args.base_seed) + day_index * 100_000,
            force_slack_kwh=float(args.force_slack_kwh),
            evaluation_pipeline=str(args.pipeline),
            out_dir=day_dir,
            visualize=False,
        )
        summary = {"service_date": entry["service_date"], **summary}
        day_summaries.append(summary)
        frame = pd.read_csv(day_dir / "results" / "controller_precision_by_scenario.csv")
        frame.insert(0, "bid_day_index", day_index)
        rollout_frames.append(frame)

    rows = pd.concat(rollout_frames, ignore_index=True)
    mean_columns = (
        "up_pass_rate",
        "down_pass_rate",
        "idle_pass_rate",
        "global_tracking_rate",
        "controller_pre_system_tracking_rate",
        "central_tracking_rate",
        "controller_pre_system_mae_kw",
        "pre_bess_mae_kw",
        "post_bess_mae_kw",
        "bess_final_soc_pct",
        "bess_throughput_kwh",
        "bess_power_limit_hits",
        "bess_energy_limit_hits",
        "central_absolute_correction_kwh",
        "central_corrected_ev_steps",
        "central_correction_active_steps",
        "forced_active_steps",
        "forced_concurrent_p95",
        "forced_kw_mean_active",
    )
    aggregate = {
        "model_dir": str(model_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "model_episode": int(selected_episode),
        "evaluation_pipeline": "system",
        "command_partition": "holdout",
        "excluded_training_commands": len(seen_in_training),
        "bid_days": len(entries),
        "command_scenarios_per_day": int(args.command_scenarios),
        "ev_seeds_per_command": int(args.ev_seeds),
        "rollouts": int(len(rows)),
        "mean_offered_capacity_kw": float(np.mean([
            day["mean_offered_capacity_kw"] for day in day_summaries
        ])),
        "soc_hit_rate": _weighted_soc(rows),
        "forced_ev_step_overrides_total": int(rows["forced_ev_step_overrides"].sum()),
        "forced_ev_step_overrides_mean": float(rows["forced_ev_step_overrides"].mean()),
        "forced_concurrent_max": int(rows["forced_concurrent_max"].max()),
        "forced_kw_max": float(rows["forced_kw_max"].max()),
        "bess_max_abs_power_kw": float(rows["bess_max_abs_power_kw"].max()),
        **{name: float(rows[name].mean()) for name in mean_columns},
    }
    rows.to_csv(output_dir / "all_rollouts.csv", index=False)
    pd.DataFrame(day_summaries).to_csv(output_dir / "summary_by_bid_day.csv", index=False)
    pd.DataFrame([aggregate]).to_csv(output_dir / "summary_overall.csv", index=False)
    (output_dir / "summary_overall.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "model_dir": str(model_dir),
                "bid_bank_dir": str(bid_bank_dir),
                "output_dir": str(output_dir),
                "observation_normalization": profile,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(json.dumps(aggregate, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
