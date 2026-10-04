"""固定入札に設計外の指令と独立EV実現を与え、1組ずつ物理的成立を判定する。"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from market.physical_lp_bidding.colgen_feasibility import certify
from market.physical_lp_bidding.data_classes import BiddingLPConfig
from market.physical_lp_bidding.joint_validation import fixed_bid_tracking_bands
from training.bid_bank import BidBank


BANKS = {
    "AEMO": "train_25_minmedmax_3of128ev_128cmd_all_commands_aemo_plan_deviation",
    "ERCOT": "train_25_minmedmax_3of128ev_128cmd_all_commands_ercot_plan_deviation",
    "GB": "train_25_minmedmax_3of128ev_128cmd_all_commands_elexon_plan_deviation",
}


def command_identity(row: dict) -> tuple[str, str]:
    return str(row.get("source_date", "")), str(row.get("source_bmu", ""))


def evaluate_day(
    market: str,
    bank_dir: str,
    entry: dict,
    time_limit_s: float,
    ev_seed_count: int,
) -> dict:
    from training.bid_bank import load_fixed_bid
    from training.lower_bid_training import _sample_ev_scenario_bank

    bid_path = Path(bank_dir) / entry["bid_path"]
    fixed_bid = load_fixed_bid(bid_path)
    design = list(fixed_bid["design_activation_scenario_payload"])
    feedback = list(fixed_bid["activation_scenario_payload"])
    design_ids = {command_identity(row) for row in design}
    feedback_ids = [command_identity(row) for row in feedback]
    if not design_ids.isdisjoint(feedback_ids):
        raise ValueError(f"設計指令と別指令が重複: {market} {entry['service_date']}")
    if len(set(feedback_ids)) != len(feedback_ids):
        raise ValueError(f"別指令に重複: {market} {entry['service_date']}")
    # 入札時の最小・中央値・最大のEV実現は、当日の実現値ではない。
    # 入札時に引いた候補128本と重ならないseedから独立に生成する。
    forecast_seed = int(fixed_bid["forecast_seed"])
    ev_seed_values = [forecast_seed + 10_000_000 + 10_007 * i for i in range(ev_seed_count)]
    design_ev_seeds = {
        forecast_seed + 10_007 * i
        for i in range(int(fixed_bid["ev_scenario_candidate_count"]))
    }
    if design_ev_seeds.intersection(ev_seed_values):
        raise ValueError(f"設計用EV seedと評価用EV seedが重複: {market} {entry['service_date']}")
    ev_bank = _sample_ev_scenario_bank(
        count=ev_seed_count,
        seed=forecast_seed,
        seed_offset=10_000_000,
        arrival_probs=fixed_bid["arrival_probabilities_by_station"],
        day_context=fixed_bid["day_context"],
        label="independent evaluation",
        service_date=fixed_bid.get("service_date"),
    )

    cfg = BiddingLPConfig(
        assessment_band_fraction=float(fixed_bid["assessment_band_fraction"]),
        apply_transition_band=bool(fixed_bid["apply_transition_band"]),
    )
    baseline = np.asarray(fixed_bid["baseline_plan"], dtype=float)
    up = np.asarray(fixed_bid["up_plan"], dtype=float)
    down = np.asarray(fixed_bid["down_plan"], dtype=float)
    rows = []
    started = time.perf_counter()
    for command, source in zip(feedback, feedback_ids):
        _, _, lower, upper = fixed_bid_tracking_bands(
            cfg,
            baseline,
            up,
            down,
            np.asarray(command["up_proxy"], dtype=float),
            np.asarray(command["down_proxy"], dtype=float),
            apply_transition_band=cfg.apply_transition_band,
        )
        trials = []
        for ev_seed, evs in zip(ev_seed_values, ev_bank):
            feasible, _, info = certify(
                evs,
                lower,
                upper,
                steps=cfg.steps,
                dt=cfg.dt_hours,
                eta_ch=cfg.eta_ch,
                return_dispatch=False,
                time_limit_s=time_limit_s,
            )
            trials.append({
                "ev_seed": ev_seed,
                "feasible": feasible,
                "reason": info.get("reason") if feasible is None else None,
            })
        rows.append({
            "source_date": source[0],
            "source_bmu": source[1],
            "trials": trials,
        })
    return {
        "result_version": 3,
        "market": market,
        "service_date": entry["service_date"],
        "bid_path": str(bid_path),
        "bid_sha256": hashlib.sha256(bid_path.read_bytes()).hexdigest(),
        "design_commands": len(design),
        "unseen_commands": len(feedback),
        "ev_sampling": "independent_seed",
        "ev_seeds": ev_seed_values,
        "ev_counts": [len(evs) for evs in ev_bank],
        "solver_time_limit_s": time_limit_s,
        "elapsed_s": time.perf_counter() - started,
        "rows": rows,
    }


def counts(day: dict) -> dict:
    trials = [trial for row in day["rows"] for trial in row["trials"]]
    return {
        "commands": len(day["rows"]),
        "trials": len(trials),
        "feasible": sum(trial["feasible"] is True for trial in trials),
        "infeasible": sum(trial["feasible"] is False for trial in trials),
        "unknown": sum(trial["feasible"] is None for trial in trials),
    }


def market_summary(days: list[dict]) -> dict:
    parts = [counts(day) for day in days]
    total = {key: sum(part[key] for part in parts) for key in parts[0]}
    total["bid_days"] = len(days)
    total["ev_seeds_per_command"] = len(days[0]["ev_seeds"])
    total["success_rate"] = (
        total["feasible"] / total["trials"] if total["unknown"] == 0 else None
    )
    total["success_rate_lower_bound"] = total["feasible"] / total["trials"]
    total["success_rate_upper_bound"] = (
        total["feasible"] + total["unknown"]
    ) / total["trials"]
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--time-limit-s", type=float, default=20.0)
    parser.add_argument("--ev-seeds", type=int, default=3)
    parser.add_argument("--max-days", type=int, default=0)
    parser.add_argument("--market", choices=("all", *BANKS), default="all")
    parser.add_argument("--bank-dir", type=Path)
    args = parser.parse_args()
    if args.workers < 1 or args.time_limit_s <= 0 or args.ev_seeds < 1:
        raise ValueError("workers、time-limit-s、ev-seedsは正の値にする")
    if args.bank_dir is not None and args.market == "all":
        parser.error("--bank-dir を使うときは --market を指定する")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    entries_by_market = {}
    selected_markets = BANKS if args.market == "all" else {args.market: BANKS[args.market]}
    for market, name in selected_markets.items():
        bank_dir = (
            args.bank_dir.resolve()
            if args.bank_dir is not None
            else ROOT / "execute_results" / "bid_banks" / name
        )
        bank = BidBank(bank_dir)
        entries = list(bank.entries)
        if args.max_days:
            entries = entries[: args.max_days]
        entries_by_market[market] = (bank_dir, entries)
    tasks = []
    day_results = []
    max_entries = max(len(entries) for _, entries in entries_by_market.values())
    for index in range(max_entries):
        for market, (bank_dir, entries) in entries_by_market.items():
            if index >= len(entries):
                continue
            entry = entries[index]
            filename = output / f"{market}_{entry['service_date']}.json"
            if filename.exists():
                day = json.loads(filename.read_text(encoding="utf-8"))
                if (
                    day.get("result_version") != 3
                    or day.get("ev_sampling") != "independent_seed"
                    or len(day.get("ev_seeds", [])) != args.ev_seeds
                ):
                    raise ValueError(f"保存済みの結果は独立EV seedの評価ではない: {filename}")
                bid_path = bank_dir / entry["bid_path"]
                if day["bid_sha256"] != hashlib.sha256(bid_path.read_bytes()).hexdigest():
                    raise ValueError(f"保存済みの結果と入札が違う: {filename}")
                if (
                    args.time_limit_s > float(day.get("solver_time_limit_s", 20.0))
                    and counts(day)["unknown"] > 0
                ):
                    raise ValueError(
                        f"{filename} に時間切れがある。時間上限を変えた再判定は"
                        "未実装なので、保存値をそのまま使えない"
                    )
                day_results.append(day)
            else:
                tasks.append((market, str(bank_dir), entry, args.time_limit_s, args.ev_seeds))
    print(
        f"入札日={len(tasks) + len(day_results)}、保存済み={len(day_results)}、"
        f"残り={len(tasks)}、同時実行={args.workers}、指令×独立EV seed={args.ev_seeds}",
        flush=True,
    )

    if tasks:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(evaluate_day, *task): task for task in tasks}
            for future in as_completed(futures):
                day = future.result()
                filename = output / f"{day['market']}_{day['service_date']}.json"
                filename.write_text(json.dumps(day, ensure_ascii=False, indent=2), encoding="utf-8")
                day_results.append(day)
                c = counts(day)
                print(
                    f"{day['market']} {day['service_date']} "
                    f"成立={c['feasible']}/{c['trials']} "
                    f"不明={c['unknown']} "
                    f"{day['elapsed_s']:.1f}s",
                    flush=True,
                )
    summary = {}
    for market in selected_markets:
        market_days = [day for day in day_results if day["market"] == market]
        summary[market] = market_summary(market_days)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
