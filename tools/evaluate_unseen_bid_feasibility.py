"""固定入札に設計外の指令と独立EV実現を与え、1組ずつ物理的成立を判定する。

    EVMA_NUM_STATIONS=<局数> python tools/evaluate_unseen_bid_feasibility.py \
        --output-dir <dir> --bank AEMO=<bank_dir> [--bank ERCOT=<bank_dir> ...]

入札日ごとに、入札の設計に使った指令とは別の指令（bank の activation_scenario_payload）
を使い、EV は入札時の候補と重ならない seed から独立に作る。指令×EV seed の1組ずつ、
固定した入札の許容幅の中で全 EV の出発時 SoC を満たす配分があるかを列生成で判定する。
すべてを知る制御での成立なので、下位制御器の追従率の上限側にあたる。
局数は環境変数 EVMA_NUM_STATIONS で指定し、bank の到着確率の局数と合わなければ止まる。

--unseen-commands N を付けると、bank の別指令の代わりに、入札時と同じ区分（feedback）と
seed から先頭N本を引き直して使う。区分の引き方は並べ替えの先頭を取るので、設計の
指令数が違う入札どうしでも同じN本で比べられる。保存されている別指令とは先頭が一致
することを確かめる。

--commands-from 市場名=ライブラリ:区分 を付けた市場は、bank の別指令の代わりに、その
ライブラリのその区分の指令を全部使う。PJM は入札と学習を疑似指令で行うので、実指令
（疑似指令の元にならなかった区分）で測るときに使う。
"""

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

RESULT_VERSION = 4


def command_identity(row: dict) -> tuple[str, str]:
    return str(row.get("source_date", "")), str(row.get("source_bmu", ""))


def commands_from_library(service_date, seed: int, library: str, partition: str) -> list[dict]:
    """Every command of one declared partition of a library, as solver payloads."""

    from market.activation_scenarios import (
        build_activation_scenario_set,
        load_proxy_shape_library,
        scenarios_to_solver_payload,
    )

    declared = load_proxy_shape_library(library)["declared_partition"]
    count = int((declared == partition).sum())
    if count == 0:
        raise ValueError(f"{library} に区分 {partition} の指令がない")
    scenarios = build_activation_scenario_set(
        service_date=service_date,
        n_scenarios=count,
        seed=seed,
        proxy_shape_dir=library,
        scenario_partition=partition,
        require_unique=True,
    )
    return scenarios_to_solver_payload(scenarios)


def evaluate_day(
    market: str,
    bank_dir: str,
    entry: dict,
    time_limit_s: float,
    ev_seed_count: int,
    commands_from: tuple[str, str] | None = None,
    unseen_commands: int = 0,
) -> dict:
    import torch

    from EnvConfig import NUM_STATIONS
    from training.bid_bank import load_fixed_bid
    from training.lower_bid_training import (
        FEEDBACK_COMMAND_SEED_OFFSET,
        _activation_scenarios_for_day,
        _sample_ev_scenario_bank,
    )

    bid_path = Path(bank_dir) / entry["bid_path"]
    fixed_bid = load_fixed_bid(bid_path)
    arrival_probs = np.asarray(fixed_bid["arrival_probabilities_by_station"])
    if arrival_probs.shape[0] != int(NUM_STATIONS):
        raise ValueError(
            f"bank の局数 {arrival_probs.shape[0]} と EVMA_NUM_STATIONS={NUM_STATIONS} が違う: {bid_path}"
        )
    design = list(fixed_bid["design_activation_scenario_payload"])
    if commands_from is None and unseen_commands:
        stored = list(fixed_bid["activation_scenario_payload"])
        feedback, _ = _activation_scenarios_for_day(
            fixed_bid.get("service_date"),
            int(fixed_bid["forecast_seed"]) + FEEDBACK_COMMAND_SEED_OFFSET,
            n_scenarios=int(unseen_commands),
            scenario_partition="feedback",
        )
        shared = min(len(stored), len(feedback))
        if [command_identity(row) for row in stored[:shared]] != [
            command_identity(row) for row in feedback[:shared]
        ]:
            raise ValueError(
                f"引き直した別指令が入札の保存分と一致しない: {market} {entry['service_date']}"
            )
        command_source = unseen_command_source(int(unseen_commands))
    elif commands_from is None:
        feedback = list(fixed_bid["activation_scenario_payload"])
        command_source = "bank activation_scenario_payload"
    else:
        library, partition = commands_from
        feedback = commands_from_library(
            fixed_bid.get("service_date"), int(fixed_bid["forecast_seed"]), library, partition
        )
        command_source = f"{library}:{partition}"
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
        arrival_probs=arrival_probs,
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
        "result_version": RESULT_VERSION,
        "market": market,
        "stations": int(NUM_STATIONS),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "service_date": entry["service_date"],
        "bid_path": str(bid_path),
        "bid_sha256": hashlib.sha256(bid_path.read_bytes()).hexdigest(),
        "design_commands": len(design),
        "unseen_commands": len(feedback),
        "unseen_command_source": command_source,
        "ev_sampling": "independent_seed",
        "ev_seeds": ev_seed_values,
        "ev_counts": [len(evs) for evs in ev_bank],
        "solver_time_limit_s": time_limit_s,
        "elapsed_s": time.perf_counter() - started,
        "rows": rows,
    }


def unseen_command_source(count: int) -> str:
    return f"feedback partition, first {int(count)} (forecast_seed + feedback offset)"


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


def parse_banks(values: list[str]) -> dict[str, Path]:
    banks: dict[str, Path] = {}
    for value in values:
        market, sep, path = value.partition("=")
        if not sep or not market or not path:
            raise ValueError(f"--bank は 市場名=フォルダ の形で指定する: {value}")
        banks[market] = Path(path).resolve()
    return banks


def parse_command_sources(values: list[str]) -> dict[str, tuple[str, str]]:
    sources: dict[str, tuple[str, str]] = {}
    for value in values:
        market, sep, rest = value.partition("=")
        library, sep2, partition = rest.rpartition(":")
        if not sep or not sep2 or not market or not library or partition not in ("train", "validation", "test"):
            raise ValueError(f"--commands-from は 市場名=ライブラリ:区分 の形で指定する: {value}")
        sources[market] = (str(Path(library).resolve()), partition)
    return sources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bank", action="append", required=True, metavar="MARKET=DIR")
    parser.add_argument("--commands-from", action="append", default=[], metavar="MARKET=LIBRARY:PARTITION")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--time-limit-s", type=float, default=20.0)
    parser.add_argument("--ev-seeds", type=int, default=3)
    parser.add_argument("--max-days", type=int, default=0)
    parser.add_argument("--unseen-commands", type=int, default=0, metavar="N")
    args = parser.parse_args()
    if args.workers < 1 or args.time_limit_s <= 0 or args.ev_seeds < 1:
        raise ValueError("workers、time-limit-s、ev-seedsは正の値にする")
    if args.unseen_commands < 0:
        raise ValueError("--unseen-commands は0以上にする")
    if args.unseen_commands and args.commands_from:
        raise ValueError("--unseen-commands と --commands-from は一緒に使えない")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    command_sources = parse_command_sources(args.commands_from)
    entries_by_market = {}
    for market, bank_dir in parse_banks(args.bank).items():
        entries = list(BidBank(bank_dir).entries)
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
                expected_source = (
                    "{}:{}".format(*command_sources[market]) if market in command_sources
                    else unseen_command_source(args.unseen_commands) if args.unseen_commands
                    else "bank activation_scenario_payload"
                )
                if (
                    day.get("result_version") != RESULT_VERSION
                    or day.get("ev_sampling") != "independent_seed"
                    or len(day.get("ev_seeds", [])) != args.ev_seeds
                    or day.get("unseen_command_source") != expected_source
                ):
                    raise ValueError(f"保存済みの結果は今の版の評価ではない: {filename}")
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
                tasks.append((market, str(bank_dir), entry, args.time_limit_s, args.ev_seeds,
                              command_sources.get(market), args.unseen_commands))
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
    for market in entries_by_market:
        market_days = [day for day in day_results if day["market"] == market]
        summary[market] = market_summary(market_days)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
