"""1日分の入札solveの所要を測る。局数は EVMA_NUM_STATIONS で外から与える。

bank構築と同じ条件に寄せるため scenario_workers は既定8。
Windows は spawn なので、子プロセスがこのモジュールを再実行しないよう
処理はすべて main() の中に置き __main__ ガードで守る。
"""
from __future__ import annotations
import sys, os, time, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent


def main() -> None:
    import numpy as np
    DAY = os.environ.get("PROBE_DAY", "2024-12-04")
    SEED = int(os.environ.get("PROBE_SEED", "1076030"))
    WORKERS = int(os.environ.get("PROBE_SCENARIO_WORKERS", "32"))

    import EnvConfig as E
    import Config as C
    print(f"局数={E.NUM_STATIONS}  局あたり上限EV={E.MAX_EV_PER_STATION}  "
          f"最低入札量={float(E.LOWER_TRAIN_RESEARCH_MINIMUM_BID_QUANTITY_KW):.0f}kW  "
          f"物理上限={E.PHYSICAL_MAX_POWER_KW:.0f}kW  workers={WORKERS}", flush=True)

    from training.lower_bid_training import (
        build_fixed_upper_bid_for_day, set_upper_bid_progress_log)
    from training.run_after_day_ahead_bid import _select_payload, _coerce_series
    from environment.readcsv import load_multiple_demand_files_with_labels
    from environment.arrival_context import ArrivalScenarioSampler

    set_upper_bid_progress_log(str(HERE / f"bidtiming_{E.NUM_STATIONS}s.log"), reset=True)
    all_data = load_multiple_demand_files_with_labels(train_split=25)
    payload = _select_payload(all_data, "all", DAY, 0)
    base_series, _ = _coerce_series(payload, int(C.EPISODE_STEPS))
    scen = ArrivalScenarioSampler().scenario_for_day(DAY)

    t0 = time.perf_counter()
    bid = build_fixed_upper_bid_for_day(base_series, DAY, arrival_scenario=scen,
                                        forecast_seed=SEED, scenario_workers=WORKERS)
    elapsed = time.perf_counter() - t0

    up = np.asarray(bid["up_plan"], float)
    down = np.asarray(bid["down_plan"], float)
    bank = bid.get("bid_ev_scenario_bank") or [[]]
    res = {
        "stations": int(E.NUM_STATIONS),
        "scenario_workers": WORKERS,
        "seconds": round(elapsed, 1),
        "ev_sessions": int(len(bank[0])),
        "minimum_bid_kw": float(E.LOWER_TRAIN_RESEARCH_MINIMUM_BID_QUANTITY_KW),
        "physical_max_kw": round(float(E.PHYSICAL_MAX_POWER_KW), 1),
        "capacity_kw_block": round(float(np.sum(up + down)), 1),
        "participating_blocks": int(np.count_nonzero((up > 1e-8) | (down > 1e-8))),
        "mean_up_kw": round(float(up.mean()), 1),
        "mean_down_kw": round(float(down.mean()), 1),
    }
    # 誰がシナリオを決着させたか。Benders の全ラウンドぶんを集計する。
    from collections import Counter
    solvers = Counter()
    feas = Counter()
    def walk(node, depth=0):
        """summary のどこに rounds/scenario_rows がぶら下がるか決め打ちしない。"""
        if depth > 6:
            return
        if isinstance(node, dict):
            for rnd in (node.get("rounds") or []):
                if isinstance(rnd, dict):
                    yield from (rnd.get("scenario_rows") or [])
            for v in node.values():
                yield from walk(v, depth + 1)
        elif isinstance(node, list):
            for v in node:
                yield from walk(v, depth + 1)

    summary = bid.get("bid_feasibility") or {}
    if True:
        if True:
            for row in walk(summary):
                solvers[str(row.get("oracle_solver", "")) or "(空)"] += 1
                feas[str(row.get("status", ""))] += 1
    if solvers:
        res["oracle_solver_counts"] = dict(solvers)
        res["scenario_status_counts"] = dict(feas)
        total = sum(solvers.values())
        print("シナリオを決着させた solver:", flush=True)
        for k, v in solvers.most_common():
            print(f"  {k:28s} {v:6d}  ({v/total:5.1%})", flush=True)
    else:
        res["oracle_solver_counts"] = "summaryに round_rows なし"
        print("round_rows が summary に載っていない", flush=True)

    print(json.dumps(res, ensure_ascii=False), flush=True)
    (HERE / f"bid_timing_{E.NUM_STATIONS}s.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{E.NUM_STATIONS}局: {elapsed/60:.1f}分  EV {res['ev_sessions']}セッション  "
          f"容量 {res['capacity_kw_block']:,.0f} kW-block  "
          f"参加 {res['participating_blocks']}/48", flush=True)


if __name__ == "__main__":
    main()
