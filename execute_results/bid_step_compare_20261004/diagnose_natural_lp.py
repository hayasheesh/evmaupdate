"""Why the natural-baseline LP is infeasible for one EV candidate (read only, no files written).

    python diagnose_natural_lp.py <stations> <growth> <YYYY-MM-DD> <forecast_seed> <candidate_index> [...]

Rebuilds the listed EV candidates exactly as the bid does (seed + 10007 x index)
and checks each vehicle's required energy against what its window allows at
full power, and each block's forced energy against the baseline ceiling.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    stations, growth, day, seed = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    indices = [int(x) for x in sys.argv[5:]]
    for key in list(os.environ):
        if key.startswith("EVMA_"):
            del os.environ[key]
    os.environ.update({"EVMA_NUM_STATIONS": stations, "EVMA_FUTURE_EV_ARRIVAL_GROWTH": growth,
                       "CUDA_VISIBLE_DEVICES": ""})
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    import torch
    torch.set_num_threads(1)
    from EnvConfig import LOWER_TRAIN_UPPER_BID_BASELINE_MAX_KW, LOWER_TRAIN_UPPER_BID_BASELINE_MIN_KW
    from environment.arrival_context import ArrivalScenarioSampler
    from market.physical_lp_bidding import ActivationScenario, BiddingLPConfig
    from market.physical_lp_bidding.solve_bidding import solve_natural_baseline_lp
    from training.lower_bid_training import sample_ev_specs_from_evenv

    cfg = BiddingLPConfig()
    scenario = ArrivalScenarioSampler().scenario_for_day(day)
    dt = cfg.dt_hours
    print(f"baseline bounds [{LOWER_TRAIN_UPPER_BID_BASELINE_MIN_KW:.0f}, {LOWER_TRAIN_UPPER_BID_BASELINE_MAX_KW:.0f}] kW")
    for index in indices:
        evs = sample_ev_specs_from_evenv(
            seed=seed + 10007 * index,
            arrival_probabilities_by_station=scenario.arrival_probabilities_by_station,
            day_context=scenario.day_context,
            service_date=day,
        )
        short = []
        for k, ev in enumerate(evs):
            a = int(np.clip(ev.arrival_t, 0, cfg.steps))
            d = int(np.clip(ev.departure_t, a + 1, cfg.steps))
            cap = float(ev.capacity_kwh)
            init = float(np.clip(ev.initial_kwh, 0, cap))
            if ev.target_required and int(ev.departure_t) <= cfg.steps:
                req = float(np.clip(ev.target_kwh, 0, cap))
            elif int(ev.departure_t) > cfg.steps:
                req = ev.terminal_min_kwh(cfg.steps, dt_hours=dt, eta_ch=cfg.eta_ch)
            else:
                req = 0.0
            room = float(ev.max_charge_kw) * dt * (d - a)
            need = req - init
            if need > min(room, cap - init) + 1e-9:
                short.append((k, a, d, int(ev.departure_t), ev.station_id, round(need, 4),
                              round(room, 4), round(cap - init, 4), ev.target_required))
        print(f"candidate {index}: {len(evs)} EVs, vehicles whose need exceeds the window: {len(short)}")
        for row in short[:10]:
            print("   ev %d arr %d dep %d (raw %d) station %s need %.4f kWh room %.4f kWh battery room %.4f target_required %s" % row)
        lp = ActivationScenario(name="n", up_signal=np.zeros(cfg.steps), down_signal=np.zeros(cfg.steps), evs=evs)
        for label, upper in (("with ceiling", LOWER_TRAIN_UPPER_BID_BASELINE_MAX_KW), ("no ceiling", None)):
            try:
                b = solve_natural_baseline_lp(lp, config=cfg, baseline_min_kw=LOWER_TRAIN_UPPER_BID_BASELINE_MIN_KW,
                                              baseline_max_kw=upper)
                print(f"   natural LP {label}: ok, max block {b.max():.0f} kW at block {int(b.argmax())}")
            except RuntimeError as exc:
                print(f"   natural LP {label}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
