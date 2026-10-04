"""入札バンクの各日について、約定量が当日の能力分布のどこに座るかを測る。

capability_distribution.py の一般化。単日の pickle ではなくバンクを読み、
局数の違う2つのバンクを同じ手順で比べられるようにする。
  usage: capability_distribution_bank.py <bank_dir> <out_json> [n_days] [n_draws]
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

BANK = sys.argv[1]
OUT = Path(sys.argv[2])
N_DAYS = int(sys.argv[3]) if len(sys.argv) > 3 else 25
N_DRAWS = int(sys.argv[4]) if len(sys.argv) > 4 else 24
EVAL_BASE = 910_000

from Config import NUM_STATIONS
from training.bid_bank import BidBank
from market.physical_lp_bidding import sample_ev_specs_from_evenv
from market.sustained_capability import sustained_capability_for_scenario
from environment.arrival_context import (
    ArrivalScenarioSampler, perturb_synthetic_arrival_probabilities)

print(f"[cap] bank={BANK} stations={NUM_STATIONS} days={N_DAYS} draws={N_DRAWS}", flush=True)
bank = BidBank(BANK)
sampler = ArrivalScenarioSampler()


def cap(evs):
    c = sustained_capability_for_scenario(evs, duration_hours=0.5)
    return np.asarray(c.up, float), np.asarray(c.down, float)


rows = []
for entry in bank.manifest["entries"][:N_DAYS]:
    day = entry["service_date"]
    fseed = int(entry["forecast_seed"])
    fb = bank.load_entry(entry)
    up = np.asarray(fb["up_plan"], float)
    down = np.asarray(fb["down_plan"], float)
    part = (up > 1e-8) | (down > 1e-8)
    if not part.any():
        print(f"[cap] {day} 参加ブロックなし、飛ばす", flush=True)
        continue

    scen = sampler.scenario_for_day(day)
    arr = getattr(scen, "arrival_probabilities_by_station", None)
    ctx = getattr(scen, "day_context", None)

    cert_evs = sample_ev_specs_from_evenv(
        seed=fseed, day_context=ctx,
        arrival_probabilities_by_station=(
            perturb_synthetic_arrival_probabilities(arr, seed=fseed)
            if arr is not None else None))
    cu, cd = cap(cert_evs)

    ups, downs, sizes = [], [], []
    for s in range(N_DRAWS):
        evs = sample_ev_specs_from_evenv(
            seed=EVAL_BASE + 1000 * s,
            arrival_probabilities_by_station=arr, day_context=ctx)
        u, d = cap(evs)
        ups.append(u); downs.append(d); sizes.append(len(evs))
    U = np.vstack(ups); D = np.vstack(downs)

    rec = {"day": day, "participating": int(part.sum()),
           "cert_sessions": len(cert_evs),
           "draw_sessions_min": int(min(sizes)), "draw_sessions_max": int(max(sizes))}
    for name, award, cert, mat in (("up", up, cu, U), ("down", down, cd, D)):
        m = part & (award > 1e-8)
        if not m.any():
            continue
        a = award[m]
        dr = mat[:, m]
        pct = (dr < a[None, :]).mean(axis=0) * 100.0
        cv = dr.std(axis=0, ddof=1) / np.maximum(dr.mean(axis=0), 1e-9)
        rec[name] = {
            "award_percentile_median": float(np.median(pct)),
            "cv_median": float(np.median(cv)),
            "dayof_median_over_award": float(np.median(np.median(dr, axis=0) / np.maximum(a, 1e-9))),
            "short_fraction": float((dr < a[None, :]).mean()),
            "cert_over_award_median": float(np.median(cert[m] / np.maximum(a, 1e-9))),
        }
    rows.append(rec)
    u = rec.get("up", {})
    print(f"[cap] {day} blocks={rec['participating']:2d} "
          f"分位={u.get('award_percentile_median', float('nan')):5.1f} "
          f"CV={u.get('cv_median', float('nan')):.3f} "
          f"当日中位/約定={u.get('dayof_median_over_award', float('nan')):.3f} "
          f"不足={u.get('short_fraction', float('nan'))*100:5.1f}%", flush=True)

summary = {"bank": BANK, "stations": int(NUM_STATIONS), "n_days": len(rows), "days": rows}
for name in ("up", "down"):
    for key in ("award_percentile_median", "cv_median",
                "dayof_median_over_award", "short_fraction"):
        vals = [r[name][key] for r in rows if name in r]
        summary[f"{name}_{key}"] = float(np.median(vals)) if vals else None
OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

print("\n===== まとめ（日をまたいだ中央値） =====", flush=True)
for name, label in (("up", "上げ"), ("down", "下げ")):
    print(f"  {label}  約定の分位 {summary[f'{name}_award_percentile_median']:5.1f}%tile   "
          f"CV {summary[f'{name}_cv_median']:.3f}   "
          f"当日中位/約定 {summary[f'{name}_dayof_median_over_award']:.3f}   "
          f"不足 {summary[f'{name}_short_fraction']*100:5.1f}%", flush=True)
print(f"  -> {OUT}", flush=True)
