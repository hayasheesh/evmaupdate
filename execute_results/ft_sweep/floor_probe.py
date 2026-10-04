"""床は MARL の MAE に比例するか。ブロック別に約定量と誤差の関係を測る。

帯は約定量の10%。制御器が中に留まるには MAE <= 0.1*Q、つまり Q >= 10*MAE。
知りたいのは、ブロック内の平均絶対誤差が
  (a) 約定量によらずほぼ一定 -> 大きい約定量ほど帯が広く楽。床は台数で上がらない
  (b) 約定量に比例          -> 合格率は約定量によらず、床は比例させるべき
のどちらか。

evaluate_controller_precision は先頭シナリオしかブロック別ファイルを出さないので、
1指令ずつ呼び、base_seed をずらして元の24本と同じ realized_seed を再現する。
"""
from __future__ import annotations
import sys, os, json, pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent


def main() -> None:
    import numpy as np, pandas as pd, torch
    DAY, FORECAST_SEED, N_CMD, BASE = "2024-12-04", 1_076_030, 24, 910_000
    PIPELINE = os.environ.get("PROBE_PIPELINE", "system")
    WARM = ROOT / "archive/direct_bid_256cmd_25d_evcount_12hbid_7station_20260916_002104"

    from training.lower_bid_training import _activation_scenarios_for_day
    from training.evaluate_controller_precision import evaluate_controller_precision
    from training.system_controller import build_agent
    from training.agent_checkpoint import find_latest_checkpoint
    from environment.EVEnv import EVEnv
    from tools.evaluator import set_env_seed

    fixed_bid = pickle.loads((HERE / "fixed_bid_2024-12-04.pkl").read_bytes())
    cmds, mode = _activation_scenarios_for_day(DAY, FORECAST_SEED, n_scenarios=N_CMD,
                                               scenario_partition="holdout")
    ckpt, ep = find_latest_checkpoint(str(WARM))
    set_env_seed(BASE)
    env = EVEnv(); env.reset(net_demand_series=np.zeros(288, dtype=np.float32))
    agent = build_agent(env)
    agent.load_actors(ckpt, ep, map_location=None if torch.cuda.is_available() else "cpu")
    agent.set_test_mode(True)
    print(f"warm start {ckpt} ep{ep}  pipeline={PIPELINE}", flush=True)

    rows = []
    for s in range(N_CMD):
        one = dict(fixed_bid)
        one.update({"activation_scenario_payload": [cmds[s]],
                    "activation_scenarios": 1, "activation_mode": mode})
        out = HERE / f"floor_{PIPELINE}" / f"cmd{s:02d}"
        # 元の測定は realized_seed = BASE + 1000*s。単指令だと s=0 なので base をずらす。
        evaluate_controller_precision(agent, one, n_seeds=1, base_seed=BASE + 1000 * s,
                                      evaluation_pipeline=PIPELINE,
                                      out_dir=str(out), visualize=True)
        f = next((out / "results").glob("realized_s00_seed00_precision_by_block.csv"), None)
        if f is None:
            print(f"  cmd{s:02d}: ブロック別ファイルなし", flush=True); continue
        d = pd.read_csv(f); d["cmd"] = s
        rows.append(d)
        print(f"  cmd{s:02d} 済", flush=True)

    b = pd.concat(rows, ignore_index=True)
    b.to_csv(HERE / f"floor_blocks_{PIPELINE}.csv", index=False, encoding="utf-8")

    recs = []
    for dr in ("up", "down"):
        m = (b[f"{dr}_active_steps"] > 0) & (b[f"{dr}_kw"] > 1e-8)
        sub = b[m]
        if not len(sub):
            continue
        q = sub[f"{dr}_kw"].to_numpy(float)
        stay = sub[f"{dr}_stay_rate"].to_numpy(float)
        recs.append({"direction": dr, "n": int(len(sub)),
                     "約定量 中位": round(float(np.median(q)), 1),
                     "滞在率 中位": round(float(np.median(stay)), 4),
                     "合格(>=90%)率": round(float((stay >= 0.90).mean()), 4)})
        # 約定量の分位ごとの合格率 -> 床の水準
        edges = np.quantile(q, [0, .2, .4, .6, .8, 1.0])
        print(f"\n=== {dr}: 約定量の五分位ごと ({len(sub)} ブロック) ===", flush=True)
        for i in range(5):
            lo, hi = edges[i], edges[i + 1]
            sel = (q >= lo) & (q <= hi if i == 4 else q < hi)
            if sel.sum() == 0:
                continue
            print(f"  {lo:7.0f}-{hi:7.0f} kW  n={int(sel.sum()):3d}  "
                  f"滞在率中位 {np.median(stay[sel]):.3f}  "
                  f"合格率 {(stay[sel] >= 0.90).mean():.3f}", flush=True)
    (HERE / f"floor_summary_{PIPELINE}.json").write_text(
        json.dumps(recs, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n保存: floor_blocks_%s.csv" % PIPELINE, flush=True)


if __name__ == "__main__":
    main()
