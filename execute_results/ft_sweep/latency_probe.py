"""制御パス（actor順伝播→中央残差配分→BESS）の1ステップ所要時間と、
1サイクルあたりのモード反転EV数を測る。

S2の応答時間予算に対する余裕を出すのが目的。環境側の物理シミュレーションと
指標集計は実運用に存在しないので、full step からの差分として別に出す。
GPUは非同期なので、各区間の前後で synchronize してから計時する。
"""
from __future__ import annotations
import os, sys, json, time, pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent

MODE = sys.argv[1] if len(sys.argv) > 1 else "7s"
os.environ.setdefault("EVMA_ACTOR_EV_COUNT", "1")
os.environ.setdefault("EVMA_LOWER_BID_CONTEXT_OBS", "1")
os.environ.setdefault("EVMA_LOWER_BID_LOOKAHEAD_BLOCKS", "24")
if MODE == "20s":
    os.environ["EVMA_NUM_STATIONS"] = "20"
else:
    os.environ["EVMA_NUM_STATIONS"] = "7"

import numpy as np, torch

import environment.EVEnv as evenv_mod
from environment.EVEnv import EVEnv
from training.system_controller import build_agent

CUDA = torch.cuda.is_available()
def sync():
    if CUDA:
        torch.cuda.synchronize()

T = {k: [] for k in ("act", "alloc", "bess", "full_step", "begin_step")}
FLIPS, ACTIVE = [], []
_prev_sign = {"v": None}

def timed(name, fn):
    def wrapper(*a, **kw):
        sync(); t0 = time.perf_counter()
        out = fn(*a, **kw)
        sync(); T[name].append(time.perf_counter() - t0)
        return out
    return wrapper

# --- 中央残差配分：所要時間 + 補正後のEV単位指令の符号反転数 ---
_orig_alloc = evenv_mod.allocate_central_ev_residual
def alloc_probe(*a, **kw):
    sync(); t0 = time.perf_counter()
    actions, info = _orig_alloc(*a, **kw)
    sync(); T["alloc"].append(time.perf_counter() - t0)
    with torch.no_grad():
        cur = torch.sign(torch.where(actions.abs() < 1e-6,
                                     torch.zeros_like(actions), actions))
        active = (actions.abs() >= 1e-6)
        ACTIVE.append(int(active.sum().item()))
        prev = _prev_sign["v"]
        if prev is not None and prev.shape == cur.shape:
            # 充電↔放電の反転のみ数える（0を挟む立ち上げ/停止は L_mode 不要）
            flipped = (prev * cur) < 0
            FLIPS.append(int(flipped.sum().item()))
        _prev_sign["v"] = cur.clone()
    return actions, info
evenv_mod.allocate_central_ev_residual = alloc_probe

EVEnv._dispatch_residual_bess = timed("bess", EVEnv._dispatch_residual_bess)
EVEnv.begin_step = timed("begin_step", EVEnv.begin_step)
_orig_step = EVEnv.step
EVEnv.step = timed("full_step", _orig_step)

from tools.evaluator import set_env_seed
PAIRED_SEED = 910_000
set_env_seed(PAIRED_SEED)
env = EVEnv()
env.reset(net_demand_series=np.zeros(288, dtype=np.float32))
agent = build_agent(env)
agent.act = timed("act", agent.act)
agent.set_test_mode(True)

n_steps = 0
if MODE == "7s":
    # 学習済み7station。実入札 + holdout24指令、system パイプライン。
    from training.lower_bid_training import _activation_scenarios_for_day
    from training.evaluate_controller_precision import evaluate_controller_precision
    from training.agent_checkpoint import find_latest_checkpoint
    WARM = ROOT / "archive/direct_bid_256cmd_25d_evcount_12hbid_7station_20260916_002104"
    ckpt, ep = find_latest_checkpoint(str(WARM))
    agent.load_actors(ckpt, ep, map_location=None if CUDA else "cpu")
    print(f"warm start: {ckpt} ep{ep}", flush=True)
    fixed_bid = pickle.loads((HERE / "fixed_bid_2024-12-04.pkl").read_bytes())
    cmds, mode = _activation_scenarios_for_day("2024-12-04", 1_076_030,
                                               n_scenarios=4, scenario_partition="holdout")
    eval_bid = dict(fixed_bid)
    eval_bid.update({"activation_scenario_payload": list(cmds),
                     "activation_scenarios": len(cmds), "activation_mode": mode})
    evaluate_controller_precision(agent, eval_bid, n_seeds=1, base_seed=PAIRED_SEED,
                                  evaluation_pipeline="system",
                                  out_dir=str(HERE / f"lat_{MODE}"), visualize=False)
else:
    # 20station は未学習で可。計時は重みの値に依存しない。
    print(f"20station: 未学習エージェントで計時のみ（stations={env.num_stations}）", flush=True)
    rng = np.random.default_rng(0)
    demand = rng.normal(0, 400, 288).astype(np.float32)
    obs = env.reset(net_demand_series=demand)
    TARGET = 600
    while n_steps < TARGET:
        a = agent.act(obs, env=env, noise=False)
        obs, _, _, done, _ = env.step(a)
        n_steps += 1
        if done:
            demand = rng.normal(0, 400, 288).astype(np.float32)
            obs = env.reset(net_demand_series=demand)
    # 最初の20回はCUDAカーネルのウォームアップなので統計から外す。
    for k in T:
        if len(T[k]) > 40:
            T[k] = T[k][20:]
    del FLIPS[:20]; del ACTIVE[:20]
    print(f"20station: 未学習エージェントで計時のみ stations={env.num_stations} steps={n_steps}", flush=True)

def stats(v):
    if not v:
        return None
    a = np.array(v) * 1000.0
    return {"n": len(a), "mean_ms": round(float(a.mean()), 3),
            "p50_ms": round(float(np.percentile(a, 50)), 3),
            "p99_ms": round(float(np.percentile(a, 99)), 3),
            "max_ms": round(float(a.max()), 3)}

ctrl = np.array(T["act"][:len(T["alloc"])]) + np.array(T["alloc"][:len(T["act"])]) \
       if T["act"] and T["alloc"] else np.array([])
n = min(len(T["act"]), len(T["alloc"]), len(T["bess"]))
ctrl = (np.array(T["act"][:n]) + np.array(T["alloc"][:n]) + np.array(T["bess"][:n])) * 1000.0

out = {
    "mode": MODE, "stations": int(env.num_stations), "cuda": CUDA,
    "sections": {k: stats(v) for k, v in T.items()},
    "control_path_ms": {"n": int(n), "mean": round(float(ctrl.mean()), 3),
                        "p99": round(float(np.percentile(ctrl, 99)), 3),
                        "max": round(float(ctrl.max()), 3)} if n else None,
    "mode_flips_per_step": {"n": len(FLIPS), "mean": round(float(np.mean(FLIPS)), 2),
                            "p99": float(np.percentile(FLIPS, 99)),
                            "max": int(np.max(FLIPS))} if FLIPS else None,
    "active_ev_setpoints_per_step": {"mean": round(float(np.mean(ACTIVE)), 1),
                                     "max": int(np.max(ACTIVE))} if ACTIVE else None,
}
(HERE / f"latency_probe_{MODE}.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=1), flush=True)
