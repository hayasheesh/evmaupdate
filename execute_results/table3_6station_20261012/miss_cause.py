"""学習が進んでSoCが上がるにつれ追従が下がる理由を調べる：同じ40組（run_table3.py と同じ日・指令・EVの乱数）で、
指定した保存点の行動器（中央の配分・force充電・蓄電池なし、学習中のテストと同じ条件）を動かし、帯を外した5分を分ける。

    python miss_cause.py <aemo|ercot> <TESTの学習エピソード数> [workers]

各5分について、行動の前の状態から次を求める（kW、EVが充電する向きを正）。
  floor   出発に間に合わせるために今充電が要るEV（余裕 = 残りの5分の数 − 定格で目標まで要る5分の数 が1以下、
          その日に出発するEVだけ）の定格の合計。指令がこれを下回ると、帯に入れればそのEVのSoCが届かない
  hi      在車のEVのうち満充電でないEVの定格の合計（群全体で出せる充電の上限）
  lo      在車のEVのうちSoCが0でないEVの定格の合計に負号を付けたもの（放電の上限）
帯を外した5分（|指令 − EVの電力| > 帯、追従する5分だけ）を4つに分ける。
  over_floor   EVの電力が指令より多い、かつ 指令 < floor（間に合わせる充電を優先すると外れる）
  over_other   EVの電力が指令より多い、かつ 指令 ≥ floor
  short_range  EVの電力が指令より少ない、かつ 指令 > hi（群全体でも届かない）
  short_other  EVの電力が指令より少ない、かつ 指令 ≤ hi
結果は miss_cause_<task>_ep<N>.json。
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def run_episode(args):
    task, ep, day, c, e, n_cmd = args
    import run_table3 as R
    R._setup(task)
    import numpy as np
    import torch
    from Config import EPISODE_STEPS
    from EnvConfig import POWER_TO_ENERGY
    from environment.normalize import load_observation_normalization_for_archive, normalize_observation
    from training.system_controller import build_agent, find_model_path_and_episode

    _fixed, _cmd, target, tol, tracking, arr = R._episode_inputs(task, day, c, n_cmd)
    env = R._make_env(task, day, c, e, target, tol, tracking, arr, central=False)
    d, _latest = R.model_dir_and_episode(task)
    load_observation_normalization_for_archive(d)
    agent = build_agent(env)
    ckpt, got = find_model_path_and_episode(str(d / 'results' / f'TEST{ep}'), ep)
    agent.load_actors(ckpt, got, map_location='cpu')
    agent.set_test_mode(True)
    dt = float(POWER_TO_ENERGY)
    counts = dict(steps=0, miss=0, over_floor=0, over_other=0, short_range=0, short_other=0,
                  floor_above_cmd=0, abs_err_kw=0.0, over_err_kw=0.0, short_err_kw=0.0)
    for t in range(EPISODE_STEPS):
        obs = normalize_observation(env.begin_step())
        on = bool(env.current_tracking_enabled)
        band = float(env.tol_narrow_metrics)
        cmd = float(env.current_net_demand)
        floor = hi = lo = 0.0
        step = int(env.step_count)
        for st in range(env.num_stations):
            idx = env._get_sorted_active_evs(st)
            soc = env.soc[st, idx].tolist()
            tgt = env.target[st, idx].tolist()
            dep = env.depart[st, idx].tolist()
            rated = env.ev_max_power_kw[st, idx].tolist()
            spk = env.ev_soc_pct_per_kwh[st, idx].tolist()
            for k in range(len(soc)):
                if soc[k] < 100.0 - 1e-6:
                    hi += rated[k]
                if soc[k] > 1e-6:
                    lo -= rated[k]
                if dep[k] <= EPISODE_STEPS and tgt[k] > soc[k]:
                    need = (tgt[k] - soc[k]) / max(rated[k] * dt * spk[k], 1e-9)
                    if (dep[k] - step + 1) - need <= 1.0:
                        floor += rated[k]
        agent.update_active_evs(env)
        with torch.no_grad():
            actions = torch.as_tensor(agent.act(obs, env=env, noise=False), dtype=torch.float32)
        _o, _l, _g, _d, info = env.apply_action(actions)
        if not on:
            continue
        r = float(info['pre_bess_residual_kw'])     # 指令 − EVの電力（正なら EV が少ない）
        counts['steps'] += 1
        counts['abs_err_kw'] += abs(r)
        counts['floor_above_cmd'] += int(cmd < floor)
        if abs(r) <= band + 1e-6:
            continue
        counts['miss'] += 1
        if r < 0:
            counts['over_err_kw'] += -r
            counts['over_floor' if cmd < floor else 'over_other'] += 1
        else:
            counts['short_err_kw'] += r
            counts['short_range' if cmd > hi else 'short_other'] += 1
    m = env.metrics
    counts.update(day=day, command=c, ev_seed=e, episode=int(got),
                  soc_hit=int(m['departing_evs_soc_met']) if 'departing_evs_soc_met' in m else None,
                  departed=int(m['departing_evs']))
    return counts


def main():
    task, ep = sys.argv[1], int(sys.argv[2])
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    mp.set_start_method('spawn', force=True)
    jobs = [(task, ep, d, c, e, 4) for d in range(5) for c in range(4) for e in range(2)]
    with mp.Pool(workers) as pool:
        rows = pool.map(run_episode, jobs)
    keys = ['steps', 'miss', 'over_floor', 'over_other', 'short_range', 'short_other', 'floor_above_cmd',
            'abs_err_kw', 'over_err_kw', 'short_err_kw', 'departed']
    tot = {k: sum(r[k] for r in rows) for k in keys}
    hits = [r['soc_hit'] for r in rows if r['soc_hit'] is not None]
    tot['soc_hit'] = sum(hits) if hits else None
    out = {'task': task, 'episode': ep, 'rows': rows, 'total': tot}
    (HERE / f'miss_cause_{task}_ep{ep}.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    s = tot['steps']
    print(json.dumps({'task': task, 'ep': ep, 'step_pass': round(100 * (1 - tot['miss'] / s), 1),
                      'mae_kw': round(tot['abs_err_kw'] / s, 2),
                      **{k: tot[k] for k in ('miss', 'over_floor', 'over_other', 'short_range', 'short_other', 'floor_above_cmd')},
                      'over_err_kw': round(tot['over_err_kw'] / s, 2), 'short_err_kw': round(tot['short_err_kw'] / s, 2),
                      'soc_hit_pct': round(100 * tot['soc_hit'] / max(tot['departed'], 1), 1) if tot['soc_hit'] is not None else None}),
          flush=True)


if __name__ == '__main__':
    main()
