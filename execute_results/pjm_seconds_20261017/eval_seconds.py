"""PJM の RegD を2秒ごとに追う層を足して、PJM の性能スコアを10秒ごとの値で出す（2026-10-17）。

    python eval_seconds.py --stage-dir <dir> --run-dir <dir> --episode N --mode <marl|marl_force|rule> --out <file.jsonl>
                           [--workers 4] [--lag 1] [--fast on|off] [--headroom reach|plain]

5分ごとの制御は今のまま（学習した行動器、または中央のルールベース。paper_eval_20261014/eval_episode.py と同じ入札・
40組・EVの乱数）。その5分の中で、指令が5分点の値から動いた分（2秒ごとの RegD から作った目標と、5分点の目標の差）を、
秒単位の層が受け持つ。層は station ごとに動き、使うのは自局のEV、共通の指令、1つ前の2秒の全体のずれ（連系点で
計れる値）だけ。
  - 各 station は、振れを station の数（設備の数で固定。EVのいない station も数える）で等分した量に、前の2秒で
    全体が受け持ちきれなかった分の等分を足して追う。受け持ちきれない station（EVがいない、余力が尽きた）は足さず、
    その分は次の2秒以降に他の station が足す。5分の制御の誤差（5分点のずれ）は追わない。
  - station の中では、上げ・下げの余力に比例して各EVに配る。余力は、定格までの電力と、電力量の上下限までの量の小さいほう。
    --headroom reach（既定）では、放電側の下限を「その5分のあと出発まで定格で充電すれば目標に届く電力量」
    （目標 − 定格 × 残りの時間 + 0.1 kWh、強制充電と同じ余裕）とSoC 0%の大きいほうにする。plain では SoC 0%。
    出発の判定は同じ5分の apply_action の中で済むので、その5分の終わりに出発するEVは枠が空になり（容量・定格0）、層は使わない。
  - 層は --lag 回前（既定1回＝2秒前）の指令を見て動く。信号の届く遅れと機器の応答の遅れの代わり。
  - 層が動かした電力量は、5分の終わりに各EVの SoC に足す（効率は環境と同じ1）。
採点は PJM Manual 12 Rev.57 §4.4.5 の式。5分の区間ごとに10秒ごとの値（30個）で
  誤差 = |応答 − 信号| / (0.5 × 区間の |信号| の平均 + 0.5 × AReg)、点数 = 1 − 誤差の平均。
10秒の遅れの許容は、信号を10秒遅らせた比べ方と遅らせない比べ方のよいほうを取る形で入れる（score_10s）。
許容なし（score_10s_nolag）と、今までの5分に1点の採点（score_5min、market/pjm_performance.py）も並べる。
AReg は今までと同じく、5分点の指令の向きの約定量（上げと下げが同じ PJM の規則の入札では、その量）。
--fast off では層を動かさず、5分の制御だけの応答（5分の間は一定）を同じ式で採点する。このとき 5分の結果は
paper_eval_20261014 の評価と同じになる。
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PE = ROOT / 'execute_results' / 'paper_eval_20261014'
SUB = 150          # 5分の中の2秒の数
SAMPLE = 5         # 10秒 = 2秒 × 5
DT_H = 2.0 / 3600.0


def run_episode(args):
    mode, day, c, e, lag, fast_on, headroom = args
    sys.path.insert(0, str(PE))
    sys.path.insert(0, str(HERE))
    import eval_episode as E
    task = 'pjm'
    E._setup(task)  # 設定を読み、ROOT を sys.path に入れる（以下の import より先に呼ぶ）
    import numpy as np
    import torch
    from Config import EPISODE_STEPS
    from market.bid_participation import participation_by_block
    from market.pjm_performance import areg_by_step, interval_scores
    from training.system_controller import apply_force_charging
    from regd2s import seconds_signal

    if mode in ('marl', 'marl_force'):
        # 本線の評価と同じく、先読みの入札観測を組み立てる前に学習時の正規化を読み込む
        from environment.normalize import load_observation_normalization_for_archive
        load_observation_normalization_for_archive(E.model_dir_and_episode(task)[0])
    fixed_bid, command, target, tol, tracking, arr = E._episode_inputs(task, day, c, 4)
    baseline = np.asarray(fixed_bid['baseline_plan'], float)
    up = np.asarray(fixed_bid['up_plan'], float)
    down = np.asarray(fixed_bid['down_plan'], float)
    part = participation_by_block(up, down)

    # 2秒ごとの目標（環境と同じ式: 基準 + 下げ×下げの割合 − 上げ×上げの割合、割合は0〜1に切る）
    r2 = seconds_signal(str(command['source']))
    up_p = np.clip(np.maximum(r2, 0.0), 0.0, 1.0)
    dn_p = np.clip(np.maximum(-r2, 0.0), 0.0, 1.0)
    for name, mine, ref in (('up', up_p[:, 0], command['up_proxy']), ('down', dn_p[:, 0], command['down_proxy'])):
        gap = float(np.max(np.abs(mine - np.asarray(ref, float)[:EPISODE_STEPS])))
        if gap > 1e-6:
            raise RuntimeError(f'2秒の信号の5分点が指令と合わない（{name}、差 {gap}）')
    blk = np.repeat(np.arange(48), 6)
    reg2 = (down[blk][:, None] * dn_p - up[blk][:, None] * up_p) * part[blk][:, None]
    target2 = baseline[blk][:, None] + reg2
    if float(np.max(np.abs(target2[part[blk], 0] - np.asarray(target, float)[part[blk]]))) > 1e-3:
        raise RuntimeError('2秒の目標の5分点が環境の目標と合わない')

    out = {'task': task, 'mode': mode, 'day': day, 'command': c, 'ev_seed': e, 'lag_steps': lag, 'fast_layer': fast_on, 'headroom': headroom,
           'service_date': str(fixed_bid.get('service_date')), 'command_source': str(command['source']),
           'offered_capacity_kw_block': float(np.sum(up + down))}
    env = E._make_env(task, day, c, e, target, tol, tracking, arr, central=(mode == 'rule'))
    agent = None
    if mode in ('marl', 'marl_force'):
        from environment.normalize import load_observation_normalization_for_archive, normalize_observation
        from training.system_controller import build_agent, find_model_path_and_episode
        d, ep = E.model_dir_and_episode(task)
        load_observation_normalization_for_archive(d)
        agent = build_agent(env)
        ckpt, ep = find_model_path_and_episode(str(d / 'results' / f'TEST{ep}'), ep)
        agent.load_actors(ckpt, ep, map_location='cpu')
        agent.set_test_mode(True)
        out.update({'model': d.name, 'episode': int(ep)})

    t0 = time.time()
    pre = np.zeros(EPISODE_STEPS)
    sig10 = np.full((EPISODE_STEPS, SUB // SAMPLE), np.nan)
    resp10 = np.full((EPISODE_STEPS, SUB // SAMPLE), np.nan)
    sat_steps = fast_steps = 0
    fast_abs_kwh = 0.0
    for t in range(EPISODE_STEPS):
        raw_obs = env.begin_step()
        if agent is not None:
            agent.update_active_evs(env)
            with torch.no_grad():
                actions = torch.as_tensor(agent.act(normalize_observation(raw_obs), env=env, noise=False), dtype=torch.float32)
        else:
            actions = env.soc.new_zeros((env.num_stations, env.max_ev_per_station))
        if mode in ('marl_force', 'rule'):
            actions, _f, _ = apply_force_charging(actions, env, slack_kwh=0.1)
        slots = [env._get_sorted_active_evs(st).clone() for st in range(env.num_stations)]
        prev_soc = env.soc.clone()
        _o, _lr, _gr, _d, info = env.apply_action(actions)
        pre[t] = float(info['pre_bess_residual_kw'])
        b = t // 6
        if not (part[b] and bool(tracking[t])):
            continue
        st_of, sl_of = [], []
        for st, s in enumerate(slots):
            for sl in s.tolist():
                st_of.append(st)
                sl_of.append(int(sl))
        st_of = np.asarray(st_of, int)
        sl_of = np.asarray(sl_of, int)
        ti_st, ti_sl = torch.as_tensor(st_of, dtype=torch.long), torch.as_tensor(sl_of, dtype=torch.long)
        kwh_pct = env.ev_kwh_per_soc_pct[ti_st, ti_sl].cpu().numpy().astype(float) if st_of.size else np.zeros(0)
        soc_now = env.soc[ti_st, ti_sl].cpu().numpy().astype(float) if st_of.size else np.zeros(0)
        p5 = ((soc_now - prev_soc[ti_st, ti_sl].cpu().numpy().astype(float)) * kwh_pct / (5.0 / 60.0)) if st_of.size else np.zeros(0)
        pmax = env.ev_max_power_kw[ti_st, ti_sl].cpu().numpy().astype(float) if st_of.size else np.zeros(0)
        e_kwh = soc_now * kwh_pct
        cap_kwh = 100.0 * kwh_pct
        floor_kwh = np.zeros(st_of.size)
        if headroom == 'reach' and st_of.size:
            # その5分のあと、出発まで定格で充電して目標に届く下限（強制充電と同じ余裕 0.1 kWh）
            target_kwh = env.target[ti_st, ti_sl].cpu().numpy().astype(float) * kwh_pct
            after = np.maximum(env.depart[ti_st, ti_sl].cpu().numpy().astype(float) - float(env.step_count), 0.0)
            floor_kwh = np.maximum(target_kwh - pmax * after * (5.0 / 60.0) + 0.1, 0.0) * (pmax > 0)
        # 5分の EV全体の電力は、環境のずれ（目標 − 全体の電力）から出す（同じ5分の終わりに出発したEVの分も入る）
        fleet5 = float(target[t]) - pre[t]
        delta = target2[t] - target2[t, 0]
        n = env.num_stations
        corr = np.zeros(env.num_stations)
        sat_prev = np.zeros(env.num_stations, bool)
        cum = np.zeros(st_of.size)
        resp = np.zeros(SUB)
        e_fast_prev = 0.0
        for j in range(SUB):
            seen = delta[j - lag] if j - lag >= 0 else 0.0
            if j:
                corr[~sat_prev] -= e_fast_prev / n
            want = seen / n + corr
            if st_of.size and fast_on:
                room_up = np.maximum(np.minimum(pmax - p5, (cap_kwh - e_kwh - cum) / DT_H), 0.0)
                room_dn = np.maximum(np.minimum(p5 + pmax, (e_kwh + cum - floor_kwh) / DT_H), 0.0)
                h_up = np.bincount(st_of, room_up, minlength=env.num_stations)
                h_dn = np.bincount(st_of, room_dn, minlength=env.num_stations)
                f_st = np.clip(want, -h_dn, h_up)
                sat_prev = np.abs(f_st - want) > 1e-9
                share_up = np.divide(room_up, h_up[st_of], out=np.zeros_like(room_up), where=h_up[st_of] > 0)
                share_dn = np.divide(room_dn, h_dn[st_of], out=np.zeros_like(room_dn), where=h_dn[st_of] > 0)
                f_ev = np.where(f_st[st_of] > 0, f_st[st_of] * share_up, f_st[st_of] * share_dn)
                cum += f_ev * DT_H
                fleet = fleet5 + float(f_ev.sum())
            else:
                sat_prev = np.abs(want) > 1e-9
                fleet = fleet5
            resp[j] = fleet
            e_fast_prev = (fleet - target2[t, j]) - (fleet5 - target2[t, 0])
            fast_steps += 1
            sat_steps += int(np.any(sat_prev))
        if st_of.size and fast_on:
            new = np.clip(soc_now + cum / np.maximum(kwh_pct, 1e-12), 0.0, 100.0)
            env.soc[ti_st, ti_sl] = torch.as_tensor(new, dtype=env.soc.dtype, device=env.soc.device)
            fast_abs_kwh += float(np.abs(cum).sum())
        sig10[t] = reg2[t, ::SAMPLE]
        resp10[t] = resp[::SAMPLE] - baseline[b]

    m = env.metrics
    dep = int(m['departing_evs'])
    hit_key = 'central_departing_evs_soc_met' if 'central_departing_evs_soc_met' in m else 'departing_evs_soc_met'
    disp_reg5 = np.asarray(target, float) - np.repeat(baseline, 6)
    award = np.where(disp_reg5 < -1e-6, np.repeat(up, 6), np.repeat(down, 6))
    out.update({'soc_hit_pct': 100.0 * int(m[hit_key]) / max(dep, 1), 'departed_evs': dep})
    out.update(E._judge(pre, tol, tracking, part, award))
    areg = areg_by_step(up, down, disp_reg5)
    assessed = np.isfinite(sig10[:, 0]) & (areg > 0)
    flat_sig = sig10.reshape(-1)
    s10, s10_nolag = np.full(EPISODE_STEPS, np.nan), np.full(EPISODE_STEPS, np.nan)
    for t in np.flatnonzero(assessed):
        best, first = -np.inf, None
        for delay in (0, 1):
            k0 = t * (SUB // SAMPLE)
            idx = np.arange(k0, k0 + SUB // SAMPLE) - delay
            idx = np.where(idx < 0, k0, idx)
            sig = flat_sig[idx]
            sig = np.where(np.isfinite(sig), sig, sig10[t, 0])
            norm = 0.5 * float(np.mean(np.abs(sig))) + 0.5 * float(areg[t])
            score = 1.0 - float(np.mean(np.abs(resp10[t] - sig))) / norm
            first = score if first is None else first
            best = max(best, score)
        s10[t], s10_nolag[t] = best, first
    s5 = interval_scores(pre, disp_reg5, areg, np.asarray(tracking, bool) & np.repeat(part, 6))
    out.update({
        'score_10s': s10.tolist(), 'score_10s_nolag': s10_nolag.tolist(), 'score_5min': s5.tolist(),
        'areg_kw': areg.tolist(), 'residual_kw': pre.tolist(),
        'fast_saturated_share': sat_steps / max(fast_steps, 1), 'fast_abs_energy_kwh': fast_abs_kwh,
        'ev_environment_seed': 4_000_000 + 1000 * day + 10 * c + e, 'seconds': time.time() - t0,
    })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage-dir', required=True)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--episode', required=True, type=int)
    ap.add_argument('--mode', required=True, choices=('marl', 'marl_force', 'rule'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--lag', type=int, default=1)
    ap.add_argument('--fast', default='on', choices=('on', 'off'))
    ap.add_argument('--headroom', default='reach', choices=('reach', 'plain'))
    ap.add_argument('--sets', default='all', help='all, or day:command:ev_seed[,..] for a quick check')
    args = ap.parse_args()
    os.environ.update({'TABLE3_SW_DIR': str(Path(args.stage_dir).resolve()), 'TABLE3_MODEL_DIR': str(Path(args.run_dir).resolve()),
                       'TABLE3_EPISODE': str(args.episode), 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {(r['day'], r['command'], r['ev_seed']) for r in map(json.loads, out.open(encoding='utf-8'))}
    if args.sets == 'all':
        sets = [(d, c, e) for d in range(5) for c in range(4) for e in range(2)]
    else:
        sets = [tuple(int(x) for x in s.split(':')) for s in args.sets.split(',')]
    jobs = [(args.mode, d, c, e, args.lag, args.fast == 'on', args.headroom) for d, c, e in sets if (d, c, e) not in done]
    mp.set_start_method('spawn', force=True)
    with mp.Pool(args.workers) as pool, open(out, 'a', encoding='utf-8') as f:
        for r in pool.imap_unordered(run_episode, jobs):
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
            f.flush()
            print(json.dumps({k: r[k] for k in ('day', 'command', 'ev_seed', 'soc_hit_pct', 'step_pass_pct', 'fast_saturated_share', 'seconds')}), flush=True)
    (out.parent / (out.stem + '.DONE')).write_text(f'{len(sets)} sets\n', encoding='utf-8')


if __name__ == '__main__':
    main()
