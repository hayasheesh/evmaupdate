"""同じ未見の指令とEVで、全情報のLPで成り立つか、中央のルールベース、分散MARLを並べる。

    python run_table3.py <aemo|ercot> <lp|marl|marl_force|rule> [commands_per_day] [ev_seeds] [workers]

組は検証用の平日5日 × holdout の指令（既定4本）× EVの乱数（既定2通り）。指令とEVの乱数は
liu_lyu_2025_port_20261011/run_port.py と rule_bess_6station_20261011/run_rule_bess.py と同じ式にする
（set_env_seed(4_000_000 + 1000 * day + 10 * c + e)）。入札は wr_cap77wd_min70_margin7 の検証用。
  lp          その日の指令とEVを全部知ったLPで、帯（5分すべて）と出発時SoCを同時に満たす充放電があるか（certify、1組20秒まで）。
              EVは、同じ seed で環境を reset して出力0の1日を流し、到着したEVを取り出す（unseen_decomposition_aemo_20261002 と同じ）。
  marl        学習した行動器だけ（wr_obsdrop_cap77wd_<task>plan_6station_gs00625 の最新の TEST の保存点、中央の配分も BESS もなし）
  marl_force  行動器の後ろに force 充電（出発に間に合わなくなるEVを充電する層）を重ねる
  rule        中央観測のルールベース（行動0＋force充電、中央が station ごとの範囲を見て配る）。BESS なし
観測の設定は学習と同じにするため、wr_gs_6station_20261011/stage_worker.py の configure を使う。
結果は result_<task>_<mode>_<cmd>cmd_<ev>ev.jsonl（1行1組）。済んだ組は飛ばす。
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

# 学習が GPU を使っているので、ここでは CPU だけで動かす。
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Windows では空の文字列だと変数が消え、GPU が見えてしまう
HERE = Path(__file__).resolve().parent
# 既定は Windows の 6 station。研究室の 20 station では TABLE3_SW_DIR に wr_gs_20station_{task}_20261011 を渡す（{task} は市場名）。
SW_PATTERN = os.environ.get('TABLE3_SW_DIR', str(HERE.parent / 'wr_gs_6station_20261011'))
OUT_DIR = Path(os.environ.get('TABLE3_OUT_DIR', str(HERE)))
_CACHE = {}


def _setup(task):
    """stage_worker を読み込んで設定する。Windows 版は configure(task)・banks(task)・TASKS、研究室版は configure()・TEST_BANK・MODEL。"""
    if 'sw' not in _CACHE:
        sw_dir = Path(SW_PATTERN.format(task=task))
        sys.path.insert(0, str(sw_dir))
        import stage_worker
        if hasattr(stage_worker, 'banks'):
            stage_worker.configure(task)
            _CACHE['test_bank'] = stage_worker.banks(task)[1]
            _CACHE['model'] = stage_worker.TASKS[task][1]
        else:
            stage_worker.configure()
            _CACHE['test_bank'] = stage_worker.TEST_BANK
            _CACHE['model'] = stage_worker.MODEL
        _CACHE['root'] = sw_dir.parents[1]
        import torch
        torch.set_num_threads(1)
        _CACHE['sw'] = stage_worker
    return _CACHE['sw']


def model_dir_and_episode(task):
    _setup(task)
    explicit = os.environ.get('TABLE3_MODEL_DIR')
    d = Path(explicit).resolve() if explicit else sorted((_CACHE['root'] / 'archive').glob(f"{_CACHE['model']}_2*"))[-1]
    ep = max(int(p.name[4:]) for p in (d / 'results').glob('TEST*') if p.name[4:].isdigit())
    # TABLE3_EPISODE で保存点を固定する（学習が進んだあとに、前と同じ保存点で測り直すため）。
    ep = int(os.environ.get('TABLE3_EPISODE', ep))
    return d, ep


def _episode_inputs(task, day, c, n_cmd):
    import numpy as np
    from Config import EPISODE_STEPS
    from training.bid_bank import BidBank
    from training.lower_bid_training import _activation_scenarios_for_day, build_fixed_upper_bid_training_episode

    _setup(task)
    bank = BidBank(_CACHE['test_bank'])
    base_bid = dict(bank.load_entry(bank.entries[day]))
    payloads, mode = _activation_scenarios_for_day(base_bid.get('service_date'), int(base_bid.get('forecast_seed', 0)) + 512_209,
                                                   n_scenarios=n_cmd, scenario_partition='holdout')
    fixed_bid = dict(base_bid, activation_scenario_payload=list(payloads), activation_scenarios=len(payloads), activation_mode=mode)
    target, tol, arr, _ = build_fixed_upper_bid_training_episode(fixed_bid, c)
    target = np.asarray(target, float).reshape(-1)[:EPISODE_STEPS]
    tol = np.asarray(tol, float).reshape(-1)[:EPISODE_STEPS]
    tracking = np.asarray(arr.get('tracking_enabled_series', np.ones(EPISODE_STEPS, bool)), bool).reshape(-1)[:EPISODE_STEPS]
    return fixed_bid, payloads[c], target, tol, tracking, arr


def _make_env(task, day, c, e, target, tol, tracking, arr, central):
    from environment.EVEnv import EVEnv
    from environment.normalize import use_instruction_scale
    from tools.evaluator import set_env_seed

    set_env_seed(4_000_000 + 1000 * day + 10 * c + e)
    use_instruction_scale(arr.get('instruction_scale_kw', 1.0))
    env = EVEnv()
    env.use_central_ev_residual_allocator = central
    env.use_residual_bess = False
    env.reset(net_demand_series=target, tol_narrow_series=tol, tracking_enabled_series=tracking,
              market_context_series=arr.get('market_context_series'),
              arrival_probabilities_by_station=arr.get('arrival_probabilities_by_station'),
              service_date=arr.get('service_date'), baseline_series=arr.get('baseline_series'))
    return env


def _capture_evs(env):
    """market/physical_lp_bidding/evenv_adapter.py と同じ取り出し方（EVEnv の SoC は %、EVSpec は割合）。"""
    import numpy as np
    import torch
    from Config import EPISODE_STEPS
    from market.physical_lp_bidding.data_classes import EVSpec

    seen, specs = set(), []

    def capture():
        for station in range(env.num_stations):
            for slot_tensor in torch.nonzero(env.ev_mask[station], as_tuple=False).squeeze(-1):
                slot = int(slot_tensor.item())
                key = (int(env.ev_ids[station, slot].item()), int(env.arrival_step[station, slot].item()))
                if key in seen:
                    continue
                seen.add(key)
                arrival = max(key[1] - 1, 0)
                raw_departure = int(env.depart[station, slot].item())
                specs.append(EVSpec(
                    arrival_t=int(np.clip(arrival, 0, EPISODE_STEPS - 1)), departure_t=max(raw_departure, arrival + 1),
                    initial_soc=float(env.soc[station, slot].item()) / 100.0, target_soc=float(env.target[station, slot].item()) / 100.0,
                    capacity_kwh=float(env.ev_capacity_kwh[station, slot].item()),
                    max_charge_kw=float(env.ev_max_power_kw[station, slot].item()),
                    max_discharge_kw=float(env.ev_max_power_kw[station, slot].item()),
                    station_id=int(station), ev_id=f'{key[0]}@{key[1]}',
                    target_required=bool(raw_departure <= EPISODE_STEPS)))

    capture()
    actions = torch.zeros((env.num_stations, env.max_ev_per_station), dtype=torch.float32, device=env.soc.device)
    with torch.no_grad():
        for _ in range(EPISODE_STEPS):
            env.begin_step()
            capture()
            _o, _l, _g, done, _i = env.apply_action(actions, build_info=False, return_observation=False)
            if bool(np.all(np.asarray(done, dtype=bool))):
                break
    return specs


def _judge(res, tol, tracking, part, award_steps):
    import numpy as np
    err = np.abs(res)
    step_pass = err <= tol + 1e-6
    blk = [float(np.mean(step_pass[k * 6:(k + 1) * 6])) >= 0.90 for k in range(48) if part[k]]
    pmask = np.repeat(part, 6)
    return {
        'step_pass_pct': 100.0 * float(np.mean(step_pass[tracking])),
        'all_steps_in_band': bool(np.all(step_pass[tracking])),
        'block_pass_pct': 100.0 * float(np.mean(blk)) if blk else float('nan'),
        'blocks_assessed': len(blk), 'blocks_failed': int(len(blk) - sum(blk)),
        'mean_error_to_award_pct': 100.0 * float(np.sum(err[pmask]) / max(np.sum(award_steps[pmask]), 1e-9)),
    }


def run_episode(args):
    task, mode, day, c, e, n_cmd = args
    _setup(task)
    import numpy as np
    import torch
    from Config import EPISODE_STEPS
    from market.bid_participation import participation_by_block
    from market.physical_lp_bidding.colgen_feasibility import certify
    from market.physical_lp_bidding.data_classes import BiddingLPConfig
    from market.physical_lp_bidding.joint_validation import fixed_bid_tracking_bands
    from training.system_controller import apply_force_charging

    # 先読みの入札観測を組み立てる前に、学習時の正規化を読み込む。
    if mode in ('marl', 'marl_force'):
        from environment.normalize import load_observation_normalization_for_archive
        model_path, _model_ep = model_dir_and_episode(task)
        load_observation_normalization_for_archive(model_path)
    fixed_bid, command, target, tol, tracking, arr = _episode_inputs(task, day, c, n_cmd)
    baseline = np.asarray(fixed_bid['baseline_plan'], float)
    up = np.asarray(fixed_bid['up_plan'], float)
    down = np.asarray(fixed_bid['down_plan'], float)
    out = {'task': task, 'mode': mode, 'day': day, 'command': c, 'ev_seed': e,
           'service_date': str(fixed_bid.get('service_date')), 'offered_capacity_kw_block': float(np.sum(up + down))}
    t0 = time.time()
    if mode == 'lp':
        # PJM の規則の入札は、指令に比例して広がる帯と指令0の帯も、入札を作ったときと同じにする
        # （training/blockwise_bid.py の BiddingLPConfig と同じ。日本の規則では 0 と None で、前と同じ帯になる）。
        cfg = BiddingLPConfig(assessment_band_fraction=float(fixed_bid['assessment_band_fraction']),
                              apply_transition_band=bool(fixed_bid['apply_transition_band']),
                              band_signal_fraction=float(fixed_bid.get('band_signal_fraction', 0.0)),
                              idle_band_fraction=(float(fixed_bid['idle_band_fraction'])
                                                  if fixed_bid.get('market_tracking_rule') == 'pjm' else None))
        _t, _tol, lower, upper = fixed_bid_tracking_bands(cfg, baseline, up, down, np.asarray(command['up_proxy'], float),
                                                         np.asarray(command['down_proxy'], float),
                                                         apply_transition_band=cfg.apply_transition_band)
        env = _make_env(task, day, c, e, target, tol, tracking, arr, central=False)
        evs = _capture_evs(env)
        feasible, _, info = certify(evs, lower, upper, steps=cfg.steps, dt=cfg.dt_hours, eta_ch=cfg.eta_ch,
                                    return_dispatch=False, time_limit_s=20.0)
        diff = max(float(np.max(np.abs(lower[tracking] - (target - tol)[tracking]))),
                   float(np.max(np.abs(upper[tracking] - (target + tol)[tracking])))) if tracking.any() else 0.0
        out.update({'feasible': feasible, 'reason': info.get('reason') if feasible is None else '', 'evs': len(evs),
                    'band_max_diff_kw': diff, 'assessed_mask_equal': bool(np.array_equal(tracking, np.isfinite(lower) & np.isfinite(upper)))})
    else:
        env = _make_env(task, day, c, e, target, tol, tracking, arr, central=(mode == 'rule'))
        agent = None
        if mode in ('marl', 'marl_force'):
            from environment.normalize import load_observation_normalization_for_archive, normalize_observation
            from training.system_controller import build_agent, find_model_path_and_episode
            d, ep = model_dir_and_episode(task)
            load_observation_normalization_for_archive(d)
            agent = build_agent(env)
            ckpt, ep = find_model_path_and_episode(str(d / 'results' / f'TEST{ep}'), ep)
            agent.load_actors(ckpt, ep, map_location='cpu')
            agent.set_test_mode(True)
            out.update({'model': d.name, 'episode': int(ep)})
        disp = np.zeros(EPISODE_STEPS)
        pre = np.zeros(EPISODE_STEPS)
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
            _o, _lr, _gr, _d, info = env.apply_action(actions)
            disp[t] = float(env.current_net_demand)
            pre[t] = float(info['pre_bess_residual_kw'])
        m = env.metrics
        part = participation_by_block(up, down)
        reg = disp - np.repeat(baseline, 6)
        award = np.where(reg < -1e-6, np.repeat(up, 6), np.repeat(down, 6))
        dep = int(m['departing_evs'])
        hit_key = 'central_departing_evs_soc_met' if 'central_departing_evs_soc_met' in m else 'departing_evs_soc_met'
        out.update({'soc_hit_pct': 100.0 * int(m[hit_key]) / max(dep, 1), 'departed_evs': dep})
        out.update(_judge(pre, tol, tracking, part, award))
        out['mae_kw'] = float(np.mean(np.abs(pre)[tracking]))
        out['ev_environment_seed'] = 4_000_000 + 1000 * day + 10 * c + e
        out['traces'] = {'residual_kw': pre.tolist(), 'target_kw': disp.tolist(),
                         'tol_kw': np.asarray(tol, float).tolist(),
                         'tracking': np.asarray(tracking, bool).tolist(),
                         'baseline_kw': baseline.tolist(), 'up_kw': up.tolist(), 'down_kw': down.tolist()}
        if os.environ.get('TABLE3_BLOCKS') == '1':
            # コマごとの上げ・下げの約定量と合否（約定量の大きさごとの合格率を出すため）。
            step_pass = np.abs(pre) <= tol + 1e-6
            out['blocks'] = [{'k': k, 'up': float(up[k]), 'down': float(down[k]),
                              'pass': bool(np.mean(step_pass[k * 6:(k + 1) * 6]) >= 0.90),
                              'max_err_kw': float(np.max(np.abs(pre[k * 6:(k + 1) * 6]))),
                              'band_kw': float(np.min(tol[k * 6:(k + 1) * 6])),
                              'baseline_kw': float(baseline[k]),
                              'baseline_step_kw': float(abs(baseline[k] - baseline[k-1])) if k and part[k-1] else None,
                              'first_step_error_kw': float(abs(pre[k * 6])),
                              'mae_kw': float(np.mean(np.abs(pre[k * 6:(k + 1) * 6])))} for k in range(48) if part[k]]
    out['seconds'] = time.time() - t0
    return out


def main():
    task, mode = sys.argv[1], sys.argv[2]
    assert mode in ('lp', 'marl', 'marl_force', 'rule')
    n_cmd = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    n_ev = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    workers = int(sys.argv[5]) if len(sys.argv) > 5 else 8
    mp.set_start_method('spawn', force=True)
    out = OUT_DIR / f'result_{task}_{mode}_{n_cmd}cmd_{n_ev}ev.jsonl'
    done = set()
    if out.exists():
        done = {(r['day'], r['command'], r['ev_seed']) for r in map(json.loads, out.open(encoding='utf-8'))}
    jobs = [(task, mode, d, c, e, n_cmd) for d in range(5) for c in range(n_cmd) for e in range(n_ev) if (d, c, e) not in done]
    with mp.Pool(workers) as pool, open(out, 'a', encoding='utf-8') as f:
        for r in pool.imap_unordered(run_episode, jobs):
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
            f.flush()
    print(json.dumps({'task': task, 'mode': mode, 'done': len(jobs) + len(done)}), flush=True)


if __name__ == '__main__':
    main()
