"""24時で判定されるEVを、本来の出発に間に合う最低の量で判定し直す（行動器の動きは base と同じ）。

    python rejudge_dayend20.py <aemo|ercot> [--workers 10]

環境は、滞在が24時を越えるEVの出発を24時に置き換え、目標を「到着から定格で充電し続けた場合のSoC」と
本来の目標の小さいほうに置き換える（EVEnv._horizon_obligation）。入札器も、環境から写したこの値を使う。
ここでは、同じ回を回しながら置き換える前の出発時刻と目標を記録し、24時の判定だけを
「24時以降に定格で充電すれば本来の出発時刻に間に合う最低の量」（EVSpec.terminal_min_kwh と同じ式）で行う。
行動器は置き換えた目標を観測し、報酬もそれで受けて学習している。判定だけを変える。
"""
import argparse
import importlib.util
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('bidfeedback20', HERE / 'bidfeedback20.py')
BF = importlib.util.module_from_spec(spec)
# Workers unpickle BF.worker_init by module name; they must find this same module, whose ACTORS rollout reads.
sys.modules['bidfeedback20'] = BF
spec.loader.exec_module(BF)
OUT = HERE / 'dayend_rejudge'


def rollout(task):
    import numpy as np
    import torch
    import environment.EVEnv as evenv_module
    from EnvConfig import POWER_TO_ENERGY
    from tools.evaluator import set_env_seed
    from environment.EVEnv import EVEnv
    from environment.normalize import use_instruction_scale, normalize_observation

    class Recording(EVEnv):
        """Keeps, per slot, the departure and target an EV had before the day-end replacement."""

        def __init__(self):
            self._real_by_slot = {}
            self._pending_real = None
            super().__init__()

        def _horizon_obligation(self, target_soc, dep, capacity_kwh, max_power_kw, arrival_step, arrival_soc):
            out = super()._horizon_obligation(target_soc, dep, capacity_kwh, max_power_kw, arrival_step, arrival_soc)
            self._pending_real = (float(target_soc), int(dep), float(capacity_kwh), float(max_power_kw), out)
            return out

        def _set_ev_slot(self, station, slot_idx, ev_id, init_soc, target_soc, dep, *args, **kwargs):
            real = getattr(self, '_pending_real', None)
            self._pending_real = None
            if real is not None and real[4][0] == int(dep) and abs(real[4][1] - float(target_soc)) < 1e-9:
                self._real_by_slot[(int(station), int(slot_idx))] = real
            else:
                self._real_by_slot[(int(station), int(slot_idx))] = None
            return super()._set_ev_slot(station, slot_idx, ev_id, init_soc, target_soc, dep, *args, **kwargs)

    judged = []
    original = evenv_module.departure_rewards

    def capture(final_socs, target_socs, **kwargs):
        judged.append((final_socs.detach().cpu().numpy().copy(), target_socs.detach().cpu().numpy().copy()))
        return original(final_socs, target_socs, **kwargs)

    evenv_module.departure_rewards = capture
    ep = task['episode']
    arr = ep['arr']
    set_env_seed(task['ev_seed'])
    use_instruction_scale(arr.get('instruction_scale_kw', 1.0))
    env = Recording()
    env.use_central_ev_residual_allocator = False
    env.use_residual_bess = False
    env.reset(net_demand_series=ep['target'], tol_narrow_series=ep['tol'],
              tracking_enabled_series=np.asarray(arr['tracking_enabled_series'], bool),
              market_context_series=arr.get('market_context_series'),
              arrival_probabilities_by_station=arr.get('arrival_probabilities_by_station'),
              service_date=task['date'], baseline_series=arr.get('baseline_series'))
    horizon = int(env.episode_steps)
    rows = []
    with torch.no_grad():
        for _ in range(horizon):
            obs = normalize_observation(env.begin_step())
            action = torch.stack([BF.ACTORS[i](obs[i]) for i in range(20)])
            leaving = []
            for st in range(env.num_stations):
                slots = sorted(s for (s_st, s) in env._departure_slots_by_step.get(int(env.step_count), []) if s_st == st)
                for slot in slots:
                    if bool(env.ev_mask[st, slot]) and int(env.depart[st, slot]) == int(env.step_count):
                        leaving.append((st, slot, float(env.ev_capacity_kwh[st, slot]),
                                        float(env.ev_max_power_kw[st, slot]), env._real_by_slot.get((st, slot))))
            before = len(judged)
            env.apply_action(action, return_observation=False, include_active_evs=False,
                             include_ev_physics=False, include_reward_breakdown=False)
            if leaving:
                final, target = judged[before]
                assert len(final) == len(leaving), (len(final), len(leaving))
                for (st, slot, cap, power, real), f, t in zip(leaving, final, target):
                    env_hit = bool(f >= t)
                    real_dep = real[1] if real is not None else int(env.step_count)
                    if real is not None and real_dep > horizon:
                        real_target_kwh = real[0] * cap / 100.0
                        future = power * float(POWER_TO_ENERGY) * (real_dep - horizon)
                        minimum_kwh = min(max(real_target_kwh - future, 0.0), cap)
                        new_hit = bool(f * cap / 100.0 >= minimum_kwh - 1e-9)
                        rows.append([1, env_hit, new_hit, f, t, real[0], real_dep, minimum_kwh, cap, power, 1])
                    else:
                        rows.append([0, env_hit, env_hit, f, t, t, int(env.step_count), np.nan, cap, power,
                                     float(real is not None)])
    evenv_module.departure_rewards = original
    m = env.get_metrics()
    r = np.asarray(rows, float)
    day_end = r[:, 0] > 0
    row = {'id': task['id'], 'date': task['date'], 'ev_seed': task['ev_seed'],
           'tracking': float(m['tracking_success_rate']), 'soc_env': 100.0 - float(m['soc_miss_rate']),
           'departing': int(len(r)), 'day_end': int(day_end.sum()),
           'hits_env': int(r[:, 1].sum()), 'hits_new': int(r[:, 2].sum()),
           'day_end_hits_env': int(r[day_end, 1].sum()), 'day_end_hits_new': int(r[day_end, 2].sum()),
           'day_end_minimum_zero': int((r[day_end, 7] <= 1e-9).sum()),
           'day_end_short_env_kwh': float(np.maximum(r[day_end, 4] - r[day_end, 3], 0) @ r[day_end, 8] / 100.0),
           'day_end_short_new_kwh': float(np.maximum(r[day_end, 7] - r[day_end, 3] * r[day_end, 8] / 100.0, 0).sum()),
           'unrecorded': int((r[:, 10] == 0).sum())}
    out = OUT / task['market']
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"{task['id']}.npy", r)
    (out / f"{task['id']}.json").write_text(json.dumps(row), encoding='utf-8')
    return row


def main():
    import multiprocessing as mp
    import numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument('market', choices=['aemo', 'ercot'])
    ap.add_argument('--workers', type=int, default=10)
    a = ap.parse_args()
    BF.load_screen(a.market)
    days = BF.prepare_days(a.market)
    tasks = []
    for day in days:
        up = np.asarray(day['bid']['up_plan'], float)
        down = np.asarray(day['bid']['down_plan'], float)
        for cmd in range(2):
            ep = BF.build_episode(day, cmd, up, down)
            for ev in range(2):
                tid = f"{day['key']}_c{cmd}_e{ev}"
                if (OUT / a.market / f'{tid}.json').exists():
                    continue
                tasks.append({'market': a.market, 'id': tid, 'date': day['date'],
                              'ev_seed': day['ev_seeds'][ev], 'episode': ep})
    print(a.market, 'tasks', len(tasks), flush=True)
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context('spawn'),
                             initializer=BF.worker_init, initargs=(a.market,)) as pool:
        for fut in as_completed([pool.submit(rollout, t) for t in tasks]):
            fut.result()
    rows = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((OUT / a.market).glob('*.json'))]
    base = {p.stem: json.loads(p.read_text(encoding='utf-8')) for p in (HERE / a.market / 'base').glob('*.json')}
    mismatch = [r['id'] for r in rows if abs(r['tracking'] - base[r['id']]['tracking']) > 1e-9
                or abs(r['soc_env'] - base[r['id']]['soc']) > 1e-9]
    dep = sum(r['departing'] for r in rows)
    de = sum(r['day_end'] for r in rows)
    n = len(rows)
    summary = {
        'cases': n, 'same_as_base': not mismatch, 'mismatch': mismatch,
        'tracking': float(np.mean([r['tracking'] for r in rows])),
        'soc_env_metric': float(np.mean([r['soc_env'] for r in rows])),
        'soc_pooled_env': 100 * sum(r['hits_env'] for r in rows) / dep,
        'soc_pooled_new': 100 * sum(r['hits_new'] for r in rows) / dep,
        'soc_case_mean_new': float(np.mean([100 * r['hits_new'] / r['departing'] for r in rows])),
        'departing_per_day': dep / n, 'day_end_per_day': de / n,
        'day_end_hit_env': 100 * sum(r['day_end_hits_env'] for r in rows) / de,
        'day_end_hit_new': 100 * sum(r['day_end_hits_new'] for r in rows) / de,
        'day_end_minimum_zero_pct': 100 * sum(r['day_end_minimum_zero'] for r in rows) / de,
        'day_end_short_env_kwh_per_day': sum(r['day_end_short_env_kwh'] for r in rows) / n,
        'day_end_short_new_kwh_per_day': sum(r['day_end_short_new_kwh'] for r in rows) / n,
        'unrecorded_evs': sum(r['unrecorded'] for r in rows),
    }
    (OUT / f'{a.market}_summary.json').write_text(json.dumps(summary, indent=1), encoding='utf-8')
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == '__main__':
    main()
