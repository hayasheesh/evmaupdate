"""10月3日の20 station AEMO（prod_aemoplan128_AB_20station_5080_scaledreward_20261003_080522）を、
当時のコードで評価し直す。学習はしない。

    python repro_old20.py validate --episode 1100
    python repro_old20.py trace --episode 1100 [--episodes 40]

コードは研究室の code/（実行の控え＋ git 8872cbb の market/ と legacy/、data/ は本体への symlink）。
設定は、実行の resume/latest.json に保存された値に合わせる（環境変数 ENV。残る違いは評価に効かない
探索の雑音、再生バッファの大きさ、数値の型だけ）。

評価用エピソードの作り方は、当時の run_after_day_ahead_bid が組んで train() に渡すものをそのまま使う。
train() を差し替えて引数を受け取り、学習せずに止める。

validate：当時の定期評価（5エピソード、乱数910000）を同じ行動器で回し、保存されている時刻ごとのCSVと比べる。
trace：同じ評価用エピソードを回し、EVごと・時刻ごとの値を記録する（今の20 stationとの比較用）。
"""
import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

B = Path(__file__).resolve().parent
CODE = B / 'code'
RUN = Path('/workspace/EVMALOCALUPDATE/archive/prod_aemoplan128_AB_20station_5080_scaledreward_20261003_080522')
BANKS = Path('/workspace/EVMALOCALUPDATE/execute_results/bid_banks')
TRAIN_BANK = BANKS / 'train_25_20station_128cmd_3ev_aemo_plan_deviation'
TEST_BANK = BANKS / 'validation_5_20station_128cmd_3ev_aemo_plan_deviation'
ENV = {
    'EVMA_NUM_STATIONS': '20',
    'EVMA_ACTOR_EV_COUNT': '1',
    'EVMA_GLOBAL_BALANCE_REWARD_ERROR_SCALE_KW': '428.57142857142856',
    'EVMA_GLOBAL_BALANCE_REWARD_LINEAR_TAIL_KW': '171.42857142857142',
    'EVMA_LOCAL_FLEET_RESIDUAL_OBS': '1',
    'EVMA_LOWER_BID_LOOKAHEAD_BLOCKS': '24',
    'EVMA_LOWER_TRAIN_RESEARCH_MINIMUM_BID_QUANTITY_KW': '250',
    'EVMA_LOWER_TRAIN_BUILD_BID_BANK': '0',
    'EVMA_LOWER_TRAIN_USE_BID_BANK': '1',
    'EVMA_FINETUNE_ENABLE': '0',
}
CAPTURED = {}


class Captured(Exception):
    pass


def setup():
    for k in list(os.environ):
        if k.startswith('EVMA_'):
            del os.environ[k]
    os.environ.update(ENV)
    os.environ.update(CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', MPLBACKEND='Agg')
    manifest = json.loads((TRAIN_BANK / 'manifest.json').read_text(encoding='utf-8'))
    os.environ['EVMA_LOWER_TRAIN_UPPER_BID_ACTIVATION_SOURCE_DIR'] = manifest['settings']['activation_source_dir']
    sys.path.insert(0, str(CODE))
    import Config
    # The replay buffer is never filled here; a small one keeps memory down.
    Config.MEMORY_SIZE = 1000
    Config.GLOBAL_CORRELATED_NOISE_GAIN = 0.0
    import torch
    torch.set_num_threads(1)
    import training.train as T

    def fake_train(**kwargs):
        CAPTURED.update(kwargs)
        raise Captured

    T.train = fake_train
    from training.run_after_day_ahead_bid import run_after_day_ahead_bid
    try:
        run_after_day_ahead_bid(episodes=1, model_name='repro_old20', forecast_seed=73000, train_split_count=25,
                                use_train_bid_bank=True, bid_bank_dir=str(TRAIN_BANK),
                                test_bid_bank_dir=str(TEST_BANK), resume_run_dir=None,
                                resume_checkpoint_interval=0)
    except Captured:
        pass
    assert 'test_episode_preparer' in CAPTURED, sorted(CAPTURED)
    return T


def build_agent(T, episode):
    from environment.EVEnv import EVEnv
    from environment.observation_config import local_obs_dim
    from training.Agent.standard_maddpg import build_marl_agent
    env = EVEnv(num_stations=T.NUM_STATIONS, num_evs=T.NUM_EVS, episode_steps=T.EPISODE_STEPS)
    agent = build_marl_agent(
        s_dim=local_obs_dim(env.max_ev_per_station), max_evs_per_station=env.max_ev_per_station,
        n_agent=env.num_stations, num_episodes=2000, batch=T.BATCH_SIZE, gamma=T.GAMMA, tau=T.TAU,
        lr_a=T.LR_ACTOR, lr_c=T.LR_CRITIC_LOCAL, lr_global_c=T.LR_GLOBAL_CRITIC, tau_global=T.TAU_GLOBAL,
        td3_sigma=T.TD3_SIGMA_GLOBAL, td3_clip=T.TD3_CLIP_GLOBAL, smoothl1_beta=T.SMOOTHL1_BETA)
    agent.load_actors(str(RUN / 'results' / f'TEST{episode}'), episode, map_location='cpu')
    return agent


def work_dir(name):
    w = B / name
    (w / 'input').mkdir(parents=True, exist_ok=True)
    (w / 'results').mkdir(parents=True, exist_ok=True)
    shutil.copy(RUN / 'input' / 'observation_normalization.json', w / 'input' / 'observation_normalization.json')
    for f in ('bid_bank_manifest.json', 'bid_bank_test_manifest.json'):
        if (RUN / 'results' / f).exists():
            shutil.copy(RUN / 'results' / f, w / 'results' / f)
    return w


def compare_csv(a, b):
    ra = list(csv.DictReader(open(a, encoding='utf-8')))
    rb = list(csv.DictReader(open(b, encoding='utf-8')))
    if len(ra) != len(rb):
        return f'rows {len(ra)} vs {len(rb)}'
    worst = 0.0
    for x, y in zip(ra, rb):
        for k in x:
            try:
                worst = max(worst, abs(float(x[k]) - float(y[k])))
            except (TypeError, ValueError):
                if x[k] != y[k]:
                    return f'column {k}: {x[k]!r} vs {y[k]!r}'
    return f'max abs diff {worst:.3g}'


def validate(episode):
    T = setup()
    from tools.evaluator import test
    agent = build_agent(T, episode)
    w = work_dir(f'validate_ep{episode}')
    results = test(agent, random_window=False, working_dir=str(w), test_results=None, test_episode_num=episode,
                   demand_data_override=CAPTURED['test_demand_data_override'],
                   arrival_sampler_override=CAPTURED['arrival_sampler_override'],
                   episode_preparer=CAPTURED['test_episode_preparer'], num_episodes=int(T.INTERIM_TEST_EPISODES),
                   eval_seed=int(T.INTERIM_TEST_SEED), enable_png=True, enable_history_png=False,
                   save_test_detail_files=True, update_history=False, verbose=False, print_summary=True)
    out = w / 'results' / f'TEST{episode}'
    ref = RUN / 'results' / f'TEST{episode}'
    for f in sorted(ref.glob('test_results_zz_station_cooperation_full_episode_*.csv')):
        mine = out / f.name
        print(f.name, compare_csv(mine, f) if mine.exists() else 'missing', flush=True)
    rows = list(csv.DictReader(open(RUN / 'results' / 'test_performance_metrics.csv', encoding='utf-8')))
    row = [r for r in rows if int(r['Episode']) == episode // 20]
    print('archived test row', json.dumps(row[0] if row else None)[:400], flush=True)


def trace(episode, episodes):
    """Per-step and per-EV records on the archived run's test episodes (no plots)."""
    import numpy as np
    import torch
    T = setup()
    import environment.EVEnv as E
    from EnvConfig import POWER_TO_ENERGY
    agent = build_agent(T, episode)
    w = work_dir(f'trace_ep{episode}')
    out = w / 'trace'
    out.mkdir(exist_ok=True)
    original = E.EVEnv.apply_action

    def recording(self, actions, *args, **kwargs):
        before = []
        for st in range(self.num_stations):
            for j, slot in enumerate(self._get_sorted_active_evs(st).tolist()):
                soc = float(self.soc[st, slot]); cap = float(self.ev_capacity_kwh[st, slot])
                power = float(self.ev_max_power_kw[st, slot]); target = float(self.target[st, slot])
                remain = int(self.depart[st, slot]) - self.step_count + 1
                lo = max(-power, -soc * cap / 100 / POWER_TO_ENERGY)
                hi = min(power, (100 - soc) * cap / 100 / POWER_TO_ENERGY)
                floor = max(lo, (target - soc) * cap / 100 / POWER_TO_ENERGY - (remain - 1) * power)
                before.append((st, j, slot, soc, target, cap, power, remain, lo, hi, floor,
                               float(actions[st, j]) if j < actions.shape[1] else float('nan')))
        req = float(self.current_net_demand); band = float(self.tol_narrow_metrics)
        enabled = bool(self.current_tracking_enabled)
        result = original(self, actions, *args, **kwargs)
        info = result[4] if isinstance(result, tuple) and len(result) >= 5 else {}
        executed = info.get('actual_ev_power_kw') if isinstance(info, dict) else None
        resid = float(getattr(self, 'last_raw_actor_residual_kw', float('nan')))
        rec = self._trace_rec
        rec['step'].append([self.step_count, enabled, req, band, resid, len(before),
                            sum(b[8] for b in before), sum(b[9] for b in before), sum(b[10] for b in before)])
        for k, b in enumerate(before):
            ex = float(executed[b[0], b[1]]) if executed is not None else float('nan')
            rec['ev'].append([self.step_count, *b[:2], *b[3:], ex])
        return result

    E.EVEnv.apply_action = recording
    original_reset = E.EVEnv.reset

    def reset(self, *args, **kwargs):
        self._trace_rec = {'step': [], 'ev': []}
        TRACES.append(self)
        return original_reset(self, *args, **kwargs)

    TRACES = []
    E.EVEnv.reset = reset
    from tools.evaluator import test
    results = test(agent, random_window=False, working_dir=str(w), test_results=None, test_episode_num=episode,
                   demand_data_override=CAPTURED['test_demand_data_override'],
                   arrival_sampler_override=CAPTURED['arrival_sampler_override'],
                   episode_preparer=CAPTURED['test_episode_preparer'], num_episodes=int(episodes),
                   eval_seed=int(T.INTERIM_TEST_SEED), enable_png=False, enable_history_png=False,
                   save_test_detail_files=False, update_history=False, verbose=False, print_summary=True)
    kept = 0
    for i, env in enumerate(TRACES):
        rec = getattr(env, '_trace_rec', None)
        if not rec or len(rec['step']) < int(T.EPISODE_STEPS):
            continue
        np.savez_compressed(out / f'episode_{kept + 1}.npz', step=np.asarray(rec['step'], float),
                            ev=np.asarray(rec['ev'], np.float32))
        kept += 1
    meta = {'episode': episode, 'episodes_requested': int(episodes), 'episodes_saved': kept,
            'step_columns': ['step', 'tracking_enabled', 'request_kw', 'band_kw', 'residual_kw', 'EV_count',
                             'fleet_min_kw', 'fleet_max_kw', 'deadline_floor_sum_kw'],
            'ev_columns': ['step', 'station', 'slot_order', 'soc_before', 'target', 'capacity_kwh', 'rated_kw',
                           'remaining_steps', 'instant_min_kw', 'instant_max_kw', 'deadline_min_kw', 'action',
                           'executed_kw']}
    (out / 'meta.json').write_text(json.dumps(meta, indent=1), encoding='utf-8')
    print(json.dumps(meta), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['validate', 'trace'])
    ap.add_argument('--episode', type=int, default=1100)
    ap.add_argument('--episodes', type=int, default=5)
    a = ap.parse_args()
    if a.action == 'validate':
        validate(a.episode)
    else:
        trace(a.episode, a.episodes)


if __name__ == '__main__':
    main()
