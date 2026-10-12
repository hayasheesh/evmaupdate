"""measure.py と同じ40組で、予測の誤差を1 station ごと・15分ごとでも出す（論文の物差しにそろえるため）。

    python measure_station.py <aemo|ercot>

予測は128通りの平均。1 station は充電器10口（MAX_EV_PER_STATION）。
  - 1 station の WAPE：5分ごとの接続台数で、Σ|予測 − 当日| ÷ Σ当日
  - 1 station の占有率の MAE：15分ごとの平均の接続台数を口数で割った占有率の、平均絶対誤差（パーセントポイント）
  - 1 station の接続台数の MAE：15分ごと（台）。あわせて当日の平均の接続台数
  - 6 station 合計の15分ごとの WAPE
結果は result_station_<task>.json。
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Windows では空の文字列だと変数が消え、GPU が見えてしまう
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'info_bid_6station_20261012'))
import info_bid as ib  # noqa: E402

t3 = ib.t3
N_DAY, N_CMD, N_EV = 5, 4, 2


def by_station(evs, steps, n_st):
    import numpy as np
    c = np.zeros((n_st, steps))
    for ev in evs:
        a, d = int(ev.arrival_t), min(int(ev.departure_t), steps)
        c[int(ev.station_id), a:d] += 1
    return c


def main():
    import numpy as np
    task = sys.argv[1]
    t3._setup(task)
    import Config
    import torch
    from Config import EPISODE_STEPS
    from EnvConfig import MAX_EV_PER_STATION
    from training import lower_bid_training as lbt
    from training.bid_bank import BidBank

    n_st = 6
    ports = int(MAX_EV_PER_STATION)
    bank = BidBank(t3._CACHE['test_bank'])
    rows = []
    for day in range(N_DAY):
        base = dict(bank.load_entry(bank.entries[day]))
        seed = int(base.get('forecast_seed', 0))
        count = int(lbt.LOWER_TRAIN_UPPER_BID_EV_SCENARIO_CANDIDATES)
        payloads = [(seed + 10007 * i, base.get('arrival_probabilities_by_station'), base.get('service_date')) for i in range(count)]
        cpu_device = Config.DEVICE
        Config.DEVICE = torch.device('cuda')  # 保存ファイルの名前は入札データを作ったときの装置で決まる
        try:
            path = lbt._ev_candidate_cache_path(payloads)
        finally:
            Config.DEVICE = cpu_device
        cands = pickle.loads(path.read_bytes())
        fc = np.mean([by_station(evs, EPISODE_STEPS, n_st) for evs in cands], axis=0)
        for c in range(N_CMD):
            for e in range(N_EV):
                act = by_station(ib.actual_evs(task, day, c, e), EPISODE_STEPS, n_st)
                f15 = fc.reshape(n_st, -1, 3).mean(2)
                a15 = act.reshape(n_st, -1, 3).mean(2)
                rows.append({
                    'day': day, 'command': c, 'ev_seed': e,
                    'station_wape_pct': [100 * float(np.abs(fc[s] - act[s]).sum() / max(act[s].sum(), 1e-9)) for s in range(n_st)],
                    'station_occ_mae_pp': [100 * float(np.abs(f15[s] - a15[s]).mean()) / ports for s in range(n_st)],
                    'station_count_mae_15min': [float(np.abs(f15[s] - a15[s]).mean()) for s in range(n_st)],
                    'station_mean_count': [float(a15[s].mean()) for s in range(n_st)],
                    'total_wape_15min_pct': 100 * float(np.abs(f15.sum(0) - a15.sum(0)).sum() / max(a15.sum(), 1e-9)),
                })
        print(day, 'done', flush=True)
    arr = lambda k: np.array([r[k] for r in rows])  # noqa: E731
    summary = {
        'station_wape_pct_mean': float(arr('station_wape_pct').mean()),
        'station_wape_pct_by_station': arr('station_wape_pct').mean(0).round(1).tolist(),
        'station_occ_mae_pp_mean': float(arr('station_occ_mae_pp').mean()),
        'station_occ_mae_pp_by_station': arr('station_occ_mae_pp').mean(0).round(2).tolist(),
        'station_count_mae_15min_mean': float(arr('station_count_mae_15min').mean()),
        'station_mean_count_by_station': arr('station_mean_count').mean(0).round(2).tolist(),
        'total_wape_15min_pct': float(arr('total_wape_15min_pct').mean()),
    }
    (HERE / f'result_station_{task}.json').write_text(json.dumps({'task': task, 'summary': summary, 'rows': rows}, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
