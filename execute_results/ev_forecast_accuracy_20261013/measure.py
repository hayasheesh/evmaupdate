"""今の入札器のEVの予測（分布だけ）が、当日のEVをどれだけ当てているかを「%」で表す（6 station の検証用の40組）。

    python measure.py <aemo|ercot>

入札器は、その日の到着の予測から EVEnv を出力0で128回流した実現（ev_candidate_cache に保存）を予測として使う。
当日のEVは table3_6station_20261012/run_table3.py と同じ組（日5 × 指令4 × EVの乱数2）で取り出す。
指標はどれも「予測＝128通りの平均」としたときの当日との差で、精度は 100% − 平均絶対誤差率。
  - 接続台数：5分ごとに充電器につながっているEVの台数（時刻ごとの誤差の和 ÷ 当日の台数の和）
  - 1日の台数：その日に来るEV（0時に停まっている車を含む）の台数
  - 必要な電力量：出発までに目標SoCへ届けるのに要る電力量の1日の合計
あわせて、予測の幅として、128通りのうち当日の値が入る範囲（5〜95%点）に当日が入った割合を出す。
結果は result_<task>.json。
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


def fleet_stats(evs, steps):
    import numpy as np
    connected = np.zeros(steps)
    energy = 0.0
    for ev in evs:
        a, d = int(ev.arrival_t), min(int(ev.departure_t), steps)
        connected[a:d] += 1
        energy += max(float(ev.target_soc) - float(ev.initial_soc), 0.0) * float(ev.capacity_kwh)
    return connected, float(len(evs)), energy


def main():
    import numpy as np
    task = sys.argv[1]
    t3._setup(task)
    from Config import EPISODE_STEPS
    from training import lower_bid_training as lbt
    from training.bid_bank import BidBank

    bank = BidBank(t3._CACHE['test_bank'])
    rows = []
    for day in range(N_DAY):
        base = dict(bank.load_entry(bank.entries[day]))
        seed = int(base.get('forecast_seed', 0))
        count = int(lbt.LOWER_TRAIN_UPPER_BID_EV_SCENARIO_CANDIDATES)
        payloads = [(seed + 10007 * i, base.get('arrival_probabilities_by_station'), base.get('service_date')) for i in range(count)]
        # 保存ファイルの名前には計算した装置が入る。入札データは GPU が見える状態で作ったので、名前だけ cuda で作る。
        import Config
        import torch
        cpu_device = Config.DEVICE
        Config.DEVICE = torch.device('cuda')
        try:
            path = lbt._ev_candidate_cache_path(payloads)
        finally:
            Config.DEVICE = cpu_device
        assert path.exists(), f'no cached candidates for day {day}: {path}'
        cands = pickle.loads(path.read_bytes())
        cstats = [fleet_stats(evs, EPISODE_STEPS) for evs in cands]
        conn = np.array([c[0] for c in cstats])
        n = np.array([c[1] for c in cstats])
        en = np.array([c[2] for c in cstats])
        for c in range(N_CMD):
            for e in range(N_EV):
                a_conn, a_n, a_en = fleet_stats(ib.actual_evs(task, day, c, e), EPISODE_STEPS)
                rows.append({
                    'day': day, 'command': c, 'ev_seed': e, 'service_date': str(base.get('service_date')),
                    'connected_mape_pct': 100 * float(np.abs(conn.mean(0) - a_conn).sum() / max(a_conn.sum(), 1e-9)),
                    'count_ape_pct': 100 * abs(n.mean() - a_n) / max(a_n, 1e-9),
                    'energy_ape_pct': 100 * abs(en.mean() - a_en) / max(a_en, 1e-9),
                    'connected_in_5_95_pct': 100 * float(np.mean((a_conn >= np.percentile(conn, 5, axis=0)) & (a_conn <= np.percentile(conn, 95, axis=0)))),
                    'count_in_5_95': bool(np.percentile(n, 5) <= a_n <= np.percentile(n, 95)),
                    'actual_count': a_n, 'forecast_count_mean': float(n.mean()), 'forecast_count_sd': float(n.std()),
                    'actual_energy_kwh': a_en, 'forecast_energy_mean_kwh': float(en.mean()), 'forecast_energy_sd_kwh': float(en.std()),
                    # 予測どうしのばらつき：128通りの1つを当日とみなしたときの、平均からの誤差（分布の幅そのもの）
                    'self_connected_mape_pct': 100 * float(np.mean(np.abs(conn - conn.mean(0)).sum(1) / np.maximum(conn.sum(1), 1e-9))),
                })
        print(day, 'done', flush=True)
    keys = ['connected_mape_pct', 'count_ape_pct', 'energy_ape_pct', 'connected_in_5_95_pct', 'self_connected_mape_pct']
    summary = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    summary['count_in_5_95_share_pct'] = 100 * float(np.mean([r['count_in_5_95'] for r in rows]))
    out = {'task': task, 'sets': len(rows), 'summary': summary, 'rows': rows}
    (HERE / f'result_{task}.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
