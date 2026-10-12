"""学習の記録（TensorBoard）から、行動器に届く勾配の大域の割合と大きさを、同じ時期の docs/hobo100100 と比べる。

    python grad_check.py <archive の実行フォルダ> [--json]

直近100エピソードの平均を出す。hobo100100 の値は200エピソードごとの平均（下の表、同じ記録から求めた）。
明らかにおかしいとみなす目安（一行の末尾に OFF を付ける）:
  - 大域の割合が 0.40〜0.75 の外（hobo100100 は全期間 0.52〜0.62）
  - 大域か局所の勾配の大きさが、同じ時期の hobo100100 の3倍より大きいか1/3より小さい
"""
import glob
import json
import sys

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# hobo100100（docs/hobo100100/performance）の200エピソードごとの平均: (大域の割合, 局所の大きさ, 大域の大きさ, 余弦)
HOBO = {0: (0.623, 2.30, 4.83, -0.25), 200: (0.608, 2.93, 5.33, -0.44), 400: (0.546, 3.76, 5.18, -0.54),
        600: (0.522, 5.19, 6.24, -0.66), 800: (0.530, 5.85, 7.17, -0.64), 1000: (0.540, 5.78, 7.44, -0.56),
        1200: (0.560, 6.40, 8.97, -0.44), 1400: (0.579, 7.49, 11.34, -0.33), 1600: (0.586, 9.42, 14.84, -0.29),
        1800: (0.584, 10.32, 16.02, -0.28)}
TAGS = {'ratio': 'GradHealth/actor_source_global_ratio', 'local': 'GradHealth/actor_source_local_raw',
        'global': 'GradHealth/actor_source_global_raw', 'cos': 'GradHealth/actor_source_local_global_cos',
        'soc': 'Metrics/soc_hit_rate', 'track': 'RawActor/tracking_success_rate'}


def main():
    run = sys.argv[1]
    data = {k: {} for k in TAGS}
    for f in sorted(glob.glob(run + '/performance/events*')):
        ea = EventAccumulator(f, size_guidance={'scalars': 0})
        ea.Reload()
        have = set(ea.Tags()['scalars'])
        for k, t in TAGS.items():
            if t in have:
                for e in ea.Scalars(t):
                    data[k][e.step] = e.value
    steps = sorted(data['ratio'])
    if not steps:
        print(json.dumps({'run': run, 'episode': None}))
        return
    last = steps[-1]
    win = {k: [v for s, v in d.items() if s > last - 100] for k, d in data.items()}
    m = {k: float(np.mean(v)) if v else float('nan') for k, v in win.items()}
    ref = HOBO[min(max((last // 200) * 200, 0), 1800)]
    off = []
    if not 0.40 <= m['ratio'] <= 0.75:
        off.append('ratio')
    for name, i in (('local', 1), ('global', 2)):
        q = m[name] / ref[i]
        if q > 3 or q < 1 / 3:
            off.append(f'{name}x{q:.1f}')
    out = {'run': run.rstrip('/\\').split('\\')[-1].split('/')[-1], 'episode': last, 'global_ratio': round(m['ratio'], 3),
           'local_raw': round(m['local'], 2), 'global_raw': round(m['global'], 2), 'cos': round(m['cos'], 3),
           'train_soc': round(m['soc'], 1), 'train_track': round(m['track'], 1),
           'hobo_ratio': ref[0], 'hobo_local': ref[1], 'hobo_global': ref[2], 'off': off}
    if '--json' in sys.argv:
        print(json.dumps(out, ensure_ascii=False))
    else:
        print(f"{out['run']} ep{last}: ratio {out['global_ratio']} (hobo {ref[0]}) local {out['local_raw']} (hobo {ref[1]}) "
              f"global {out['global_raw']} (hobo {ref[2]}) cos {out['cos']} soc/track {out['train_soc']}/{out['train_track']}"
              + (' OFF:' + ','.join(off) if off else ''))


if __name__ == '__main__':
    main()
