"""学習し直しの実行の TensorBoard から、行動器に届く勾配の大きさと大域の割合、批評家の損失を、100回ごとの平均で出す（2026-10-20）。

    python grad_series.py <archive の実行フォルダ> [...]

grad_check.py（wr_gs_6station_20261011）と同じ記録の名前を使う。回は学習し直しの回（本線の2000回を引いた値）。
"""
import glob
import sys
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TAGS = {'global_ratio': 'GradHealth/actor_source_global_ratio', 'local_raw': 'GradHealth/actor_source_local_raw',
        'global_raw': 'GradHealth/actor_source_global_raw'}


def series(run):
    data = {k: {} for k in TAGS}
    extra = {}
    for f in sorted(glob.glob(str(Path(run) / 'performance' / 'events*'))):
        ea = EventAccumulator(f, size_guidance={'scalars': 0})
        ea.Reload()
        have = set(ea.Tags()['scalars'])
        for k, t in TAGS.items():
            if t in have:
                for e in ea.Scalars(t):
                    data[k][e.step] = e.value
        for t in have:
            if 'loss' in t.lower() and ('critic' in t.lower() or 'Critic' in t):
                for e in ea.Scalars(t):
                    extra.setdefault(t, {})[e.step] = e.value
    return data, extra


def main():
    for run in sys.argv[1:]:
        data, extra = series(run)
        steps = sorted(data['global_ratio'])
        print(Path(run).name, 'steps', (steps[0], steps[-1]) if steps else None, 'loss tags', sorted(extra)[:4])
        if not steps:
            continue
        base = 2000 if steps[0] > 2000 else 0
        for lo in range(0, int(steps[-1] - base) + 1, 100):
            row = []
            for k in TAGS:
                xs = [v for s, v in data[k].items() if lo < s - base <= lo + 100]
                row.append(f'{k} {np.mean(xs):.3f}' if xs else f'{k} -')
            for t in sorted(extra)[:2]:
                xs = [v for s, v in extra[t].items() if lo < s - base <= lo + 100]
                row.append(f'{t.split("/")[-1]} {np.mean(xs):.4g}' if xs else '')
            print(f'  {lo + 1:4d}-{lo + 100:4d}  ' + '  '.join(row))


if __name__ == '__main__':
    main()
