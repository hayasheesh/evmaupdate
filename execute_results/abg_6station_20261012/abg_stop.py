"""ABG 6 station（AEMO・ERCOT）を、テストの大域の報酬が最高値から伸びなくなったら保存して止める。10分ごとに見る。

判定：テストの Global_Reward の100エピソード移動平均（テスト5回分）の最高値を、300エピソード更新していない。
ABG は大域の批評家だけで学習する脇の実験なので、大域の報酬だけを見る。エピソード数の下限は置かない。
止めるときは training.training_resume.request_stop（今のエピソードを終えて保存してから止まる）。
記録は abg_stop.log。2本とも止めたら終わる。
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
import time

import pandas as pd

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
sys.path.insert(0, str(ROOT))
from training.training_resume import request_stop, stop_requested  # noqa: E402

LOG = RUN / 'abg_stop.log'
NAMES = {'aemo': 'wr_obsdrop_cap77wd_aemoplan_6station_abg', 'ercot': 'wr_obsdrop_cap77wd_ercotplan_6station_abg'}


def log(msg):
    with LOG.open('a', encoding='utf-8') as f:
        f.write(f'{datetime.now().isoformat(timespec="seconds")} {msg}\n')


def stalled(d):
    df = pd.read_csv(d / 'results' / 'test_episode_rewards_all.csv')
    ma = df['Global_Reward'].rolling(5).mean()
    last = int(df['Episode'].iloc[-1])
    best_ep = int(df['Episode'][ma.idxmax()])
    return last - best_ep >= 300, last, float(ma.max()), best_ep


def main():
    log('start')
    done = set()
    while len(done) < 2:
        for task, name in NAMES.items():
            if task in done:
                continue
            d = sorted((ROOT / 'archive').glob(f'{name}_2*'))[-1]
            if stop_requested(d):
                done.add(task)
                continue
            try:
                ok, last, best, best_ep = stalled(d)
            except Exception as e:
                log(f'{task} read failed: {e}')
                continue
            if ok:
                request_stop(d)
                done.add(task)
                log(f'{task} stalled at ep{last} (best 100-ep global {best:.3f} at ep{best_ep}); stop requested for {d.name}')
        time.sleep(600)
    log('done')


if __name__ == '__main__':
    main()
