"""Windows の 6 station（大域の報酬1/16）の AEMO と ERCOT が収束したら保存して止め、PJM の入札がそろって枠が空いたら
PJM 6 station の学習（同じ設定、stage_worker.py pjm pretrain）を始める。10分ごとに見る。

収束の判定（2026-10-06 に使った決まり）：1000エピソード以上、かつ 追従率＋SoC達成率 のテストの和の
100エピソード移動平均（テスト5回分）の最高値を、300エピソード更新していない。
止めるときは training.training_resume.request_stop（今のエピソードを終えて保存してから止まる）。
PJM の学習は、PJM の入札（DONE）がそろい、AEMO か ERCOT のどちらかが止まってから始める。
記録は converge.log。PJM の学習を始め、2本とも止まったら終わる。
"""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
sys.path.insert(0, str(ROOT))
from training.training_resume import request_stop, stop_requested  # noqa: E402

LOG = RUN / 'converge.log'
NAMES = {'aemo': 'wr_obsdrop_cap77wd_aemoplan_6station_gs00625', 'ercot': 'wr_obsdrop_cap77wd_ercotplan_6station_gs00625'}
STATUS = ROOT / 'execute_results' / 'wr_gs_6station_20261011'


def log(msg):
    with LOG.open('a', encoding='utf-8') as f:
        f.write(f'{datetime.now().isoformat(timespec="seconds")} {msg}\n')


def run_dir(task):
    return sorted((ROOT / 'archive').glob(f'{NAMES[task]}_2*'))[-1]


def converged(d):
    import pandas as pd
    eps = json.loads((d / 'results' / 'test_history.json').read_text(encoding='utf-8'))['episodes']
    df = pd.read_csv(d / 'results' / 'test_performance_metrics.csv')
    n = min(len(df), len(eps))
    df = df.iloc[:n].copy()
    df['ep'] = eps[:n]
    ma = (df['SoC_Hit_Rate_%'] + df['Dispatch_Tracking_Rate_%']).rolling(5).mean()
    last = int(df['ep'].iloc[-1])
    best_ep = int(df['ep'][ma.idxmax()])
    return last >= 1000 and last - best_ep >= 300, last, float(ma.max()), best_ep


def finished(task):
    """学習のプロセスが終わったか（run_status の result が入る）。"""
    try:
        st = json.loads((STATUS / f'{task}_run_status.json').read_text(encoding='utf-8'))
    except Exception:
        return False
    return bool(st.get('result'))


def main():
    pjm_started = False
    log('start')
    while True:
        for task in ('aemo', 'ercot'):
            d = run_dir(task)
            if stop_requested(d) or finished(task):
                continue
            ok, last, best, best_ep = converged(d)
            if ok:
                request_stop(d)
                log(f'{task} converged at ep{last} (best 100-ep sum {best:.1f} at ep{best_ep}); stop requested for {d.name}')
        if not pjm_started and (RUN / 'DONE').exists() and (finished('aemo') or finished('ercot')):
            out = open(RUN / 'logs' / 'pjm_pretrain.stdout.log', 'a', encoding='utf-8')
            err = open(RUN / 'logs' / 'pjm_pretrain.stderr.log', 'a', encoding='utf-8')
            subprocess.Popen([sys.executable, '-X', 'utf8', '-u', str(RUN / 'stage_worker.py'), 'pjm', 'pretrain'],
                             cwd=ROOT, stdout=out, stderr=err, stdin=subprocess.DEVNULL, creationflags=0x08000000)
            pjm_started = True
            (RUN / 'PJM_PRETRAIN_STARTED').touch()
            log('PJM pretrain started')
        if pjm_started and finished('aemo') and finished('ercot'):
            log('done')
            return
        time.sleep(600)


if __name__ == '__main__':
    main()
