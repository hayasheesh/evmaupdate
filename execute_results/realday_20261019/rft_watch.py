"""realday の finetune（Windows の control）の見張り。GPU が熱いときに止める（2026-10-19）。

    python rft_watch.py

GPU の温度は30秒ごとに nvidia-smi で読み、rft_gpu_temperature.csv に残す。止める条件は abg_pjm_watch.py と同じ：
70 °C 以上、直近2分の中央値が60 °C 以上で直近20分の最低から8 °C 以上上がった、直近20分の最低が58 °C 以上。
当てはまったらすべての学習に停止を依頼し（今のエピソードを終えて保存してから止まる）、RFT_GPU_STOPPED.txt を書く。
2分たっても学習が残っていて75 °C 以上なら強制終了する。回数は決めた数まで回すので、伸びの判定では止めない。
記録は rft_watch.log。始まって30分たち、学習のプロセスが残っていなければ終わる。
"""
from __future__ import annotations

from collections import deque
from datetime import datetime
from pathlib import Path
import statistics
import subprocess
import sys
import time

import psutil

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
sys.path.insert(0, str(ROOT))
from training.training_resume import request_stop, stop_requested  # noqa: E402

LOG = RUN / 'rft_watch.log'
TEMP = RUN / 'rft_gpu_temperature.csv'
NAMES = ('realday_aemo6_control',)


def log(msg):
    with LOG.open('a', encoding='utf-8') as f:
        f.write(f'{datetime.now().isoformat(timespec="seconds")} {msg}\n')


def run_dir(name):
    found = sorted((ROOT / 'archive').glob(f'{name}_2*'))
    return found[-1] if found else None


def trainers():
    out = []
    for p in psutil.process_iter(['pid', 'cmdline']):
        cmd = ' '.join(p.info['cmdline'] or [])
        if 'realft_worker.py' in cmd and ' pretrain' in cmd:
            out.append(p)
    return out


def gpu_temp():
    q = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu,utilization.gpu,power.draw', '--format=csv,noheader,nounits'],
                       capture_output=True, text=True, timeout=30).stdout.strip().split(',')
    return float(q[0]), q[1].strip(), q[2].strip()


def main():
    log('start')
    if not TEMP.exists():
        TEMP.write_text('time,gpu_c,util_pct,power_w\n', encoding='utf-8')
    hist = deque(maxlen=40)  # 30秒ごと × 40 = 20分
    hot_since = None
    started = time.time()
    while True:
        try:
            t, util, power = gpu_temp()
        except Exception as e:
            log(f'nvidia-smi failed: {e}')
            t = None
        if t is not None:
            with TEMP.open('a', encoding='utf-8') as f:
                f.write(f'{datetime.now().isoformat(timespec="seconds")},{t},{util},{power}\n')
            hist.append(t)
            recent = list(hist)[-4:]
            reason = None
            if t >= 70:
                reason = f'{t} C >= 70'
            elif len(hist) >= 4 and statistics.median(recent) >= 60 and statistics.median(recent) - min(hist) >= 8:
                reason = f'2-min median {statistics.median(recent)} C, 8 C above the 20-min low {min(hist)}'
            elif len(hist) == hist.maxlen and min(hist) >= 58:
                reason = f'20-min low {min(hist)} C >= 58'
            if reason and hot_since is None:
                hot_since = time.time()
                for name in NAMES:
                    d = run_dir(name)
                    if d is not None and not stop_requested(d):
                        request_stop(d)
                (RUN / 'RFT_GPU_STOPPED.txt').write_text(f'{datetime.now().isoformat(timespec="seconds")} {reason}\n', encoding='utf-8')
                log(f'GPU guard: {reason}; stop requested for both runs')
            if hot_since is not None and time.time() - hot_since > 120 and t >= 75:
                for p in trainers():
                    p.kill()
                    log(f'GPU guard: killed pid {p.pid} at {t} C')
        if time.time() - started > 1800 and not trainers():
            log('no finetune trainer running; done')
            return
        time.sleep(30)


if __name__ == '__main__':
    main()
