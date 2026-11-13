"""realday の finetune の保存点が出たら評価する（Windows と研究室、2026-10-19）。

    python rft_eval_queue.py <windows|lab>

5分ごとに見て、要る行動器の TEST<回>（行動器6本）がそろった仕事から eval_real.py を回す（12並列）。
対象は、機械学習の水準の予測（realday.ALPHA_ML）で作った入札と学習の世界（ml_<日>）。本当の状況は同じ日。
  windows：本線の行動器（2000回）を3つの日で。control の100回ごとを3つの日で（入札は ml_<日>）。
  lab    ：ml_<日> の finetune と、一から（scratch_ml_<日>）の100回ごとを、その日で。
記録は rft_eval_queue_<machine>.log。済んだ仕事（記録に exit 0 がある）は飛ばす。全部済んだら終わる。
"""
from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EPISODES = tuple(range(100, 1001, 100))
PATTERNS = ('busy', 'quiet', 'average')


def ready(name, ep):
    found = sorted((ROOT / 'archive').glob(f'realday_aemo6_{name}_2*'))
    return bool(found) and all((found[-1] / 'results' / f'TEST{ep}' / f'actor_{i}_ep{ep}.pth').exists() for i in range(6))


def jobs(machine):
    """機械学習の水準の予測（ml_<日>）の入札で、本当の状況で測る。
    windows：本線の行動器（A）、control（C）。
    lab    ：学習し直し（B、ml_<日>）、一から（D、scratch_ml_<日>）、いろいろな日で一から（E、multi、500回ごと）、
             E からの学習し直し（F、multift_ml_<日>）。"""
    out = []
    if machine == 'windows':
        for p in PATTERNS:
            out.append(([], [f'ml_{p}', '--actors', 'main:2000'], f'main2000_ml_{p}'))
        for ep in EPISODES:
            for p in PATTERNS:
                out.append(([('control', ep)], [f'ml_{p}', '--actors', f'control:{ep}'], f'control{ep}_ml_{p}'))
    else:
        for ep in EPISODES:
            for p in PATTERNS:
                out.append(([(f'ml_{p}', ep)], [f'ml_{p}', '--actors', f'ml_{p}:{ep}'], f'ml_{p}{ep}'))
                out.append(([(f'multift_ml_{p}', ep)], [f'ml_{p}', '--actors', f'multift_ml_{p}:{ep}'], f'multift_ml_{p}{ep}'))
                out.append(([(f'scratch_ml_{p}', ep)], [f'ml_{p}', '--actors', f'scratch_ml_{p}:{ep}'], f'scratch_ml_{p}{ep}'))
        for ep in (500, 1000, 1500, 2000):
            for p in PATTERNS:
                out.append(([('multi', ep)], [f'ml_{p}', '--actors', f'multi:{ep}'], f'multi{ep}_ml_{p}'))
    return out


def main():
    machine = sys.argv[1]
    log_path = HERE / f'rft_eval_queue_{machine}.log'

    def log(msg):
        with log_path.open('a', encoding='utf-8') as f:
            f.write(f'{datetime.now().isoformat(timespec="seconds")} {msg}\n')

    log('start')
    done = set()
    if log_path.exists():
        for line in log_path.read_text(encoding='utf-8').splitlines():
            parts = line.split()
            if len(parts) == 4 and parts[2:] == ['exit', '0']:
                done.add(parts[1])
    todo = [x for x in jobs(machine) if x[2] not in done]
    kw = {'creationflags': 0x08000000} if os.name == 'nt' else {}
    while todo:
        for item in list(todo):
            need, args, tag = item
            if all(ready(n, e) for n, e in need):
                if need:
                    time.sleep(60)  # 保存の書き込みが終わるのを待つ
                with (HERE / 'logs' / f'eval_{tag}.log').open('a', encoding='utf-8') as out:
                    code = subprocess.run([sys.executable, '-X', 'utf8', '-u', str(HERE / 'eval_real.py'), *args, '--workers', '12'],
                                          cwd=HERE, stdout=out, stderr=subprocess.STDOUT, **kw).returncode
                log(f'{tag} exit {code}')
                if code != 0:
                    (HERE / f'FAILED_{tag}').write_text(datetime.now().isoformat(timespec='seconds') + '\n', encoding='utf-8')
                todo.remove(item)
        if todo:
            time.sleep(300)
    log('all done')


if __name__ == '__main__':
    main()
