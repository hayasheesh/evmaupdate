"""finetune の保存点が出たら評価する（Windows、2026-10-19）。

    python ft_eval_queue.py

5分ごとに見て、要る行動器の TEST<回>（行動器6本）がそろった仕事から eval_day.py を回し、summarize_day.py でまとめる。
  到着台数の世界（count）：本線の行動器（2000回）と、count と control の100回ごと（到着台数の世界・その世界の入札）。
  1000回のあとは、本線の世界・本線の入札でも count・control を測る（その日に合わせたことで、ほかの日の性能が落ちたかを見る）。
記録は ft_eval_queue.log。済んだ仕事（記録に exit 0 がある）は飛ばす。全部済んだら終わる。
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LOG = HERE / 'ft_eval_queue.log'
EPISODES = tuple(range(100, 1001, 100))
CREATE_NO_WINDOW = 0x08000000


def log(msg):
    with LOG.open('a', encoding='utf-8') as f:
        f.write(f'{datetime.now().isoformat(timespec="seconds")} {msg}\n')


def ready(name, ep):
    found = sorted((ROOT / 'archive').glob(f'ftday_aemo6_20240201_{name}_2*'))
    return bool(found) and all((found[-1] / 'results' / f'TEST{ep}' / f'actor_{i}_ep{ep}.pth').exists() for i in range(6))


def run(args, tag):
    with (HERE / 'logs' / f'eval_{tag}.log').open('a', encoding='utf-8') as out:
        code = subprocess.run([sys.executable, '-X', 'utf8', '-u', str(HERE / 'eval_day.py'), *args, '--workers', '12'],
                              cwd=HERE, stdout=out, stderr=subprocess.STDOUT, creationflags=CREATE_NO_WINDOW).returncode
    log(f'{tag} exit {code}')
    subprocess.run([sys.executable, '-X', 'utf8', str(HERE / 'summarize_day.py')], cwd=HERE, capture_output=True,
                   creationflags=CREATE_NO_WINDOW)
    return code


def jobs():
    out = [([], ['--actors', 'main:2000', '--world', 'count', '--bid', 'count'], 'main2000_count')]
    for ep in EPISODES:
        out.append(([('count', ep), ('control', ep)],
                    ['--actors', f'count:{ep},control:{ep}', '--world', 'count', '--bid', 'count'], f'ft{ep}_count'))
    out.append(([('count', 1000), ('control', 1000)],
                ['--actors', 'count:1000,control:1000', '--world', 'main', '--bid', 'main'], 'ft1000_mainmain'))
    return out


def main():
    log('start')
    done = set()
    if LOG.exists():
        for line in LOG.read_text(encoding='utf-8').splitlines():
            parts = line.split()
            if len(parts) == 4 and parts[2:] == ['exit', '0']:
                done.add(parts[1])
    todo = [x for x in jobs() if x[2] not in done]
    while todo:
        for item in list(todo):
            need, args, tag = item
            if all(ready(name, ep) for name, ep in need):
                time.sleep(60)  # 保存の書き込みが終わるのを待つ
                if run(args, tag) != 0:
                    (HERE / f'FAILED_{tag}').write_text(datetime.now().isoformat(timespec='seconds') + '\n', encoding='utf-8')
                todo.remove(item)
        if todo:
            time.sleep(300)
    log('all done')


if __name__ == '__main__':
    main()
