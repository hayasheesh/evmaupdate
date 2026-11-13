"""テストの日を増やす（AEMO 6 station、研究室、2026-10-20）。

    python more_days.py [--episodes 300]

days_more.json の6日（realday.MORE_PATTERNS）それぞれで、機械学習の水準の予測（realday.ALPHA_ML）の世界を使って次を順に回す。
1. build_real.py ml_<日>：その日の入札、学習の束、本当の状況（3本ずつ並べる）
2. run_rft.py ml_<日> --episodes <回>：本線の保存した状態から学習し直す（B）。同時に3本まで。GPU の空きが3 GB 未満なら待つ
3. eval_real.py ml_<日>：本線の行動器（A、2000回）、B の100回ごと、control（C）の同じ回、
   いろいろな日で一から学習した行動器（E、2000回。E の学習が済んでから）。評価は1本ずつ
記録は logs/more_days.log（段の始めと終わり、段の区切り "milestone <名前>"）と more_days_eval.log（"<時刻> <名前> exit <code>"）。
失敗したら FAILED_more_<名前> を書き、その日の残りは飛ばす。済んだ段（build_ml_<日>.json がある、学習の状態が complete、
評価の記録に exit 0 がある）は、起動し直しても飛ばす。
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
LOG = HERE / 'logs' / 'more_days.log'
EVAL_LOG = HERE / 'more_days_eval.log'
BUILD_SLOTS = 3
TRAIN_SLOTS = 3
GPU_FREE_MIB = 3000


def log(msg, path=LOG):
    path.parent.mkdir(exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(f'{datetime.now().isoformat(timespec="seconds")} {msg}\n')


def fail(name):
    (HERE / f'FAILED_more_{name}').write_text(datetime.now().isoformat(timespec='seconds') + '\n', encoding='utf-8')


def ready(name, ep):
    found = sorted((ROOT / 'archive').glob(f'realday_aemo6_{name}_2*'))
    return bool(found) and all((found[-1] / 'results' / f'TEST{ep}' / f'actor_{i}_ep{ep}.pth').exists() for i in range(6))


def gpu_free_mib():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=60).stdout.split(',')
        return int(out[1]) - int(out[0])
    except Exception:  # noqa: BLE001  数えられなければ待つ
        return 0


def status(mode):
    p = HERE / f'{mode}_run_status.json'
    if not p.exists():
        return None
    s = json.loads(p.read_text(encoding='utf-8'))
    if s.get('state') == 'running':
        try:
            os.kill(int(s['pid']), 0)
        except (OSError, KeyError, ValueError):
            return 'lost'
        return 'running'
    return s.get('result')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=300)
    args = ap.parse_args()
    import realday as rd
    days = tuple(rd.MORE_PATTERNS)
    eps = tuple(range(100, args.episodes + 1, 100))
    log(f'start days={",".join(days)} episodes={args.episodes}')
    done = set()
    if EVAL_LOG.exists():
        for line in EVAL_LOG.read_text(encoding='utf-8').splitlines():
            parts = line.split()
            if len(parts) == 4 and parts[2:] == ['exit', '0']:
                done.add(parts[1])
    failed = set()   # 入札か学習が失敗した日
    bad = set()      # 失敗した評価
    building, training = {}, {}
    milestones = set()

    def built(d):
        return (HERE / f'build_ml_{d}.json').exists()

    def evals(d):
        """(名前, 前に要るもの, 行動器)。"""
        out = [(f'more_main2000_ml_{d}', lambda: True, 'main:2000')]
        for ep in eps:
            out.append((f'more_ml_{d}{ep}', lambda ep=ep: ready(f'ml_{d}', ep) and ready('control', ep),
                        f'ml_{d}:{ep},control:{ep}'))
        out.append((f'more_multi2000_ml_{d}', lambda: ready('multi', 2000), 'multi:2000'))
        return out

    def milestone(name, tags):
        if name not in milestones and all(t in done for t in tags):
            milestones.add(name)
            log(f'milestone {name}')

    while True:
        # 1. 入札
        for d, proc in list(building.items()):
            code = proc.poll()
            if code is None:
                continue
            del building[d]
            log(f'build ml_{d} exit {code}')
            if code != 0 or not built(d):
                failed.add(d)
                fail(f'build_ml_{d}')
        for d in days:
            if d in failed or d in building or built(d) or len(building) >= BUILD_SLOTS:
                continue
            log(f'build ml_{d} start')
            with (HERE / 'logs' / f'build_ml_{d}.log').open('a', encoding='utf-8') as out:
                building[d] = subprocess.Popen([sys.executable, '-X', 'utf8', '-u', 'build_real.py', f'ml_{d}', '--workers', '4'],
                                               cwd=HERE, stdout=out, stderr=subprocess.STDOUT)
        # 2. 学習し直し
        for d, proc in list(training.items()):
            code = proc.poll()
            if code is None:
                continue
            del training[d]
            log(f'train ml_{d} exit {code}')
            if code != 0:
                failed.add(d)
                fail(f'train_ml_{d}')
        for d in days:
            if d in failed or d in training or not built(d) or len(training) >= TRAIN_SLOTS:
                continue
            st = status(f'ml_{d}')
            if st in ('complete', 'running'):
                continue
            if st is not None:
                failed.add(d)
                fail(f'train_ml_{d}')
                log(f'train ml_{d} status {st}')
                continue
            if gpu_free_mib() < GPU_FREE_MIB:
                break
            log(f'train ml_{d} start')
            training[d] = subprocess.Popen([sys.executable, '-X', 'utf8', '-u', 'run_rft.py', f'ml_{d}', '--episodes', str(args.episodes)],
                                           cwd=HERE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(120)  # 読み込みで GPU の使用量が上がるのを待ってから次を数える
        # 3. 評価（1本ずつ）
        pending = []
        for d in days:
            if not built(d):
                if d not in failed:
                    pending.append(d)
                continue
            for tag, need, actors in evals(d):
                if tag in done or tag in bad or (d in failed and not tag.startswith(('more_main', 'more_multi'))):
                    continue
                pending.append(tag)
                if not need():
                    continue
                time.sleep(30)  # 保存の書き込みが終わるのを待つ
                with (HERE / 'logs' / f'eval_{tag}.log').open('a', encoding='utf-8') as out:
                    code = subprocess.run([sys.executable, '-X', 'utf8', '-u', 'eval_real.py', f'ml_{d}', '--actors', actors,
                                           '--workers', '8'], cwd=HERE, stdout=out, stderr=subprocess.STDOUT).returncode
                log(f'{tag} exit {code}', EVAL_LOG)
                pending.remove(tag)
                if code == 0:
                    done.add(tag)
                else:
                    bad.add(tag)
                    fail(tag)
        ok = [d for d in days if d not in failed]
        milestone('main2000_all', [f'more_main2000_ml_{d}' for d in ok])
        for ep in eps:
            milestone(f'b{ep}_all', [f'more_ml_{d}{ep}' for d in ok])
        milestone('multi2000_all', [f'more_multi2000_ml_{d}' for d in ok])
        if not building and not training and not pending:
            break
        time.sleep(60)
    log(f'all done failed={",".join(sorted(failed)) or "none"} bad_evals={",".join(sorted(bad)) or "none"}')


if __name__ == '__main__':
    main()
