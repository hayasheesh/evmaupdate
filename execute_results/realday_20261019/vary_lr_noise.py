"""学習し直しの lr とノイズを変える（AEMO 6 station、研究室、2026-10-20）。

    python vary_lr_noise.py [--episodes 200]

realft_worker.VARIANTS の変え方（lr を 0.3・3・10 倍、ノイズなし、ノイズ3倍）それぞれで、機械学習の水準の予測の世界の3日
（空いた日、5%の日、混んだ日）を、本線の保存した状態から学習し直す（run_rft.py vary_<変え方>_ml_<日>）。
ほかの設定は B（ml_<日>）と同じ。B の学習し直しが一番効いた日、ほとんど効かなかった日、下がった日を選んだ。
同時に学習するのは2本まで（状態のファイルで数える）。GPU の空きが3 GB 未満なら待つ。
評価は100回ごとに eval_real.py ml_<日> --actors vary_<変え方>_ml_<日>:<回>（1本ずつ）。
記録は logs/vary_lr_noise.log（段の始めと終わり、区切り "milestone <変え方>_<回>"）と vary_eval.log。
失敗したら FAILED_vary_<名前> を書き、その学習の残りは飛ばす。済んだもの（学習の状態が complete、評価の記録に exit 0）は飛ばす。
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import more_days as md  # noqa: E402

LOG = HERE / 'logs' / 'vary_lr_noise.log'
EVAL_LOG = HERE / 'vary_eval.log'
ORDER = ('lrx10', 'lrx3', 'lrx03', 'noise0', 'noisex3')
DAYS = ('quiet', 'q05', 'busy')
SLOTS = 2  # 1本で GPU を約2.55 GB 使う。E のあとの F（3本）と並んでも 16 GB に収まるように


def log(msg, path=LOG):
    md.log(msg, path)


def fail(name):
    (HERE / f'FAILED_vary_{name}').write_text(datetime.now().isoformat(timespec='seconds') + '\n', encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=200)
    args = ap.parse_args()
    import realft_worker as fw
    assert set(ORDER) == set(fw.VARIANTS)
    eps = tuple(range(100, args.episodes + 1, 100))
    modes = [f'vary_{v}_ml_{d}' for v in ORDER for d in DAYS]
    log(f'start modes={len(modes)} episodes={args.episodes}')
    done, bad, failed = set(), set(), set()
    if EVAL_LOG.exists():
        for line in EVAL_LOG.read_text(encoding='utf-8').splitlines():
            parts = line.split()
            if len(parts) == 4 and parts[2:] == ['exit', '0']:
                done.add(parts[1])
    training = {}
    milestones = set()
    while True:
        # 学習の数え方は状態のファイルで行う（起動し直した場合も、前の起動が始めた学習を数える）
        for m, proc in list(training.items()):
            if proc.poll() is not None:
                del training[m]
        states = {m: md.status(m) for m in modes}
        for m, st in states.items():
            if st not in (None, 'running', 'complete') and m not in failed:
                failed.add(m)
                fail(f'train_{m}')
                log(f'train {m} status {st}')
        running = [m for m, st in states.items() if st == 'running']
        for m in modes:
            if len(running) >= SLOTS:
                break
            if states[m] is not None or m in failed or m in training:
                continue
            if md.gpu_free_mib() < md.GPU_FREE_MIB:
                break
            log(f'train {m} start')
            training[m] = subprocess.Popen([sys.executable, '-X', 'utf8', '-u', 'run_rft.py', m, '--episodes', str(args.episodes)],
                                           cwd=HERE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            running.append(m)
            time.sleep(120)
        pending = []
        for m in modes:
            day = m.split('_ml_', 1)[1]
            for ep in eps:
                tag = f'{m}{ep}'
                if tag in done or tag in bad or m in failed:
                    continue
                pending.append(tag)
                if not md.ready(m, ep):
                    continue
                time.sleep(30)
                with (HERE / 'logs' / f'eval_{tag}.log').open('a', encoding='utf-8') as out:
                    code = subprocess.run([sys.executable, '-X', 'utf8', '-u', 'eval_real.py', f'ml_{day}', '--actors', f'{m}:{ep}',
                                           '--workers', '8'], cwd=HERE, stdout=out, stderr=subprocess.STDOUT).returncode
                log(f'{tag} exit {code}', EVAL_LOG)
                pending.remove(tag)
                if code == 0:
                    done.add(tag)
                else:
                    bad.add(tag)
                    fail(tag)
        for v in ORDER:
            for ep in eps:
                tags = [f'vary_{v}_ml_{d}{ep}' for d in DAYS if f'vary_{v}_ml_{d}' not in failed]
                name = f'{v}_{ep}'
                if name not in milestones and tags and all(t in done for t in tags):
                    milestones.add(name)
                    log(f'milestone {name}')
        states = {m: md.status(m) for m in modes}
        if not pending and all(st not in (None, 'running') or m in failed for m, st in states.items()):
            break
        time.sleep(60)
    log(f'all done failed={",".join(sorted(failed)) or "none"} bad_evals={",".join(sorted(bad)) or "none"}')


if __name__ == '__main__':
    main()
