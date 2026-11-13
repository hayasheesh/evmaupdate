"""いろいろな日の入札（build_multi.py --one）の一部を Windows で作り、できたものから研究室の multi_bids へ写す（2026-10-19）。

    python multi_win.py [--parallel 10]

研究室は学習で混んでいて、1日分の候補128通りに1時間以上かかるため。作るのは学習用の5〜24日目と中間テスト用の0〜4日目
（研究室が先に始めた学習用の0〜4日目は作らない）。研究室の build_multi.py はファイルがあれば飛ばすので、写したものは作り直さない。
組んだ日（multi_days.json）は研究室で選んだものを写して使う。記録は logs/multi_win.log。
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
LAB = 'smartgrid-gpu:/home/hayashi/workspace/EVMALOCALUPDATE/execute_results/realday_20261019/multi_bids/'
LOG = HERE / 'logs' / 'multi_win.log'


def log(msg):
    with LOG.open('a', encoding='utf-8') as f:
        f.write(f'{datetime.now().isoformat(timespec="seconds")} {msg}\n')


def run(job):
    split, index = job
    name = f'{split}_{index:02d}.pkl'
    out = HERE / 'multi_bids' / name
    if not out.exists():
        with (HERE / 'logs' / f'multi_{split}_{index:02d}.log').open('a', encoding='utf-8') as f:
            code = subprocess.run([sys.executable, '-X', 'utf8', '-u', str(HERE / 'build_multi.py'), '--one', split, str(index)],
                                  cwd=HERE, stdout=f, stderr=subprocess.STDOUT, creationflags=0x08000000).returncode
        if code != 0:
            log(f'{name} build exit {code}')
            return job, code
    code = subprocess.run(['scp', '-o', 'BatchMode=yes', '-q', str(out), LAB], capture_output=True,
                          creationflags=0x08000000).returncode
    log(f'{name} built and copied (scp exit {code})')
    return job, code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parallel', type=int, default=10)
    args = ap.parse_args()
    (HERE / 'logs').mkdir(exist_ok=True)
    (HERE / 'multi_bids').mkdir(exist_ok=True)
    jobs = [('train', i) for i in range(5, 25)] + [('validation', j) for j in range(5)]
    log(f'start {len(jobs)} jobs')
    with ThreadPoolExecutor(args.parallel) as ex:
        for job, code in ex.map(run, jobs):
            pass
    log('all done')


if __name__ == '__main__':
    main()
