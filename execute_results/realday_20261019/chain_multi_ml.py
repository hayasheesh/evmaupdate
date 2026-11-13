"""いろいろな日で一から学習し（E）、その2000回目から機械学習の水準の予測の世界で学習し直す（F）流れを順に回す（研究室、2026-10-19）。

    python chain_multi_ml.py [--from-multi]

1. multi_bids に30日分の入札（学習用 train_00〜24、中間テスト用 validation_00〜04）がそろうのを待つ（Windows の multi_win.py が写す）
2. build_multi.py で束と prev_multi.pkl を作る（入札はそろっているので作り直さない）
3. 機械学習の水準の予測の学習し直し（ml_<日>、B）が終わるのを待つ（GPU を空けるため）
4. run_rft.py multi --episodes 2000（E）
5. run_rft.py multift_ml_busy / multift_ml_quiet / multift_ml_average --episodes 1000（F、3本並べる）
--from-multi は、入札と束がそろっている前提で 4 から始める。失敗したらそこで止まって CHAIN_FAILED.txt を書く。
記録は logs/chain_multi_ml.log。
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
LOG = HERE / 'logs' / 'chain_multi_ml.log'
NEEDED = [f'train_{i:02d}.pkl' for i in range(25)] + [f'validation_{j:02d}.pkl' for j in range(5)]


def log(msg):
    LOG.parent.mkdir(exist_ok=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(f'{datetime.now().isoformat(timespec="seconds")} {msg}\n')


def run(args, name):
    log(f'start {name}')
    with (HERE / 'logs' / f'chain_{name}.out').open('a', encoding='utf-8') as out:
        code = subprocess.run([sys.executable, '-X', 'utf8', '-u', *args], cwd=HERE, stdout=out, stderr=subprocess.STDOUT).returncode
    log(f'{name} exit {code}')
    if code != 0:
        (HERE / 'CHAIN_FAILED.txt').write_text(f'{datetime.now().isoformat(timespec="seconds")} {name} exit {code}\n', encoding='utf-8')
        raise SystemExit(code)


def ml_running() -> bool:
    out = subprocess.run(['pgrep', '-f', 'realft_worker.py ml_'], capture_output=True, text=True).stdout.strip()
    return bool(out)


def main():
    if '--from-multi' not in sys.argv:
        log('wait for 30 bids')
        while not all((HERE / 'multi_bids' / n).exists() for n in NEEDED):
            time.sleep(120)
        time.sleep(60)  # 最後の写しが書き終わるのを待つ
        run(['build_multi.py', '--parallel', '1'], 'build_multi_banks')
    log('wait for ml_<day> runs to finish')
    while ml_running():
        time.sleep(300)
    run(['run_rft.py', 'multi', '--episodes', '2000'], 'multi')
    log('start multift_ml x3')
    procs = {p: subprocess.Popen([sys.executable, '-X', 'utf8', '-u', 'run_rft.py', f'multift_ml_{p}', '--episodes', '1000'], cwd=HERE)
             for p in ('busy', 'quiet', 'average')}
    for p, proc in procs.items():
        log(f'multift_ml_{p} exit {proc.wait()}')
    log('all done')


if __name__ == '__main__':
    main()
