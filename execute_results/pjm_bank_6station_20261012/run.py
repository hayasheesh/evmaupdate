"""PJM（疑似指令 pjm_regd_phase_shift）の 6 station の入札だけを作る（学習はしない）。

table3_6station_20261012 の入札の変種が両方終わる（DONE_aemo と DONE_ercot）のを待ってから、
学習用25日と検証用5日を同時に作る。終わったら DONE、失敗したら FAILED を置く。
"""
import subprocess
import sys
import time
from pathlib import Path

RUN = Path(__file__).resolve().parent
T = RUN.parent / 'table3_6station_20261012'


def spawn(stage, *extra):
    out = open(RUN / 'logs' / f'pjm_{stage}.stdout.log', 'a', encoding='utf-8')
    err = open(RUN / 'logs' / f'pjm_{stage}.stderr.log', 'a', encoding='utf-8')
    return subprocess.Popen([sys.executable, '-X', 'utf8', '-u', str(RUN / 'stage_worker.py'), 'pjm', stage, *extra],
                            cwd=RUN.parents[1], stdout=out, stderr=err, stdin=subprocess.DEVNULL, creationflags=0x08000000)


def main():
    while not ((T / 'DONE_aemo').exists() and (T / 'DONE_ercot').exists()):
        time.sleep(120)
    codes = [p.wait() for p in (spawn('train_bank', '6', '3'), spawn('test_bank', '3', '3'))]
    (RUN / ('DONE' if all(c == 0 for c in codes) else 'FAILED')).touch()


if __name__ == '__main__':
    main()
