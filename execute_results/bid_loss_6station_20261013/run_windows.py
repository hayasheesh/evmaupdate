"""Windows で、入札時の損の表（6 station）の残りを回す（研究室から10/13 20:30に移した。data/・calib.json・済んだ結果は研究室の写し）。

    python run_windows.py      （start_windows.ps1 が WMI で画面を出さずに起動する）

AEMO と ERCOT を並べて、それぞれ並列7で loss_bid.py run を回す（入札1本が1.0〜2.7 GB。Windows は64 GB）。
済んだ組は飛ばす。記録は logs/run_<task>.log、終わったら DONE を置く。
"""
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(task):
    with (HERE / 'logs' / f'run_{task}.log').open('a', encoding='utf-8') as log:
        subprocess.run([sys.executable, '-X', 'utf8', '-u', str(HERE / 'loss_bid.py'), 'run', task, '7'], cwd=HERE,
                       stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)


with ThreadPoolExecutor(2) as ex:
    list(ex.map(run, ['aemo', 'ercot']))
(HERE / 'DONE').touch()
