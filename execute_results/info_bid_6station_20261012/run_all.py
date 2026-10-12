"""info_bid.py を、AEMO と ERCOT を並べて、oracle → oracle_used → cmd → ev の順に回す。各市場の並列数は7。
終わったら DONE を置く。記録は logs/<task>_<mode>.log。
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODES = ['oracle', 'oracle_used', 'cmd', 'ev']


def chain(task):
    for mode in MODES:
        with (HERE / 'logs' / f'{task}_{mode}.log').open('a', encoding='utf-8') as log:
            subprocess.run([sys.executable, '-X', 'utf8', '-u', str(HERE / 'info_bid.py'), task, mode, '7'],
                           cwd=HERE, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)


if __name__ == '__main__':
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(2) as ex:
        list(ex.map(chain, ['aemo', 'ercot']))
    (HERE / 'DONE').touch()
