"""表3の6 stationの marl（ep1100）と rule を、コマごとの約定量と合否（TABLE3_BLOCKS=1）付きで blocks6/ に測り直す。
AEMO と ERCOT を並べ、それぞれ marl → rule の順。並列数は各4。終わったら blocks6/DONE を置く。
"""
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / 'blocks6'
OUT.mkdir(exist_ok=True)
env = dict(os.environ, TABLE3_BLOCKS='1', TABLE3_EPISODE='1100', TABLE3_OUT_DIR=str(OUT))


def chain(task):
    for mode in ('marl', 'rule'):
        with (OUT / f'{task}_{mode}.log').open('a', encoding='utf-8') as log:
            subprocess.run([sys.executable, '-X', 'utf8', '-u', str(HERE / 'run_table3.py'), task, mode, '4', '2', '4'],
                           cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)


with ThreadPoolExecutor(2) as ex:
    list(ex.map(chain, ['aemo', 'ercot']))
(OUT / 'DONE').touch()
