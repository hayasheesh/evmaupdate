"""いろいろな日で学習する流れを順に回す（研究室、2026-10-19）。

    python chain_multi.py

1. build_multi.py（組んだ日30日の入札と束、3日ずつ並べる）
2. run_rft.py multi --episodes 2000（いろいろな日で一から学習）
3. run_rft.py multift_busy / multift_quiet / multift_average --episodes 1000（2の2000回目から、その日で finetune、3本並べる）
どこかで失敗したら、そこで止まって CHAIN_FAILED.txt を書く。記録は logs/chain_multi.log。
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
LOG = HERE / 'logs' / 'chain_multi.log'


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


def main():
    run(['build_multi.py', '--parallel', '3'], 'build_multi')
    run(['run_rft.py', 'multi', '--episodes', '2000'], 'multi')
    log('start multift x3')
    procs = {p: subprocess.Popen([sys.executable, '-X', 'utf8', '-u', 'run_rft.py', f'multift_{p}', '--episodes', '1000'], cwd=HERE)
             for p in ('busy', 'quiet', 'average')}
    for p, proc in procs.items():
        code = proc.wait()
        log(f'multift_{p} exit {code}')
    log('all done')


if __name__ == '__main__':
    main()
