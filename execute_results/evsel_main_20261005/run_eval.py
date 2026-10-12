"""run_bank.py で作った入札の未見通過率を測る（研究室の本線で動かす）。

    python execute_results/evsel_main_20261005/run_eval.py <設計の指令数> [workers]

未見の指令は、入札の feedback 区分の先頭128本に固定する（--unseen-commands 128）。
設計の指令数が128でも256でも同じ128本になり、前回研究室で測った今の選び方と
下側2＋最多（どちらも設計128本）とも同じ指令になる。
EV は入札時の候補と重ならない seed から3通り、1組の計算は20秒まで。前回と同じ。
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from run_bank import SIGNAL_SET, bank_dir  # noqa: E402


def main() -> None:
    commands = int(sys.argv[1])
    workers = sys.argv[2] if len(sys.argv) > 2 else '9'
    for key in list(os.environ):
        if key.startswith('EVMA_'):
            del os.environ[key]
    os.environ.update({
        'EVMA_NUM_STATIONS': '7',
        'EVMA_MARL_ALGORITHM': 'hybrid',
        'EVMA_ACTIVATION_SIGNAL_SET': SIGNAL_SET,
    })
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['PYTHONUTF8'] = '1'
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    import tools.evaluate_unseen_bid_feasibility as tool

    assert Path(tool.__file__).resolve().parents[1] == ROOT, tool.__file__
    out = ROOT / 'execute_results' / f'bid_unseen_evsel_main_7station_{commands}cmd_low3max'
    sys.argv = ['evaluate_unseen_bid_feasibility.py', '--output-dir', str(out),
                '--bank', f'AEMO={bank_dir(commands)}', '--unseen-commands', '128',
                '--ev-seeds', '3', '--time-limit-s', '20', '--workers', workers]
    tool.main()


if __name__ == '__main__':
    main()
