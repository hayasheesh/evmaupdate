"""論文のひな形の図4用：本線の検証用の入札と、build_variant_bank.py で作った変種の入札の未見通過率を、同じ指令とEVで測る。

    python run_unseen_variants.py <aemo|ercot> <out_dir> <workers> <bank_dir_name> [<bank_dir_name> ...]

stage_worker は run_table3.py と同じく TABLE3_SW_DIR で選び、その configure で局の並びなどを本線と同じにしてから
tools/evaluate_unseen_bid_feasibility.py を呼ぶ。指令は --unseen-commands 128（入札時と同じ区分と seed から先頭128本を
引き直すので、設計の指令数が違う入札どうしでも同じ128本になる）、EVは入札時の候補と重ならない seed の3通り、1組20秒まで。
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
HERE = Path(__file__).resolve().parent
SW_PATTERN = os.environ.get('TABLE3_SW_DIR', str(HERE.parent / 'wr_gs_6station_20261011'))


def main():
    task, out_dir, workers, names = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    sw_dir = Path(SW_PATTERN.format(task=task))
    sys.path.insert(0, str(sw_dir))
    import stage_worker
    if hasattr(stage_worker, 'banks'):
        stage_worker.configure(task)
    else:
        stage_worker.configure()
    import torch  # noqa: F401
    import market.physical_lp_bidding  # noqa: F401
    import training.bid_bank  # noqa: F401
    import tools.evaluate_unseen_bid_feasibility as tool

    banks = sw_dir.parents[1] / 'execute_results' / 'bid_banks'
    argv = ['evaluate_unseen_bid_feasibility.py', '--output-dir', out_dir, '--unseen-commands', '128',
            '--ev-seeds', '3', '--time-limit-s', '20', '--workers', workers]
    for name in names:
        argv += ['--bank', f'{name}={banks / name}']
    sys.argv = argv
    tool.main()


if __name__ == '__main__':
    main()
