"""全部入りの7局の入札（AEMO・ERCOT、学習用25日）の未見通過率を Windows で測る。

    python run_unseen.py <workers>

条件は前回（execute_results/bid_unseen_low2max_7station_20261006）と同じ：入札日25日、bank に保存された
設計外の指令128本、入札時の候補と重ならない seed の EV 3通り、1組20秒まで。結果は
execute_results/bid_unseen_allin_7station_20261007。
"""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
BANKS = ROOT / 'execute_results' / 'bid_banks'
OUT = ROOT / 'execute_results' / 'bid_unseen_allin_7station_20261007'
BANK_LIST = [
    'AEMO=low2max_min500_margin25_train_25_7station_128cmd_3ev_aemo_plan_deviation',
    'ERCOT=low2max_min500_margin25_train_25_7station_128cmd_3ev_ercot_plan_deviation',
]


def main():
    workers = sys.argv[1] if len(sys.argv) > 1 else '20'
    for key in list(os.environ):
        if key.startswith('EVMA_'):
            del os.environ[key]
    os.environ['EVMA_NUM_STATIONS'] = '7'
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['PYTHONUTF8'] = '1'
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    import torch  # noqa: F401
    import market.physical_lp_bidding  # noqa: F401
    import training.bid_bank  # noqa: F401
    import tools.evaluate_unseen_bid_feasibility as tool

    argv = ['evaluate_unseen_bid_feasibility.py', '--output-dir', str(OUT),
            '--ev-seeds', '3', '--time-limit-s', '20', '--workers', workers]
    for bank in BANK_LIST:
        market, name = bank.split('=')
        argv += ['--bank', f'{market}={BANKS / name}']
    sys.argv = argv
    tool.main()


if __name__ == '__main__':
    main()
