"""論文のひな形の図4用：本線と同じ入札の条件のまま、EVの実現の選び方か設計に使う指令の本数だけを変えた検証用の入札（5日）を作る。

    python build_variant_bank.py <aemo|ercot> <selection> <commands> [day_workers] [scenario_workers]

selection は EnvConfig の EVMA_LOWER_TRAIN_UPPER_BID_EV_SCENARIO_SELECTION の値（session_count、low_connection_2_max_count など）。
stage_worker は run_table3.py と同じく TABLE3_SW_DIR で選ぶ（既定は Windows の 6 station）。
出力は本線の検証用の入札の名前の「128cmd_3ev」を「<commands>cmd_<選び方の略>」に替えた bid_banks の下のフォルダ。
EV の候補は EVMA_EV_CANDIDATE_CACHE で本線と共有するので、同じ日の候補128本は本線と同じになる。
"""
from __future__ import annotations

import importlib
import multiprocessing
import os
from pathlib import Path
import sys
import types

# 学習が GPU を使っているので、ここでは CPU だけで動かす。
os.environ['CUDA_VISIBLE_DEVICES'] = ''
HERE = Path(__file__).resolve().parent
SW_PATTERN = os.environ.get('TABLE3_SW_DIR', str(HERE.parent / 'wr_gs_6station_20261011'))
SHORT = {'session_count': 'minmedmax', 'low_connection_2_max_count': 'low2max',
         'low_connection_1_median_max_count': 'low1medmax', 'low_connection_3_max_count': 'low3max'}


def main() -> int:
    task, selection, commands = sys.argv[1], sys.argv[2], int(sys.argv[3])
    day_workers = sys.argv[4] if len(sys.argv) > 4 else '1'
    scenario_workers = sys.argv[5] if len(sys.argv) > 5 else '3'
    multiprocessing.set_start_method('spawn', force=True)
    sw_dir = Path(SW_PATTERN.format(task=task))
    sys.path.insert(0, str(sw_dir))
    import stage_worker
    # configure() は最後に Config を読み込む。入札の条件を上書きしてから本物の Config を読むため、その間だけ空の Config を置く。
    sys.modules['Config'] = types.ModuleType('Config')
    if hasattr(stage_worker, 'banks'):
        stage_worker.configure(task)
        base = stage_worker.banks(task)[1]
    else:
        stage_worker.configure()
        base = stage_worker.TEST_BANK
    del sys.modules['Config']
    assert '_128cmd_3ev_' in base.name, base.name
    out = base.parent / base.name.replace('_128cmd_3ev_', f'_{commands}cmd_{SHORT[selection]}_')
    os.environ['EVMA_LOWER_TRAIN_UPPER_BID_EV_SCENARIO_SELECTION'] = selection
    os.environ['EVMA_LOWER_TRAIN_UPPER_BID_ACTIVATION_SCENARIOS'] = str(commands)
    os.environ['EVMA_LOWER_TRAIN_TEST_BID_BANK_DIR'] = str(out)
    os.environ['EVMA_EV_CANDIDATE_CACHE'] = '1'
    import EnvConfig
    assert EnvConfig.LOWER_TRAIN_UPPER_BID_EV_SCENARIO_SELECTION == selection
    assert EnvConfig.LOWER_TRAIN_UPPER_BID_ACTIVATION_SCENARIOS == commands
    print(f'building {out}', flush=True)
    module = importlib.import_module('tools.build_training_bid_bank')
    sys.argv = ['build_training_bid_bank.py', '--split', 'test', '--days', '5',
                '--train-split-count', '25', '--paired-train-days', '25', '--paired-test-days', '5',
                '--workers', day_workers, '--scenario-workers', scenario_workers, '--output-dir', str(out)]
    return int(module.main() or 0)


if __name__ == '__main__':
    raise SystemExit(main())
