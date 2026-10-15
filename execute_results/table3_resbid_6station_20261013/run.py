"""収束した本線（resbid の設定）の 6 station（既定は AEMO、ep2000）を、表3と同じ40組で評価する。

    python run.py [task] [episode]      （WMI で画面を出さずに起動する）

table3_6station_20261012/run_table3.py をそのまま使い、読み込む設定と出力先と保存点だけを環境変数で替える。
  TABLE3_SW_DIR   resbid_6station_20261012（本線の stage_worker。観測に全体のずれと入札の先読みが入る）
  TABLE3_EPISODE  2000（学習の最後の保存点）
  TABLE3_BLOCKS   1（コマごとの約定量と合否も書く。帯の幅ごとの合格率のグラフに使う）
marl（行動器だけ）と marl_force（後ろに強制充電を重ねる）を順に回す。蓄電池は使わない。
結果はこのフォルダの result_<task>_<mode>_4cmd_2ev.jsonl、記録は logs/、終わったら DONE_<task> を置く（AEMO の最初の回は DONE）。
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASK = sys.argv[1] if len(sys.argv) > 1 else 'aemo'
EPISODE = sys.argv[2] if len(sys.argv) > 2 else '2000'
T3 = HERE.parent / 'table3_6station_20261012' / 'run_table3.py'
env = dict(os.environ, TABLE3_SW_DIR=str(HERE.parent / 'resbid_6station_20261012'), TABLE3_OUT_DIR=str(HERE),
           TABLE3_EPISODE=EPISODE, TABLE3_BLOCKS='1', CUDA_VISIBLE_DEVICES='-1')
for mode in ('marl', 'marl_force'):
    with open(HERE / 'logs' / f'{TASK}_{mode}.log', 'a', encoding='utf-8') as log:
        subprocess.run([sys.executable, '-X', 'utf8', '-u', str(T3), TASK, mode, '4', '2', '12'], cwd=T3.parent, env=env,
                       stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
(HERE / f'DONE_{TASK}').touch()
