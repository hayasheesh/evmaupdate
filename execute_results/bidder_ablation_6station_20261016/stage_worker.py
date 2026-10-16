"""C4（最低入札量と余裕を振る）と、手当てを1つずつ外す比較の評価に使う stage_worker（6 station、AEMO・ERCOT、2026-10-16）。

paper_eval_20261014/run_eval.py の --stage-dir にこのフォルダを渡し、変種は環境変数 BIDDER_VARIANT で選ぶ。
行動器は本線（resbid_6station_20261012）の2000回のものをそのまま使い、入札だけを変種の検証用の入札（5日）に替える。
変種の入札は selection_table_6station_20261015/table.py の build で作る。
  q1・q35・q110・q180   最低入札量 q（kW）。q1 は解く単位の下限で、最低入札量なしにあたる
  m0・m14              アセスメントIの余裕（kW）
  w0                  基準値の段差の罰なし
  sel_random0〜2       EVの実現を無作為に3通り（ほかは本線と同じ）
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BANKS = ROOT / 'execute_results' / 'bid_banks'
_spec = importlib.util.spec_from_file_location('resbid_stage_worker', ROOT / 'execute_results/resbid_6station_20261012/stage_worker.py')
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
TASKS = base.TASKS
_base_banks = base.banks

# table.py の BIDDER と同じ値。table.py は読み込むと sys.path と環境変数を書き換えるので、ここに写しを置く。
BIDDER = {'q1': (1, 7, 0.5), 'q35': (35, 7, 0.5), 'q110': (110, 7, 0.5), 'q180': (180, 7, 0.5),
          'm0': (70, 0, 0.5), 'm14': (70, 14, 0.5), 'w0': (70, 7, 0.0)}
RANDOM = {'sel_random0': 'random0', 'sel_random1': 'random1', 'sel_random2': 'random2'}


def variant() -> str:
    v = os.environ['BIDDER_VARIANT']
    if v not in BIDDER and v not in RANDOM:
        raise ValueError(f'unknown BIDDER_VARIANT {v}')
    return v


def banks(task: str) -> tuple[Path, Path]:
    v = variant()
    signal_set = TASKS[task][0]
    if v in RANDOM:
        name = f'wr_cap77wd_min70_margin7_validation_5_6station_128cmd_{RANDOM[v]}_{signal_set}'
    else:
        q, m, w = BIDDER[v]
        prefix = f'min{q:g}_margin{m:g}' + ('' if w == 0.5 else f'_step{w:g}')
        name = f'wr_cap77wd_{prefix}_validation_5_6station_128cmd_low2max_{signal_set}'
    test = BANKS / name
    if not (test / 'manifest.json').exists():
        raise FileNotFoundError(test)
    return _base_banks(task)[0], test


def configure(task: str):
    q, m, w = BIDDER.get(variant(), (70, 7, 0.5))
    base.FLAGS['EVMA_LOWER_TRAIN_RESEARCH_MINIMUM_BID_QUANTITY_KW'] = f'{q:g}'
    base.FLAGS['EVMA_ASSESSMENT_I_CONTROL_MARGIN_KW'] = f'{m:g}'
    base.FLAGS['EVMA_LOWER_TRAIN_UPPER_BID_BASELINE_STEP_WEIGHT'] = f'{w:g}'
    base.banks = banks  # base.configure が評価用の入札の場所を環境変数に書くときに、変種の入札を使わせる
    return base.configure(task)
