#!/bin/bash
# B（ml_<日>）と control（C）の20・40・60・80回の行動器を、9日の本当の状況で測る（研究室、2026-10-20）。
# 学習し直しを短い回数で打ち切る場合の成績を見るため。済んだら EVAL_EARLY_DONE.txt を書く。
cd "$(dirname "$0")"
PY=/workspace/.venvs/evma20/bin/python
for d in busy quiet average q05 q20 q35 q65 q80 q95; do
  $PY -X utf8 -u eval_real.py ml_$d --actors ml_$d:20,ml_$d:40,ml_$d:60,ml_$d:80,control:20,control:40,control:60,control:80 --workers 6 \
    >> logs/eval_early_ml_$d.log 2>&1
  echo "$(date -Iseconds) ml_$d exit $?" >> logs/eval_early.log
done
date -Iseconds > EVAL_EARLY_DONE.txt
