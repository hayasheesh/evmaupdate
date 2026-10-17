#!/bin/bash
# 6 station の PJM の40組を、秒単位の層なし（off）・あり（余力の下限が SoC 0% の plain、出発までに目標へ届く分を残す reach）で評価する（2026-10-17）。
cd "$(dirname "$0")"
declare -A STAGE=([main6]=../resbid_6station_20261012 [pjmrule6]=../pjmrule_6station_20261015)
declare -A RUN=([main6]=../../archive/wr_cap77wd_pjmplan_6station_resbid_tanh_g04l15_20261013_081409
                [pjmrule6]=../../archive/wr_cap77wd_pjmplan_6station_pjmrule75_g04l15_20261015_180534)
declare -A FLAGS=([off]="--fast off" [plain]="--fast on --headroom plain" [reach]="--fast on --headroom reach")
for layer in reach off plain; do
  for bid in main6 pjmrule6; do
    for mode in marl marl_force rule; do
      name=${bid}_${mode}_${layer}
      [ -f results/$name.DONE ] && continue
      python -X utf8 eval_seconds.py --stage-dir ${STAGE[$bid]} --run-dir ${RUN[$bid]} --episode 2000 --mode $mode \
        ${FLAGS[$layer]} --out results/$name.jsonl --workers 20 > results/$name.log 2>&1
      echo "$name exit $? $(wc -l < results/$name.jsonl) sets"
    done
  done
done
