#!/bin/bash
# 研究室の学習の実行を、学習状態（resume/training_state_*.pth）以外まるごと Windows の archive へ写す。
# 行動器の重み（results/TEST*）、テストの図と表、TensorBoard の記録、コードの控えが入る。研究室の実行は読むだけ。
#   bash lab_full_mirror.sh '<研究室の archive の下の名前のパターン>' [...]
# 見張り（watch.sh）が30分ごとに、研究室で動いている本線の実行を写す。mirror_skip.txt にある名前（失敗した実行など）は写さない。
ROOT=/c/Users/admin/Desktop/EVMALOCALUPDATE
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20"
for PATTERN in "$@"; do
  for RUN in $(timeout 60 $SSH smartgrid-gpu "cd /home/hayashi/workspace/EVMALOCALUPDATE/archive && ls -d $PATTERN 2>/dev/null"); do
    grep -qxF "$RUN" "$(dirname "$0")/mirror_skip.txt" 2>/dev/null && continue
    timeout 1200 $SSH smartgrid-gpu "cd /home/hayashi/workspace/EVMALOCALUPDATE/archive && tar cf - --exclude='*/resume/training_state_*.pth' $RUN" \
      | tar xf - -C "$ROOT/archive" || continue
    printf '%s\n' "研究室PCの実行の写し。学習状態（resume/training_state_ep*.pth）だけ入っていない。" \
      "元: smartgrid-gpu:/home/hayashi/workspace/EVMALOCALUPDATE/archive/$RUN" "更新: $(date '+%Y-%m-%d %H:%M')" > "$ROOT/archive/$RUN/LAB_MIRROR.txt"
    echo "mirrored $RUN"
  done
done
