#!/bin/bash
# 公開リポジトリ（https://github.com/hayasheesh/evmaupdate、main）へ今の写しを送る。変更が無ければ何もしない。
# 選ぶ範囲は tools/export_public_snapshot.py の規則（コード・文書・結果の表・実行スクリプト。グラフ、モデル、ログ、入力データは入れない）。
# 失敗したときだけ "publish failed" を出して 1 で終わる。
ROOT=/c/Users/admin/Desktop/EVMALOCALUPDATE
PUB=/c/Users/admin/Desktop/evmaupdate_public
cd $ROOT && python tools/export_public_snapshot.py $PUB >/dev/null 2>&1 || { echo "publish failed: export"; exit 1; }
cd $PUB && git add -A >/dev/null 2>&1
git diff --cached --quiet && exit 0
git commit -q -m "Update the snapshot ($(date '+%Y-%m-%d %H:%M'))

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" >/dev/null 2>&1 || { echo "publish failed: commit"; exit 1; }
timeout 900 git push -q origin main >/dev/null 2>&1 || { echo "publish failed: push"; exit 1; }
