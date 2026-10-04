#!/usr/bin/env bash
# keep_checkpoints.py は resume ディレクトリが未作成だと即終了する。
# 学習開始直後はまだ無いので、出来るまで待ってから渡す。
set -eu
cd "$(dirname "$0")/../.."
RUN="$1"
while [ ! -d "$RUN/resume" ]; do sleep 30; done
exec python execute_results/ft_sweep/keep_checkpoints.py "$RUN" --watch 30
