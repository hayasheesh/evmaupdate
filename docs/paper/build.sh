#!/bin/bash
# 図を作り直してから、paper.tex を PDF にする（Tectonic、XeLaTeX）。
#   bash docs/paper/build.sh
# Tectonic は C:/Users/admin/tools/tectonic/tectonic.exe（0.16.9）。初回だけパッケージを取りに行く。
set -e
cd "$(dirname "$0")"
PYTHONIOENCODING=utf-8 python make_figs.py
/c/Users/admin/tools/tectonic/tectonic.exe -X compile paper.tex
