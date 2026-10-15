#!/bin/bash
# paper.tex を PDF にする（Tectonic / XeLaTeX、IEEEtran journal）。
#   bash docs/paper/build.sh
# Tectonic は C:/Users/admin/tools/tectonic/tectonic.exe（0.16.9）。初回だけパッケージを取りに行く。
set -e
cd "$(dirname "$0")"
/c/Users/admin/tools/tectonic/tectonic.exe -X compile --keep-logs build_tsg.tex
cp build_tsg.pdf paper.pdf
