"""resume チェックポイントを保持規則どおりに残す。

本体 (`training_resume.save_training_resume`) は保存のたびに新しい順 2 個だけ
残して他を消す。`--resume-checkpoint-interval 20` にすると、100 の節目も
20 分ほどで消える。本体は resume のソース指紋に入っていて編集できないので、
消される前にハードリンクを別ディレクトリへ張って救い出す。

ハードリンクなので容量は増えない。本体が元の名前を消しても、リンク側から
同じ中身が読める。

保持規則（リンク側に適用）:
  * 100 の倍数の episode は残す
  * それ以外は直近の 100 の節目より後のものだけ残す
  * 最低 2 世代は必ず残す

usage:
  python keep_checkpoints.py <run_dir> --watch 30
  python keep_checkpoints.py <run_dir>            # 1 回だけ
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

STATE_RE = re.compile(r"^training_state_ep(\d+)\.pth$")
COARSE = 100
MIN_GENERATIONS = 2
KEEP_DIRNAME = "kept"


def states(directory: Path) -> dict[int, Path]:
    found: dict[int, Path] = {}
    if not directory.is_dir():
        return found
    for path in directory.glob("training_state_ep*.pth"):
        m = STATE_RE.match(path.name)
        if m is not None:
            found[int(m.group(1))] = path
    return found


def episodes_to_keep(episodes, latest: int) -> set[int]:
    known = {int(e) for e in episodes}
    boundary = (int(latest) // COARSE) * COARSE
    keep = {e for e in known if e % COARSE == 0 or e > boundary}
    keep.add(int(latest))
    for episode in sorted(known, reverse=True):
        if len(keep) >= MIN_GENERATIONS:
            break
        keep.add(episode)
    return keep


def sweep(run_dir: Path) -> None:
    resume_dir = run_dir / "resume"
    keep_dir = resume_dir / KEEP_DIRNAME
    keep_dir.mkdir(parents=True, exist_ok=True)

    live = states(resume_dir)
    kept = states(keep_dir)
    # 1. 本体が消す前にリンクを張る
    for episode, path in sorted(live.items()):
        if episode in kept:
            continue
        target = keep_dir / path.name
        try:
            os.link(path, target)
        except FileExistsError:
            pass
        except OSError as exc:
            print(f"  リンクできない ep{episode}: {exc}", flush=True)
            continue
        print(f"  確保 ep{episode} ({path.stat().st_size / 1024**2:.0f} MB)", flush=True)
        kept[episode] = target

    if not kept:
        return
    # 2. 規則どおりリンクを間引く
    latest = max(kept)
    survive = episodes_to_keep(kept, latest)
    dropped = []
    for episode in sorted(kept):
        if episode in survive:
            continue
        try:
            kept[episode].unlink()
        except OSError as exc:
            print(f"  消せない ep{episode}: {exc}", flush=True)
            continue
        dropped.append(episode)
    if dropped:
        remaining = states(keep_dir)
        total = sum(p.stat().st_size for p in remaining.values()) / 1024**3
        print(f"  間引き {dropped} -> 残り {sorted(remaining)} 合計 {total:.1f} GB",
              flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", default=None)
    parser.add_argument("--watch", type=int, default=0)
    parser.add_argument(
        "--target-file",
        default=None,
        help="run_dir を書いたテキストファイル。スケジューラから使う。",
    )
    args = parser.parse_args(argv)
    if args.run_dir is None and args.target_file is None:
        print("run_dir か --target-file のどちらかが要る", file=sys.stderr)
        return 2
    if args.run_dir is not None:
        run_dir = Path(args.run_dir).expanduser().resolve()
    else:
        # The scheduler runs this without a working directory, so a relative
        # path in the target file resolves against the project root instead.
        target = Path(args.target_file).expanduser().resolve()
        text = target.read_text(encoding="utf-8").strip()
        candidate = Path(text).expanduser()
        run_dir = (
            candidate.resolve()
            if candidate.is_absolute()
            else (Path(__file__).resolve().parents[2] / candidate).resolve()
        )
    if not (run_dir / "resume").is_dir():
        print(f"resume ディレクトリがない: {run_dir}", file=sys.stderr)
        return 1
    while True:
        sweep(run_dir)
        if args.watch <= 0:
            return 0
        time.sleep(args.watch)


if __name__ == "__main__":
    raise SystemExit(main())
