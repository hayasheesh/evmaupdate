"""Copy the parts of this project that go to the public GitHub repository.

    python tools/export_public_snapshot.py <target_dir> [--dry-run]

The target is a separate git working tree. Everything in it except ``.git`` is
replaced, so a file that is no longer selected disappears from the next
commit. The working repository and its history are not touched.

Selected:
- code and documents the working repository tracks or would track
  (``git ls-files --cached --others --exclude-standard``);
- from ``archive/`` and ``execute_results/``: result tables (.csv, .json) and
  the scripts directly inside each run folder;
- from ``data/``: the scripts that build the inputs and their metadata, not the
  data (third-party datasets, some without a stated licence).
Left out everywhere: figures, PDFs, slides, spreadsheets, model weights and
checkpoints, replay buffers and pickles, logs, archives, per-run code
snapshots, and any single file above MAX_BYTES.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 5 * 1024 * 1024
EXCLUDED_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".pdf", ".pptx", ".xlsx", ".xls",
    ".pth", ".pt", ".ckpt", ".npz", ".npy", ".pkl", ".pickle", ".h5",
    ".log", ".zip", ".gz", ".tar", ".7z", ".pyc",
}
EXCLUDED_PARTS = {"code_snapshot", "__pycache__", "resume", "runs", "performance", ".codex_tmp"}
RESULT_SUFFIXES = {".csv", ".json"}
SCRIPT_SUFFIXES = {".py", ".sh", ".ps1", ".md", ".txt"}
EXCLUDED_PREFIXES = ("docs/hobo100100/",)


def _excluded(rel: str) -> bool:
    path = Path(rel)
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return True
    if any(part in EXCLUDED_PARTS or part.startswith("events.out.tfevents") for part in path.parts):
        return True
    return rel.startswith(EXCLUDED_PREFIXES)


def tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT, capture_output=True, check=True,
    ).stdout.decode("utf-8")
    return [rel for rel in out.split("\0") if rel and (ROOT / rel).is_file()]


def result_files() -> list[str]:
    selected = []
    for top in ("archive", "execute_results"):
        base = ROOT / top
        if not base.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in EXCLUDED_PARTS]
            rel_dir = Path(dirpath).relative_to(ROOT)
            depth = len(rel_dir.parts)
            for name in filenames:
                rel = (rel_dir / name).as_posix()
                suffix = Path(name).suffix.lower()
                if suffix in RESULT_SUFFIXES or (suffix in SCRIPT_SUFFIXES and depth <= 2):
                    selected.append(rel)
    return selected


def data_scripts() -> list[str]:
    base = ROOT / "data"
    if not base.is_dir():
        return []
    selected = []
    for path in base.rglob("*"):
        if path.is_file() and (path.suffix.lower() == ".py" or path.name == "metadata.json"):
            selected.append(path.relative_to(ROOT).as_posix())
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("target")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    target = Path(args.target).resolve()
    if target == ROOT or ROOT in target.parents:
        raise SystemExit("target must be outside the working repository")

    candidates = sorted(set(tracked_files()) | set(result_files()) | set(data_scripts()))
    chosen, too_large = [], []
    for rel in candidates:
        if _excluded(rel):
            continue
        size = (ROOT / rel).stat().st_size
        if size > MAX_BYTES:
            too_large.append((rel, size))
            continue
        chosen.append((rel, size))
    total = sum(size for _, size in chosen)
    by_top: dict[str, list[int]] = {}
    for rel, size in chosen:
        entry = by_top.setdefault(rel.split("/", 1)[0] if "/" in rel else "(root)", [0, 0])
        entry[0] += 1
        entry[1] += size
    print(json.dumps({
        "files": len(chosen), "total_mb": round(total / 1048576, 1),
        "by_top": {k: {"files": v[0], "mb": round(v[1] / 1048576, 2)} for k, v in sorted(by_top.items())},
        "skipped_over_limit": [(rel, round(size / 1048576, 1)) for rel, size in too_large],
    }, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0

    target.mkdir(parents=True, exist_ok=True)
    for child in target.iterdir():
        if child.name == ".git":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    for rel, _ in chosen:
        if rel == ".gitignore":
            continue
        dest = target / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dest)
    # The working .gitignore leaves out archive/, execute_results/ and data/,
    # which the snapshot deliberately carries in part.
    (target / ".gitignore").write_text("__pycache__/\n*.pyc\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
