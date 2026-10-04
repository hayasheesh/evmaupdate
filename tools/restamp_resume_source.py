"""Re-stamp a resume manifest's source fingerprint after a proven-neutral edit.

The resume guard hashes the training sources and refuses a resume when any of
them changed. That is the right default: almost every edit to those files moves
the experiment, and a run that silently continued across one would be two runs
reported as one.

It is a proxy, though. The question the guard exists to ask is whether the
experiment changed, and it answers a narrower one -- whether the bytes did. An
edit that provably produces the same numbers fails the byte test and passes the
real one. Re-stamping is how that case is recorded rather than worked around:
the old fingerprint is kept, the reason is written into the run's event log, and
the manifest says who decided.

Re-stamp only with evidence. "It should not matter" is not evidence; a digest
over the diagnostics and the weights, taken from the same state under both
versions and compared, is. Pass that evidence as --reason so the run carries it.

The fingerprint lives in two places -- the manifest and the state payload it
describes -- and the loader checks that they agree, so both have to move. The
payload is rewritten under a new name and the manifest is pointed at it; the
original stays on disk, which is what makes the whole operation reversible.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _edited_since_head(tracked_paths) -> list[str]:
    """Which of the fingerprinted sources differ from the committed copy."""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD", "--", *tracked_paths],
            cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    return sorted(line for line in result.stdout.splitlines() if line.strip())


def _rewrite_payload(resume_dir: Path, manifest: dict, context: dict):
    """Copy the state file with the new context in it; leave the original alone.

    The loader compares the context inside the payload against the manifest's,
    so re-stamping one without the other trades a source-fingerprint refusal for
    a context-mismatch refusal. Tensors round-trip through torch.save/load
    unchanged, so the state this writes is the state that was saved.
    """
    import hashlib

    import torch

    source = resume_dir / manifest["state_file"]
    if not source.is_file():
        raise FileNotFoundError(f"resume state file is missing: {source}")

    stem = source.stem.split(".restamped")[0]
    target = resume_dir / f"{stem}.restamped.pth"

    print(f"\nrewriting payload context: {source.name} -> {target.name}")
    state = torch.load(source, map_location="cpu", weights_only=False)
    state["context"] = context
    tmp = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    torch.save(state, tmp)
    os.replace(tmp, target)
    del state

    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    size = target.stat().st_size
    print(f"  {size / (1024 * 1024):.1f} MiB  sha256 {digest.hexdigest()[:16]}...")
    return target.name, int(size), digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dir", help="Pretrain run directory holding resume/latest.json.")
    parser.add_argument(
        "--reason",
        required=True,
        help="Why the changed sources produce the same run. Name the evidence.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the change. Without it the differences are only listed.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    from training.training_resume import (
        RESUME_MANIFEST,
        _PRETRAIN_SOURCE_FILES,
        _fingerprint_files,
        _resume_dir,
    )

    resume_dir = _resume_dir(args.run_dir)
    manifest_path = resume_dir / RESUME_MANIFEST
    if not manifest_path.is_file():
        print(f"no resume manifest under {resume_dir}", file=sys.stderr)
        return 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    saved = manifest["context"]["source"]
    current = _fingerprint_files(PROJECT_ROOT, _PRETRAIN_SOURCE_FILES)

    if saved["sha256"] == current["sha256"]:
        print("sources already match the manifest; nothing to re-stamp")
        return 0

    # The manifest keeps one digest over all the files rather than one per file,
    # so it cannot say which of them moved. Ask git instead: the files are
    # tracked, and what differs from the committed copy is what was edited.
    changed = _edited_since_head(current["files"])

    print(f"manifest : {manifest_path}")
    print(f"episode  : {manifest.get('completed_training_episode')}")
    print(f"was      : {saved['sha256']}")
    print(f"now      : {current['sha256']}")
    print("edited relative to HEAD:")
    for path in changed:
        print(f"    {path}")
    if not changed:
        print("    (none -- the manifest predates the current commit)")

    if not args.apply:
        print("\nlisting only; pass --apply to write it")
        return 0

    backup = manifest_path.with_name(f"{manifest_path.stem}.pre-restamp.json")
    if not backup.exists():
        shutil.copy2(manifest_path, backup)

    manifest["context"]["source"] = current

    state_name, state_bytes, state_sha = _rewrite_payload(
        resume_dir, manifest, manifest["context"]
    )
    manifest["state_file"] = state_name
    manifest["state_bytes"] = state_bytes
    manifest["state_sha256"] = state_sha

    tmp = manifest_path.with_name(f".{manifest_path.name}.tmp-{os.getpid()}")
    tmp.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, manifest_path)

    event = {
        "event": "source_fingerprint_restamped",
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_training_episode": manifest.get("completed_training_episode"),
        "changed_files": changed,
        "previous_sha256": saved["sha256"],
        "current_sha256": current["sha256"],
        "reason": args.reason,
        "backup": backup.name,
    }
    with (resume_dir / "events.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")

    print(f"\nre-stamped. previous manifest kept at {backup.name}")
    print("recorded in events.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
