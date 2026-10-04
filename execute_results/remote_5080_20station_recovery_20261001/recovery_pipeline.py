"""Retry the missing day and build validation bids before GPU learning."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys


RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
OLD_RUN = ROOT / "execute_results/remote_5080_20station_20260930"
TRAIN_BANK = ROOT / "execute_results/bid_banks/train_25_20station_256cmd_3ev_aemo_plan_deviation"
TEST_BANK = ROOT / "execute_results/bid_banks/validation_5_20station_256cmd_3ev_aemo_plan_deviation"
RETRY_SOURCE_SHA256 = "ce96f1a3be8fcdb6cf34f6b6d7c3a1e8858d3e5d85af00cf70fffc3455bfef9c"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save(state: dict) -> None:
    target = RUN / "status.json"
    temporary = RUN / "status.json.tmp"
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)


def preflight() -> dict:
    source = ROOT / "training/blockwise_bid.py"
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if source_hash != RETRY_SOURCE_SHA256:
        raise RuntimeError("The retry bid source differs from the reviewed revision")
    original = json.loads((OLD_RUN / "status.json").read_text(encoding="utf-8"))
    if original.get("stage") != "failed":
        raise RuntimeError("The original pipeline is not stopped")
    manifest = json.loads((TRAIN_BANK / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("complete") or manifest.get("completed_days") != 24:
        raise RuntimeError("Expected 24 completed train dates")
    entries = list(manifest.get("entries") or [])
    if {int(row["index"]) for row in entries} != set(range(25)) - {9}:
        raise RuntimeError("The train bank has unexpected completed dates")
    for row in entries:
        day_dir = TRAIN_BANK / "days" / f'{int(row["index"]):03d}_{row["service_date"]}'
        if not (day_dir / "entry.json").exists() or not (day_dir / "fixed_bid.pkl").exists():
            raise RuntimeError(f'Missing original result for index {row["index"]}')
    failed_day = TRAIN_BANK / "days/009_2024-08-10"
    if (failed_day / "entry.json").exists() or (failed_day / "fixed_bid.pkl").exists():
        raise RuntimeError("The retry date already has a completed result")
    if not (RUN / "previous_failed_day/upper_bid_progress.log").exists():
        raise RuntimeError("The previous failed-day log has not been preserved")
    return {
        "reused_train_days": 24,
        "retry_train_index": 9,
        "retry_train_date": "2024-08-10",
        "retry_bid_source_sha256": source_hash,
        "original_bid_source_sha256": "724deeca27dbdbe0a2d09ced32017a09ad1450759b8be6a842ecccd0c43ea6ef",
        "validation_days": 5,
    }


def launch(label: str, argv: list[str], state: dict, env: dict) -> subprocess.Popen:
    stdout = (RUN / f"{label}.stdout.log").open("ab")
    stderr = (RUN / f"{label}.stderr.log").open("ab")
    try:
        child = subprocess.Popen(
            [sys.executable, "-u", *argv],
            cwd=ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
        )
    finally:
        stdout.close()
        stderr.close()
    state["children"][label] = {"pid": child.pid, "started_at_utc": now()}
    save(state)
    print(f"[recovery] stage={label} pid={child.pid}", flush=True)
    return child


def require_complete(bank: Path, expected: int) -> None:
    manifest = json.loads((bank / "manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("complete") or int(manifest.get("completed_days", -1)) != expected:
        raise RuntimeError(f"Incomplete bank: {bank.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    proof = preflight()
    if args.preflight_only:
        print(json.dumps(proof, indent=2), flush=True)
        return

    lock = RUN / "pipeline.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(f"pid={os.getpid()} started_at_utc={now()}\n")

    state = {
        "stage": "starting",
        "started_at_utc": now(),
        "completed_at_utc": None,
        "error": None,
        "children": {},
        **proof,
    }
    save(state)
    try:
        os.sched_setaffinity(0, set(range(20)))
        state["cpu_affinity"] = sorted(os.sched_getaffinity(0))
        stage = runpy.run_path(str(OLD_RUN / "stage_worker.py"), run_name="recovery_config")
        stage["configure"]()
        env = os.environ.copy()
        state["stage"] = "parallel_banks"
        save(state)
        retry = launch(
            "retry_train_day", [str(RUN / "retry_failed_day.py")], state, env
        )
        validation = launch(
            "validation_bank", [str(OLD_RUN / "stage_worker.py"), "test_bank"],
            state, env,
        )
        for label, child in (("retry_train_day", retry), ("validation_bank", validation)):
            code = child.wait()
            state["children"][label]["exit_code"] = code
            state["children"][label]["completed_at_utc"] = now()
            save(state)
        if retry.returncode or validation.returncode:
            raise RuntimeError("A bid-bank stage failed")

        require_complete(TRAIN_BANK, 25)
        require_complete(TEST_BANK, 5)
        state["stage"] = "pretrain"
        save(state)
        learner = launch(
            "pretrain", [str(OLD_RUN / "stage_worker.py"), "pretrain"], state, env
        )
        code = learner.wait()
        state["children"]["pretrain"]["exit_code"] = code
        state["children"]["pretrain"]["completed_at_utc"] = now()
        if code:
            raise RuntimeError(f"pretrain exited with code {code}")
        state["stage"] = "complete"
    except Exception as exc:
        state["stage"] = "failed"
        state["error"] = repr(exc)
        raise
    finally:
        state["completed_at_utc"] = now()
        save(state)


if __name__ == "__main__":
    main()
