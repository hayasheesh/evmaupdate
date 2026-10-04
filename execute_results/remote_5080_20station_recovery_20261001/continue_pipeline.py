"""Continue the existing validation job while retrying the missing train date."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

import recovery_pipeline as base


RUN = base.RUN
STATE_FILE = RUN / "continuation_status.json"


def save(state: dict) -> None:
    temporary = RUN / "continuation_status.json.tmp"
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATE_FILE)


def launch(label: str, argv: list[str], state: dict, env: dict) -> subprocess.Popen:
    stdout = (RUN / f"{label}.stdout.log").open("ab")
    stderr = (RUN / f"{label}.stderr.log").open("ab")
    try:
        child = subprocess.Popen(
            [sys.executable, "-u", *argv],
            cwd=base.ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
        )
    finally:
        stdout.close()
        stderr.close()
    state["children"][label] = {"pid": child.pid, "started_at_utc": base.now()}
    save(state)
    print(f"[continuation] stage={label} pid={child.pid}", flush=True)
    return child


def wait_for_existing_validation(state: dict) -> None:
    while True:
        original = json.loads((RUN / "status.json").read_text(encoding="utf-8"))
        validation = original.get("children", {}).get("validation_bank", {})
        code = validation.get("exit_code")
        if code is not None:
            state["existing_validation_exit_code"] = int(code)
            save(state)
            if int(code) != 0:
                raise RuntimeError("The existing validation bank failed")
            base.require_complete(base.TEST_BANK, 5)
            return
        time.sleep(30)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    proof = base.preflight()
    original = json.loads((RUN / "status.json").read_text(encoding="utf-8"))
    validation = original.get("children", {}).get("validation_bank", {})
    if not validation.get("pid"):
        raise RuntimeError("The existing validation stage was not started")
    if validation.get("exit_code") not in (None, 0):
        raise RuntimeError("The existing validation stage failed")
    if args.preflight_only:
        print(json.dumps({**proof, "validation_pid": validation["pid"]}, indent=2))
        return

    lock = RUN / "continuation.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(f"pid={os.getpid()} started_at_utc={base.now()}\n")

    state = {
        **proof,
        "stage": "starting",
        "started_at_utc": base.now(),
        "completed_at_utc": None,
        "error": None,
        "existing_validation_pid": validation["pid"],
        "children": {},
    }
    save(state)
    try:
        os.sched_setaffinity(0, set(range(20)))
        state["cpu_affinity"] = sorted(os.sched_getaffinity(0))
        stage = runpy.run_path(
            str(base.OLD_RUN / "stage_worker.py"), run_name="continuation_config"
        )
        stage["configure"]()
        env = os.environ.copy()

        state["stage"] = "retry_train_day"
        save(state)
        retry = launch(
            "retry_train_day_v2", [str(RUN / "retry_failed_day.py")], state, env
        )
        code = retry.wait()
        state["children"]["retry_train_day_v2"]["exit_code"] = code
        state["children"]["retry_train_day_v2"]["completed_at_utc"] = base.now()
        save(state)
        if code:
            raise RuntimeError(f"retry_train_day exited with code {code}")
        base.require_complete(base.TRAIN_BANK, 25)

        state["stage"] = "waiting_for_validation"
        save(state)
        wait_for_existing_validation(state)

        state["stage"] = "pretrain"
        save(state)
        learner = launch(
            "pretrain", [str(base.OLD_RUN / "stage_worker.py"), "pretrain"],
            state, env,
        )
        code = learner.wait()
        state["children"]["pretrain"]["exit_code"] = code
        state["children"]["pretrain"]["completed_at_utc"] = base.now()
        if code:
            raise RuntimeError(f"pretrain exited with code {code}")
        state["stage"] = "complete"
    except Exception as exc:
        state["stage"] = "failed"
        state["error"] = repr(exc)
        raise
    finally:
        state["completed_at_utc"] = base.now()
        save(state)


if __name__ == "__main__":
    main()
