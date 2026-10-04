"""Exact pause/resume support for long lower-MARL pretraining runs.

The actor checkpoints under ``results/TEST*`` are intentionally optimized for
evaluation and offline-to-online fine-tuning.  They do not contain enough
state to claim that a stopped pretrain is the same experiment after restart.
This module owns the stricter, rotating training-state checkpoint used by
``pre_train.py --resume-run``:

* online and target networks plus optimizer state (provided by the agent),
* the complete replay buffer in its physical circular layout,
* Python / NumPy / Torch / CUDA RNG state,
* learned-episode and environment-episode cursors,
* reward/performance histories, and
* immutable run context fingerprints for source code and bid banks.

The latest complete checkpoint is selected through a small JSON manifest.  A
checkpoint is first written to a temporary file, atomically renamed, hashed,
and only then published in the manifest, so an interrupted write cannot be
mistaken for a resumable state.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


RESUME_KEEP_GENERATIONS = 2
RESUME_SCHEMA_VERSION = 1
RESUME_DIRNAME = "resume"
RESUME_MANIFEST = "latest.json"
STOP_REQUEST_FILENAME = "STOP_REQUESTED"
RESUME_EVENT_LOG = "events.jsonl"


class ResumeStateError(RuntimeError):
    """Raised when a run cannot be resumed without changing the experiment."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint_files(root: Path, relative_paths: Iterable[Path]) -> dict[str, Any]:
    digest = hashlib.sha256()
    files: list[str] = []
    for relative in sorted((Path(p) for p in relative_paths), key=lambda p: p.as_posix()):
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"resume fingerprint input does not exist: {path}")
        rel_text = relative.as_posix()
        files.append(rel_text)
        digest.update(rel_text.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return {"sha256": digest.hexdigest(), "files": files}


def fingerprint_bid_bank(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Hash the persisted inputs that determine one fixed-bid bank."""

    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"bid bank does not exist: {root}")
    relative = [
        item.relative_to(root)
        for item in root.rglob("*")
        if item.is_file() and item.suffix.lower() in {".json", ".pkl", ".pickle", ".npz"}
    ]
    if not relative:
        raise ResumeStateError(f"bid bank has no fingerprintable artifacts: {root}")
    result = _fingerprint_files(root, relative)
    result["path"] = str(root)
    return result


_PRETRAIN_SOURCE_FILES = (
    Path("Config.py"),
    Path("EnvConfig.py"),
    Path("pre_train.py"),
    Path("environment/EVEnv.py"),
    Path("environment/central_residual_allocator.py"),
    Path("environment/normalize.py"),
    Path("environment/observation_config.py"),
    Path("training/Agent/maddpg.py"),
    Path("training/Agent/standard_maddpg.py"),
    # The networks themselves. Without these a run could be resumed into a
    # different actor or critic architecture and the guard would not notice,
    # which is exactly what the pending observation and mixer changes would do.
    Path("training/Agent/actor.py"),
    Path("training/Agent/critic.py"),
    Path("training/Agent/noise.py"),
    Path("training/Agent/replay_buffer.py"),
    Path("training/lower_bid_training.py"),
    Path("training/run_after_day_ahead_bid.py"),
    Path("training/train.py"),
    Path("training/training_resume.py"),
)


def _json_stable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_json_stable(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _json_stable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    raise TypeError(type(value).__name__)


def _runtime_module_signature(module_name: str) -> dict[str, Any]:
    module = importlib.import_module(module_name)
    result: dict[str, Any] = {}
    for name in sorted(dir(module)):
        if not name.isupper():
            continue
        try:
            result[name] = _json_stable(getattr(module, name))
        except TypeError:
            continue
    return result


def build_pretrain_resume_context(
    *,
    project_root: str | os.PathLike[str],
    model_name: str,
    forecast_seed: int,
    train_split_count: int,
    bid_bank_dir: str | os.PathLike[str],
    test_bid_bank_dir: str | os.PathLike[str],
    observation_normalization_profile: dict[str, Any],
) -> dict[str, Any]:
    """Build the immutable identity checked before an exact resume."""

    root = Path(project_root).expanduser().resolve()
    return {
        "kind": "lower_marl_bid_bank_pretrain",
        "model_name": str(model_name),
        "forecast_seed": int(forecast_seed),
        "train_split_count": int(train_split_count),
        "train_bid_bank": fingerprint_bid_bank(bid_bank_dir),
        "test_bid_bank": fingerprint_bid_bank(test_bid_bank_dir),
        "observation_normalization_profile": observation_normalization_profile,
        "runtime": {
            "Config": _runtime_module_signature("Config"),
            "EnvConfig": _runtime_module_signature("EnvConfig"),
            "observation_config": _runtime_module_signature(
                "environment.observation_config"
            ),
        },
        "source": _fingerprint_files(root, _PRETRAIN_SOURCE_FILES),
    }


def capture_rng_state() -> dict[str, Any]:
    """Capture every global RNG stream used by training and EVEnv."""

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore RNG streams after model/replay construction has consumed RNG."""

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise ResumeStateError(
                "resume state contains CUDA RNG state but CUDA is unavailable"
            )
        torch.cuda.set_rng_state_all([item.cpu() for item in cuda_state])


def _resume_dir(run_dir: str | os.PathLike[str]) -> Path:
    return Path(run_dir).expanduser().resolve() / RESUME_DIRNAME


def stop_request_path(run_dir: str | os.PathLike[str]) -> Path:
    return _resume_dir(run_dir) / STOP_REQUEST_FILENAME


def request_stop(run_dir: str | os.PathLike[str]) -> Path:
    """Ask a running trainer to checkpoint and stop after its current episode."""

    run = Path(run_dir).expanduser().resolve()
    if not run.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {run}")
    path = stop_request_path(run)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(f"requested_at_utc={_utc_now()}\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def stop_requested(run_dir: str | os.PathLike[str]) -> bool:
    return stop_request_path(run_dir).is_file()


def clear_stop_request(run_dir: str | os.PathLike[str]) -> None:
    try:
        stop_request_path(run_dir).unlink()
    except FileNotFoundError:
        pass


def _append_event(run_dir: Path, payload: dict[str, Any]) -> None:
    event_path = run_dir / RESUME_DIRNAME / RESUME_EVENT_LOG
    event_path.parent.mkdir(parents=True, exist_ok=True)
    with event_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def read_resume_manifest(run_dir: str | os.PathLike[str]) -> dict[str, Any]:
    path = _resume_dir(run_dir) / RESUME_MANIFEST
    if not path.is_file():
        raise ResumeStateError(
            f"no exact resume manifest found under {Path(run_dir).resolve()}; "
            "TEST* checkpoints are warm-start artifacts, not exact pretrain resume states"
        )
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if int(manifest.get("schema_version", -1)) != RESUME_SCHEMA_VERSION:
        raise ResumeStateError(
            f"unsupported resume schema {manifest.get('schema_version')!r}; "
            f"expected {RESUME_SCHEMA_VERSION}"
        )
    if not bool(manifest.get("exact", False)):
        raise ResumeStateError("resume manifest is not marked exact")
    return manifest


def _context_mismatches(saved: Any, current: Any, prefix: str = "context") -> list[str]:
    if isinstance(saved, dict) and isinstance(current, dict):
        mismatches: list[str] = []
        for key in sorted(set(saved) | set(current)):
            child = f"{prefix}.{key}"
            if key not in saved:
                mismatches.append(f"{child}: missing from saved context")
            elif key not in current:
                mismatches.append(f"{child}: missing from current context")
            else:
                mismatches.extend(_context_mismatches(saved[key], current[key], child))
        return mismatches
    if saved != current:
        return [f"{prefix}: saved={saved!r} current={current!r}"]
    return []


def save_training_resume(
    *,
    run_dir: str | os.PathLike[str],
    agent: Any,
    completed_training_episode: int,
    completed_environment_episodes: int,
    all_rewards: list[Any],
    all_local_rewards: list[Any],
    all_global_rewards: list[Any],
    performance_metrics: dict[str, Any],
    all_episode_data: dict[Any, Any],
    context: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    """Atomically publish a rotating, exact training-state checkpoint."""

    if not hasattr(agent, "training_resume_state_dict"):
        raise ResumeStateError(
            f"agent {type(agent).__name__} does not implement exact resume state"
        )
    episode = int(completed_training_episode)
    environment_episodes = int(completed_environment_episodes)
    if episode < 0 or environment_episodes < episode:
        raise ValueError(
            "invalid resume cursors: "
            f"training={episode} environment={environment_episodes}"
        )

    run = Path(run_dir).expanduser().resolve()
    resume_dir = run / RESUME_DIRNAME
    resume_dir.mkdir(parents=True, exist_ok=True)
    state_name = f"training_state_ep{episode}.pth"
    state_path = resume_dir / state_name
    tmp_path = resume_dir / f".{state_name}.tmp-{os.getpid()}"
    state = {
        "schema_version": RESUME_SCHEMA_VERSION,
        "exact": True,
        "saved_at_utc": _utc_now(),
        "reason": str(reason),
        "completed_training_episode": episode,
        "completed_environment_episodes": environment_episodes,
        "context": context,
        "agent": agent.training_resume_state_dict(),
        "rng": capture_rng_state(),
        "histories": {
            "all_rewards": list(all_rewards),
            "all_local_rewards": list(all_local_rewards),
            "all_global_rewards": list(all_global_rewards),
            "performance_metrics": performance_metrics,
            "all_episode_data": all_episode_data,
        },
    }
    try:
        # torch.save alone leaves the bytes in the page cache, where the file
        # already carries its final size and the digest taken straight
        # afterwards still reads the cached copy, so a checkpoint can be
        # published unreadable. Force it down before the name points at it.
        with open(tmp_path, "wb") as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, state_path)
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass

    file_size = state_path.stat().st_size
    manifest = {
        "schema_version": RESUME_SCHEMA_VERSION,
        "exact": True,
        "saved_at_utc": state["saved_at_utc"],
        "reason": str(reason),
        "completed_training_episode": episode,
        "completed_environment_episodes": environment_episodes,
        "state_file": state_name,
        "state_bytes": int(file_size),
        "state_sha256": _sha256_file(state_path),
        "context": context,
    }
    _atomic_json_write(resume_dir / RESUME_MANIFEST, manifest)
    _append_event(
        run,
        {
            "event": "saved",
            "at_utc": manifest["saved_at_utc"],
            "episode": episode,
            "environment_episodes": environment_episodes,
            "reason": str(reason),
            "state_file": state_name,
            "state_bytes": int(file_size),
        },
    )

    # Keep one generation behind. Deleting every older file made a single bad
    # publish total: nothing to fall back to, and fifteen hours of training
    # riding on one write surviving.
    keep = set(sorted(
        resume_dir.glob("training_state_ep*.pth"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )[:RESUME_KEEP_GENERATIONS])
    for old_path in resume_dir.glob("training_state_ep*.pth"):
        if old_path not in keep:
            try:
                old_path.unlink()
            except OSError:
                pass
    return manifest


def _load_state(path: Path, map_location: Any) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _newest_loadable_state(
    resume_dir: Path, map_location: Any, *, exclude: Path
) -> "tuple[dict[str, Any], Path] | None":
    """The most recent kept checkpoint that actually reads back."""

    candidates = sorted(
        (path for path in resume_dir.glob("training_state_ep*.pth")
         if path != exclude),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        try:
            return _load_state(path, map_location), path
        except Exception:
            continue
    return None


def load_training_resume(
    run_dir: str | os.PathLike[str],
    *,
    expected_context: dict[str, Any],
    map_location: Any = "cpu",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify and load the latest exact state for ``run_dir``."""

    run = Path(run_dir).expanduser().resolve()
    manifest = read_resume_manifest(run)
    mismatches = _context_mismatches(manifest.get("context"), expected_context)
    if mismatches:
        detail = "\n  - ".join(mismatches[:20])
        raise ResumeStateError(
            "resume context differs from the saved experiment:\n  - " + detail
        )
    state_path = run / RESUME_DIRNAME / str(manifest["state_file"])
    if not state_path.is_file():
        raise ResumeStateError(f"resume state file is missing: {state_path}")
    actual_size = state_path.stat().st_size
    if actual_size != int(manifest["state_bytes"]):
        raise ResumeStateError(
            f"resume state size mismatch: expected {manifest['state_bytes']}, got {actual_size}"
        )
    actual_hash = _sha256_file(state_path)
    verified = actual_hash == str(manifest["state_sha256"])
    state = None
    if verified:
        try:
            state = _load_state(state_path, map_location)
        except Exception:
            state = None
    if state is None:
        # Size and digest can both agree while the payload is unreadable, which
        # is what a half-flushed publish looks like. Take the generation behind
        # it rather than losing the run, and say which one was used.
        fallback = _newest_loadable_state(
            run / RESUME_DIRNAME, map_location, exclude=state_path
        )
        if fallback is None:
            raise ResumeStateError(
                f"resume state {state_path.name} is unusable "
                f"({'does not load' if verified else 'SHA-256 mismatch'}) "
                "and no earlier checkpoint survived"
            )
        state, state_path = fallback
        print(
            f"[resume] {manifest['state_file']} is unusable; falling back to "
            f"{state_path.name} at episode "
            f"{state['completed_training_episode']}",
            flush=True,
        )
        manifest = dict(manifest)
        manifest["state_file"] = state_path.name
        manifest["completed_training_episode"] = int(
            state["completed_training_episode"]
        )
        manifest["completed_environment_episodes"] = int(
            state["completed_environment_episodes"]
        )
    if int(state.get("schema_version", -1)) != RESUME_SCHEMA_VERSION:
        raise ResumeStateError("resume payload schema does not match its manifest")
    if state.get("context") != manifest.get("context"):
        raise ResumeStateError("resume payload context does not match its manifest")
    _append_event(
        run,
        {
            "event": "loaded",
            "at_utc": _utc_now(),
            "episode": int(state["completed_training_episode"]),
            "environment_episodes": int(state["completed_environment_episodes"]),
            "state_file": str(manifest["state_file"]),
        },
    )
    return state, manifest
