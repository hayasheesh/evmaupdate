"""Runtime helpers for the deployed MADDPG + force-charging controller."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

import torch

from Config import (
    BATCH_SIZE,
    GAMMA,
    LR_ACTOR,
    LR_CRITIC_LOCAL,
    LR_GLOBAL_CRITIC,
    SMOOTHL1_BETA,
    TAU,
    TAU_GLOBAL,
    TD3_CLIP_GLOBAL,
    TD3_SIGMA_GLOBAL,
)
from EnvConfig import POWER_TO_ENERGY
from training.Agent import MADDPG
from training.Agent.standard_maddpg import build_marl_agent


_ACTOR_PATTERN = re.compile(r"actor_\d+_ep(\d+)\.pth$")
_TEST_DIR_PATTERN = re.compile(r"TEST(\d+)$")


def _actor_episode(filename: str) -> int | None:
    match = _ACTOR_PATTERN.fullmatch(filename)
    return int(match.group(1)) if match is not None else None


def find_model_path_and_episode(
    base_dir: str | Path, requested_episode: int | None = None
) -> tuple[str, int]:
    """Locate a per-station actor checkpoint below a run directory."""

    base = Path(base_dir)
    search_dirs = [base]
    results_dir = base / "results"
    if results_dir.is_dir():
        search_dirs.extend(
            path for path in results_dir.iterdir()
            if path.is_dir() and path.name.startswith("TEST")
        )

    candidates: list[tuple[int, float, Path]] = []
    for directory in search_dirs:
        if not directory.is_dir():
            continue
        for path in directory.iterdir():
            episode = _actor_episode(path.name)
            if episode is None:
                continue
            if requested_episode is not None and episode != int(requested_episode):
                continue
            candidates.append((episode, path.stat().st_mtime, directory))

    if not candidates:
        raise FileNotFoundError(f"No actor files were found under: {base}")
    if requested_episode is None:
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    else:
        candidates.sort(key=lambda item: item[1], reverse=True)
    episode, _, directory = candidates[0]
    return str(directory), int(episode)


def build_agent(env) -> MADDPG:
    """Construct the lower controller named by Config.MARL_ALGORITHM from an EVEnv."""

    return build_marl_agent(
        s_dim=int(env._get_obs().shape[1]),
        max_evs_per_station=env.max_ev_per_station,
        n_agent=env.num_stations,
        num_episodes=1,
        batch=BATCH_SIZE,
        gamma=GAMMA,
        tau=TAU,
        lr_a=LR_ACTOR,
        lr_c=LR_CRITIC_LOCAL,
        lr_global_c=LR_GLOBAL_CRITIC,
        tau_global=TAU_GLOBAL,
        td3_sigma=TD3_SIGMA_GLOBAL,
        td3_clip=TD3_CLIP_GLOBAL,
        smoothl1_beta=SMOOTHL1_BETA,
    )


def validate_actor_checkpoint_compatibility(
    agent: MADDPG,
    model_path: str | Path,
    episode: int,
    expected_stations: int,
) -> None:
    """Require one saved actor for every station in the current environment."""

    del agent
    root = Path(model_path)
    required = [root / f"actor_{index}_ep{episode}.pth" for index in range(expected_stations)]
    missing = [path for path in required if not path.is_file()]
    if not missing:
        return
    preview = ", ".join(path.name for path in missing[:5])
    if len(missing) > 5:
        preview += f", ... (+{len(missing) - 5} more)"
    found = len(list(root.glob(f"actor_*_ep{episode}.pth")))
    raise FileNotFoundError(
        f"Checkpoint requires {expected_stations} station actors for episode {episode}; "
        f"missing {preview}. Found {found} in {root}."
    )


def apply_force_charging(actions, env, *, slack_kwh: float = 0.0):
    """Raise EV actions only when needed to keep departure energy reachable."""

    if isinstance(actions, torch.Tensor):
        actions = actions.clone()
    else:
        actions = torch.as_tensor(actions, dtype=torch.float32, device=env.soc.device)
    forced = 0
    forced_by_station = [0] * env.num_stations
    forced_kw = 0.0
    slack = max(0.0, float(slack_kwh))

    for station in range(env.num_stations):
        active = torch.nonzero(env.ev_mask[station], as_tuple=False).squeeze(-1)
        if active.numel() == 0:
            continue
        sorted_active = env._sort_active_evs(station, active)
        for order_index, ev_tensor in enumerate(sorted_active):
            ev_index = int(ev_tensor.item())
            max_power_kw = float(env.ev_max_power_kw[station, ev_index].item())
            max_step_kwh = max_power_kw * float(POWER_TO_ENERGY)
            if max_step_kwh <= 0.0:
                continue

            kwh_per_soc_pct = float(env.ev_kwh_per_soc_pct[station, ev_index].item())
            if kwh_per_soc_pct <= 0.0:
                kwh_per_soc_pct = 1.0
            need_soc_pct = float(
                (env.target[station, ev_index] - env.soc[station, ev_index]).item()
            )
            need_kwh = max(0.0, need_soc_pct) * kwh_per_soc_pct + slack
            if need_kwh <= 0.0:
                continue

            remaining_steps = (
                int(env.depart[station, ev_index].item()) - int(env.step_count) + 1
            )
            if remaining_steps <= 0:
                continue
            floor_kwh = need_kwh - (remaining_steps - 1) * max_step_kwh
            if floor_kwh <= -max_step_kwh:
                continue
            min_action = min(floor_kwh, max_step_kwh) / max_step_kwh
            current_action = float(actions[station, order_index].item())
            if current_action >= min_action - 1e-6:
                continue
            actions[station, order_index] = min_action
            forced += 1
            forced_by_station[station] += 1
            forced_kw += (min_action - current_action) * max_power_kw

    apply_force_charging.last_forced_kw = float(forced_kw)
    return actions, forced, forced_by_station


apply_force_charging.last_forced_kw = 0.0


def _test_directories(results_dir: Path) -> dict[int, Path]:
    found: dict[int, Path] = {}
    for path in results_dir.iterdir():
        match = _TEST_DIR_PATTERN.fullmatch(path.name) if path.is_dir() else None
        if match is not None:
            found[int(match.group(1))] = path
    return found


def _test_history_episodes(results_dir: Path) -> list[int] | None:
    path = results_dir / "test_history.json"
    if not path.is_file():
        return None
    values = json.loads(path.read_text(encoding="utf-8")).get("episodes")
    if not isinstance(values, list):
        return None
    try:
        return [int(value) for value in values]
    except (TypeError, ValueError):
        return None


# A MARL result is a policy trained for at least this many episodes (AGENTS.md).
MIN_REPORTED_TRAINING_EPISODE = 1000


def choose_best_episode(
    model_dir: str | Path,
    *,
    min_episode: int = MIN_REPORTED_TRAINING_EPISODE,
) -> int:
    """Choose the checkpoint with maximum test SoC plus tracking rate.

    Only checkpoints at or after ``min_episode`` are candidates. The tracking
    rate counts every assessed step, no-instruction steps included; a row
    written before those were recorded has no rate and is skipped.
    """

    results_dir = Path(model_dir) / "results"
    test_dirs = _test_directories(results_dir)
    if not test_dirs:
        raise FileNotFoundError(f"No TEST* directories found under: {results_dir}")
    history = _test_history_episodes(results_dir)
    if history is None:
        raise FileNotFoundError(f"{results_dir / 'test_history.json'} is missing; pass the episode explicitly")
    metrics_path = results_dir / "test_performance_metrics.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"{metrics_path} is missing; pass the episode explicitly")
    with metrics_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    best: tuple[int, float] | None = None
    for index, row in enumerate(rows):
        try:
            score = float(row["SoC_Hit_Rate_%"]) + float(
                row["Dispatch_Tracking_Rate_%"]
            )
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(score):
            continue
        if index >= len(history):
            continue
        episode = history[index]
        if episode not in test_dirs or episode < int(min_episode):
            continue
        if best is None or score > best[1] or (score == best[1] and episode > best[0]):
            best = (episode, score)
    if best is None:
        raise ValueError(
            f"no checkpoint at or after episode {int(min_episode)} has a test score "
            f"with no-instruction steps in {metrics_path}; pass the episode explicitly"
        )
    return best[0]
