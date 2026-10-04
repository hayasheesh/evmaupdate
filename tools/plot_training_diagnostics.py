"""Export compact learning and optimization diagnostics from one archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def _rolling(values, window: int = 50):
    return pd.Series(np.asarray(values, dtype=float)).rolling(window, min_periods=1).mean().to_numpy()


def _load_scalars(archive: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    files = list((archive / "performance").glob("events.out.tfevents.*"))
    if not files:
        return {}
    event_file = max(files, key=lambda path: path.stat().st_size)
    accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    accumulator.Reload()
    result = {}
    for tag in accumulator.Tags().get("scalars", []):
        events = accumulator.Scalars(tag)
        result[tag] = (
            np.asarray([event.step for event in events], dtype=float),
            np.asarray([event.value for event in events], dtype=float),
        )
    return result


def _series(scalars, tag: str, through_episode: int | None):
    if tag not in scalars:
        return np.asarray([]), np.asarray([])
    x, y = scalars[tag]
    if through_episode is not None:
        keep = x <= int(through_episode)
        x, y = x[keep], y[keep]
    return x, y


def _plot_tag(ax, scalars, tag, label, through_episode, *, smooth=20, **kwargs):
    x, y = _series(scalars, tag, through_episode)
    if y.size:
        ax.plot(x, _rolling(y, smooth), label=label, **kwargs)


def plot_training_diagnostics(
    archive_dir: str | Path,
    *,
    through_episode: int | None = None,
    output_dir: str | Path | None = None,
) -> list[Path]:
    archive = Path(archive_dir).resolve()
    results = archive / "results"
    out = Path(output_dir).resolve() if output_dir else results / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    suffix = f"_ep{int(through_episode)}" if through_episode is not None else ""
    written: list[Path] = []

    train_rewards_path = results / "episode_rewards_all.csv"
    test_metrics_path = results / "test_performance_metrics.csv"
    history_path = results / "test_history.json"
    if train_rewards_path.exists() and test_metrics_path.exists() and history_path.exists():
        train = pd.read_csv(train_rewards_path)
        test = pd.read_csv(test_metrics_path)
        history = json.loads(history_path.read_text(encoding="utf-8"))
        episodes = np.asarray(history.get("episodes", []), dtype=int)[: len(test)]
        test_local_rewards = np.asarray(history.get("local_rewards", []), dtype=float)[: len(episodes)]
        test_global_rewards = np.asarray(history.get("global_rewards", []), dtype=float)[: len(episodes)]
        if through_episode is not None:
            train = train[train["Episode"] <= int(through_episode)]
            keep = episodes <= int(through_episode)
            test_local_rewards = test_local_rewards[keep]
            test_global_rewards = test_global_rewards[keep]
            episodes, test = episodes[keep], test.iloc[np.flatnonzero(keep)]

        fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
        ax = axes[0, 0]
        ax.plot(train["Episode"], _rolling(train["Local_Reward"]), label="Train local (rolling 50)")
        ax.plot(train["Episode"], _rolling(train["Global_Reward"]), label="Train global (rolling 50)")
        if len(episodes):
            ax.plot(episodes, test_local_rewards, label="Test local")
            ax.plot(episodes, test_global_rewards, label="Test global")
        ax.set_title("Reward")
        ax.legend(fontsize=8)

        ax = axes[0, 1]
        if len(episodes):
            ax.plot(episodes, test["SoC_Hit_Rate_%"], label="Local SoC hit")
            ax.plot(episodes, test["Dispatch_Tracking_Rate_%"], label="Global tracking")
        ax.axhline(100, color="tab:blue", linestyle="--", alpha=0.4)
        ax.axhline(90, color="tab:orange", linestyle="--", alpha=0.4)
        ax.set_ylim(0, 105)
        ax.set_title("Held-out pass rates")
        ax.legend(fontsize=8)

        ax = axes[1, 0]
        if len(episodes):
            surplus = 100 * test["Surplus_Steps_Within_Narrow"].to_numpy() / np.maximum(test["Surplus_Steps"], 1)
            shortage = 100 * test["Shortage_Steps_Within_Narrow"].to_numpy() / np.maximum(test["Shortage_Steps"], 1)
            ax.plot(episodes, surplus, label="Surplus direction")
            ax.plot(episodes, shortage, label="Shortage direction")
        ax.set_ylim(0, 105)
        ax.set_title("Tracking by direction")
        ax.legend(fontsize=8)

        ax = axes[1, 1]
        if len(episodes):
            ax.plot(episodes, test["Avg_SoC_Deficit_kWh"], color="tab:red")
        ax.set_title("Average missed SoC energy")
        ax.set_ylabel("kWh")
        for axis in axes.flat:
            axis.set_xlabel("Training episode")
            axis.grid(alpha=0.25)
        path = out / f"training_progress{suffix}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path)

    scalars = _load_scalars(archive)
    if scalars:
        fig, axes = plt.subplots(3, 2, figsize=(16, 14), constrained_layout=True)
        ax = axes[0, 0]
        _plot_tag(ax, scalars, "Gradient/actor_source_local_raw_mean", "Actor from local", through_episode)
        _plot_tag(ax, scalars, "Gradient/actor_source_global_raw_mean", "Actor from global", through_episode)
        _plot_tag(ax, scalars, "Gradient/global_critic_raw", "Global critic", through_episode)
        ax.set_yscale("log")
        ax.set_title("Raw gradient norms before clipping")
        ax.legend(fontsize=8)

        ax = axes[0, 1]
        _plot_tag(ax, scalars, "Gradient/actor_source_cos_mean", "Local/global cosine", through_episode)
        _plot_tag(ax, scalars, "Gradient/actor_source_global_ratio_mean", "Global norm ratio", through_episode)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title("Actor gradient interaction")
        ax.legend(fontsize=8)

        ax = axes[1, 0]
        _plot_tag(ax, scalars, "Loss/local_critic_mean", "Local critic", through_episode)
        _plot_tag(ax, scalars, "Loss/global_critic", "Global critic", through_episode)
        ax.set_title("Critic losses")
        ax.legend(fontsize=8)

        ax = axes[1, 1]
        actor_clip_tags = sorted(tag for tag in scalars if tag.startswith("Clipping/actor_agent"))
        actor_clips = []
        clip_x = None
        for tag in actor_clip_tags:
            x, y = _series(scalars, tag, through_episode)
            if y.size:
                clip_x, actor_clips = x, actor_clips + [y]
        if actor_clips:
            n = min(map(len, actor_clips))
            matrix = np.vstack([values[-n:] for values in actor_clips])
            x = clip_x[-n:]
            ax.plot(x, _rolling(matrix.mean(axis=0), 20), label="Actor mean")
            ax.plot(x, _rolling(matrix.max(axis=0), 20), label="Actor max")
        _plot_tag(ax, scalars, "Clipping/global_critic", "Global critic", through_episode)
        ax.set_ylim(bottom=0)
        ax.set_title("Gradient clipping fraction")
        ax.legend(fontsize=8)

        ax = axes[2, 0]
        _plot_tag(ax, scalars, "Q/local_mean", "Local Q", through_episode)
        _plot_tag(ax, scalars, "Q/global", "Global Q", through_episode)
        ax.set_title("Q-value drift")
        ax.legend(fontsize=8)

        ax = axes[2, 1]
        _plot_tag(ax, scalars, "Training/epsilon", "Epsilon", through_episode, smooth=1)
        _plot_tag(ax, scalars, "Training/ou_noise_scale", "OU noise", through_episode, smooth=1)
        _plot_tag(ax, scalars, "GlobalCritic/td_error_abs_mean", "Global TD error", through_episode)
        ax.set_title("Exploration and TD error")
        ax.legend(fontsize=8)
        for axis in axes.flat:
            axis.set_xlabel("Training episode")
            axis.grid(alpha=0.25)
        path = out / f"optimization_health{suffix}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--episode", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    for path in plot_training_diagnostics(
        args.archive, through_episode=args.episode, output_dir=args.output_dir
    ):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
