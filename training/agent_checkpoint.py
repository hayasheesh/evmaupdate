"""Locate and restore MADDPG training checkpoints for warm-start fine-tuning.

Checkpoint format (written by MADDPG.save_checkpoint at each interim test):

    <run_dir>/results/TEST<ep>/actor_<i>_ep<ep>.pth   (legacy, one per station)
    <run_dir>/results/TEST<ep>/agent_state_ep<ep>.pth (critics + optimizers)

Older archives only contain the actor files; loading those is a weights-only
warm start (critics/optimizers stay freshly initialized).
"""

from __future__ import annotations

import os
import re
from pathlib import Path


_ACTOR_FILE_RE = re.compile(r"^actor_(\d+)_ep(\d+)\.pth$")

# The OU half of the exploration schedule, which a fine-tune still scales: at
# the default factor it lands the noise where the pretrain left it.
_OU_SCALE_ATTRS = (
    "ou_noise_scale_initial",
    "ou_noise_scale_final",
    "ou_noise_scale",
)


def _actor_episodes_in_dir(path) -> list[int]:
    try:
        names = os.listdir(path)
    except OSError:
        return []
    episodes = set()
    for name in names:
        match = _ACTOR_FILE_RE.match(str(name))
        if match:
            episodes.add(int(match.group(2)))
    return sorted(episodes)


def find_latest_checkpoint(root) -> tuple[str, int]:
    """Return (checkpoint_dir, episode) of the newest actor checkpoint.

    `root` may be the checkpoint directory itself (results/TEST<ep>), a run
    work_dir, or a run's results directory; the newest episode wins.
    """

    root_path = Path(root)
    candidates: list[tuple[int, str]] = []
    direct = _actor_episodes_in_dir(root_path)
    if direct:
        candidates.append((direct[-1], str(root_path)))
    for base in (root_path / "results", root_path):
        if not base.is_dir():
            continue
        for child in sorted(base.iterdir()):
            if child.is_dir():
                episodes = _actor_episodes_in_dir(child)
                if episodes:
                    candidates.append((episodes[-1], str(child)))
    if not candidates:
        raise FileNotFoundError(
            f"no actor checkpoints (actor_*_ep*.pth) found under {root_path}"
        )
    episode, path = max(candidates, key=lambda item: (item[0], item[1]))
    return path, int(episode)


def initialize_agent_from_checkpoint(agent, checkpoint, *, episode=None) -> dict:
    """Load a saved checkpoint into an already constructed agent.

    Prefers the full-bundle load (actors + critics + optimizers) when the
    agent supports it and the bundle file exists; otherwise falls back to the
    legacy actors-only load. The replay buffer is untouched (fresh runs start
    with an empty buffer by construction).
    """

    if episode is None:
        checkpoint_dir, checkpoint_episode = find_latest_checkpoint(checkpoint)
    else:
        checkpoint_dir, checkpoint_episode = str(checkpoint), int(episode)
    if hasattr(agent, "load_checkpoint"):
        info = dict(agent.load_checkpoint(checkpoint_dir, checkpoint_episode))
    elif hasattr(agent, "load_actors"):
        agent.load_actors(checkpoint_dir, checkpoint_episode)
        info = {
            "path": str(checkpoint_dir),
            "episode": int(checkpoint_episode),
            "actors": True,
            "critics": False,
            "optimizers": False,
        }
    else:
        raise TypeError(
            f"agent {type(agent).__name__} supports neither load_checkpoint nor load_actors"
        )
    applied_lrs = apply_configured_learning_rates(agent)
    print(
        "[warm-start] initialized agent from "
        f"{info.get('path')} ep{info.get('episode')} "
        f"(actors={info.get('actors')} critics={info.get('critics')} "
        f"optimizers={info.get('optimizers')})",
        flush=True,
    )
    if applied_lrs:
        info["learning_rates"] = dict(applied_lrs)
        print(
            "[warm-start] learning rates re-applied after the restore: "
            + "  ".join(f"{name}={lr:g}" for name, lr in applied_lrs.items()),
            flush=True,
        )
    info["offline_replay_size"] = restore_balanced_replay(
        agent, checkpoint_dir, checkpoint_episode
    )
    return info


def apply_configured_learning_rates(agent) -> dict:
    """Put the configured learning rates back after a checkpoint restore.

    torch's Optimizer.load_state_dict replaces param_groups wholesale, and a
    param group carries the learning rate, so restoring the pretrain's
    optimizer bundle also restores the pretrain's rate. EVMA_LR_ACTOR and its
    two companions were therefore applied at construction and overwritten a
    moment later: a fine-tune printed the rate it was asked for and stepped at
    the saved one. Adam's moments are what the restore is for and they stay.
    """

    from Config import LR_ACTOR, LR_CRITIC_LOCAL, LR_GLOBAL_CRITIC

    targets = (
        ("opt_a", float(LR_ACTOR)),
        ("opt_c", float(LR_CRITIC_LOCAL)),
        ("opt_c2", float(LR_CRITIC_LOCAL)),
        ("opt_global_c1", float(LR_GLOBAL_CRITIC)),
        ("opt_global_c2", float(LR_GLOBAL_CRITIC)),
    )
    applied: dict = {}
    for attr, lr in targets:
        holder = getattr(agent, attr, None)
        if holder is None:
            continue
        optimizers = holder if isinstance(holder, (list, tuple)) else [holder]
        for optimizer in optimizers:
            if optimizer is None:
                continue
            for group in optimizer.param_groups:
                group["lr"] = lr
            applied[attr] = lr
    return applied


def restore_balanced_replay(agent, checkpoint_dir, checkpoint_episode) -> int:
    """Seed the buffer with the checkpoint's transitions as an offline region.

    Fine-tuning from a warm-started policy with an empty buffer refits the
    restored critic on whatever little online data exists, which degrades the
    policy before it improves. Inheriting the pretrain transitions and mixing
    them into every minibatch is the balanced-replay remedy from the
    offline-to-online RL literature. A missing snapshot is not an error: the
    run just falls back to the previous empty-buffer behaviour.
    """

    from Config import (
        BALANCED_REPLAY_DECAY_STEPS,
        BALANCED_REPLAY_RATIO_FINAL,
        BALANCED_REPLAY_RATIO_INITIAL,
    )

    if not hasattr(agent, "load_replay_snapshot"):
        return 0
    loaded = int(agent.load_replay_snapshot(checkpoint_dir, checkpoint_episode))
    if loaded <= 0:
        print(
            "[warm-start] no replay snapshot found; fine-tune starts with an "
            "empty buffer (balanced replay disabled)",
            flush=True,
        )
        return 0
    buf = getattr(agent, "buf", None)
    if buf is None or not hasattr(buf, "mark_offline_region"):
        return 0
    buf.mark_offline_region(
        ratio_initial=float(BALANCED_REPLAY_RATIO_INITIAL),
        ratio_final=float(BALANCED_REPLAY_RATIO_FINAL),
        decay_steps=int(BALANCED_REPLAY_DECAY_STEPS),
    )
    print(
        f"[warm-start] balanced replay: {loaded} offline transitions "
        f"(offline share {float(BALANCED_REPLAY_RATIO_INITIAL):.2f} -> "
        f"{float(BALANCED_REPLAY_RATIO_FINAL):.2f} over "
        f"{int(BALANCED_REPLAY_DECAY_STEPS)} online steps)",
        flush=True,
    )
    return loaded


def apply_finetune_exploration_schedule(agent, *, noise_scale, episodes=None) -> dict:
    """Give the fine-tune its own epsilon schedule; scale the OU noise as before.

    The epsilon endpoints the pretrain uses belong to a pretrain: 1.0 down to
    0.005 over EPSILON_END_EPISODE episodes, starting from an untrained actor.
    Scaling them by the fine-tune factor and restarting the schedule -- which is
    what happened, because the warm start does not restore the episode counter
    -- left the slot-level random override at 30% on the first fine-tune episode
    and still 21% on the three hundredth, against the 0.5% the pretrain finished
    at.  The adaptation spent its whole budget re-exploring the policy it had
    been handed.

    The OU scaling is deliberately unchanged.  At the default factor it lands
    the noise within a few percent of where the pretrain left it -- 0.30 against
    0.2854 at episode 1340 -- so epsilon was the half that was wrong.

    `episodes` is the fine-tune's own budget.  Epsilon decays across it, which
    is the only length that finishes inside the run it belongs to.
    """

    factor = float(noise_scale)
    if not (factor >= 0.0):
        raise ValueError(
            f"exploration noise scale must be non-negative, got {noise_scale!r}"
        )
    from Config import (FINETUNE_EPSILON_END_EPISODE, FINETUNE_EPSILON_FINAL,
                        FINETUNE_EPSILON_INITIAL)

    # hasattr, because not every agent type carries an OU schedule. No try
    # around the arithmetic though: an agent that has the attribute and cannot
    # scale it is a caller bug, and swallowing it would leave the fine-tune
    # running at the pretrain's noise with nothing said.
    for name in _OU_SCALE_ATTRS:
        if hasattr(agent, name):
            setattr(agent, name, float(getattr(agent, name)) * factor)

    end_episode = int(FINETUNE_EPSILON_END_EPISODE)
    if end_episode <= 0:
        end_episode = max(1, int(episodes)) if episodes else 1
    applied = {
        "epsilon_start_episode": 1,
        "epsilon_end_episode": end_episode,
        "epsilon_initial": float(FINETUNE_EPSILON_INITIAL),
        "epsilon_final": float(FINETUNE_EPSILON_FINAL),
        "epsilon": float(FINETUNE_EPSILON_INITIAL),
    }
    for name, value in applied.items():
        if hasattr(agent, name):
            setattr(agent, name, value)
    return applied


def _value_heads(module):
    """Yield the (name, submodule) value heads of one critic.

    Local critics carry a single `q_head`; the QMIX-style global critic carries
    a per-station head and the two mixer heads. Anything else is left alone:
    resetting an encoder would discard the representation the warm start is
    there to provide.
    """

    for name in ("q_head", "per_agent_head", "mixer_w", "mixer_b"):
        head = getattr(module, name, None)
        if head is not None:
            yield name, head


def _reinit(module, scope: str, init_gain: float) -> int:
    """Reinitialize a head in place and return how many Linear layers moved."""

    import torch.nn as nn

    linears = [m for m in module.modules() if isinstance(m, nn.Linear)]
    if not linears:
        return 0
    targets = linears[-1:] if scope == "last_linear" else linears
    for layer in targets:
        nn.init.xavier_uniform_(layer.weight, gain=float(init_gain))
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    if scope != "last_linear":
        for m in module.modules():
            if isinstance(m, nn.LayerNorm) and m.elementwise_affine:
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    return len(targets)


def reset_critic_heads(agent, scope: str) -> dict:
    """Reinitialize the critics' value heads after a warm start.

    Both the online and the target copy are reset, and the optimizer moments
    for the reset parameters are dropped. Leaving the target or the moments
    alone would pull the fresh head straight back to the value it was meant to
    forget, which is the opposite of the intent.
    """

    import copy

    scope = str(scope or "off").strip().lower()
    if scope == "off":
        return {"scope": "off", "critics": 0, "layers": 0}

    groups = (
        ("local", "critics", "t_critics", "opt_c"),
        ("local2", "critics2", "t_critics2", "opt_c2"),
        ("global", "global_critic1", "t_global_critic1", "opt_global_c1"),
        ("global2", "global_critic2", "t_global_critic2", "opt_global_c2"),
    )
    touched_layers = 0
    touched_critics = 0
    head_names: set[str] = set()
    for _label, online_attr, target_attr, opt_attr in groups:
        online = getattr(agent, online_attr, None)
        if online is None:
            continue
        online_list = online if isinstance(online, (list, tuple)) else [online]
        for net in online_list:
            if net is None:
                continue
            gain = float(getattr(net, "init_gain", 1.0))
            moved = 0
            for name, head in _value_heads(net):
                moved += _reinit(head, scope, gain)
                head_names.add(name)
            if moved:
                touched_critics += 1
                touched_layers += moved
        # The target must not keep the old head.
        target = getattr(agent, target_attr, None)
        if target is not None:
            fresh = [copy.deepcopy(net) for net in online_list]
            if isinstance(target, list):
                target[:] = fresh
            else:
                setattr(agent, target_attr, fresh[0])
        # Stale Adam moments for a freshly initialized weight are misleading.
        opt = getattr(agent, opt_attr, None)
        if opt is not None:
            for optimizer in (opt if isinstance(opt, (list, tuple)) else [opt]):
                if optimizer is not None:
                    optimizer.state.clear()
    return {
        "scope": scope,
        "critics": touched_critics,
        "layers": touched_layers,
        "heads": sorted(head_names),
    }


# What a fixed bid changes about a day is its award band and the command
# profile that follows from it. In this critic that information only reaches
# the value function through the mixer, which reads the demand/time context and
# the realized station powers. The per-station embedding and the per-agent
# utility head describe station physics, which the day does not change.
_GLOBAL_DAY_INVARIANT_PATHS = ("per_station_sa", "per_agent_head")

_FREEZE_SCOPES = ("off", "all", "global_mixer", "global_mixer_strict")


def _freeze_in_place(params) -> int:
    """Pin parameters by zeroing their gradient as it is produced.

    This is needed on top of dropping them from the optimizer because the
    gradient-norm clip is taken over a whole critic: a frozen parameter that
    kept accumulating an unzeroed gradient would inflate that norm and shrink
    the step taken by the parameters that are still learning.
    """

    import torch

    def _zero(grad):
        return torch.zeros_like(grad)

    pinned = 0
    for p in params:
        if getattr(p, "_evma_frozen", False):
            continue
        p.register_hook(_zero)
        p._evma_frozen = True
        pinned += 1
    return pinned


def _rebuild_optimizer(optimizer, keep):
    """Return an Adam over `keep` only, carrying the moments of the survivors.

    Excluding the parameters from the optimizer is what actually holds them.
    `requires_grad` does not survive here, because the actor update restores it
    on every critic parameter once it has run, and a zero gradient is not
    enough on its own: Adam still moves a parameter that carries a moment from
    an earlier step.
    """

    import torch.optim as optim

    if optimizer is None:
        return None
    lr = float(optimizer.param_groups[0].get("lr", 1e-4))
    keep = list(keep)
    fresh = optim.Adam([{"params": keep}], lr=lr)
    for p in keep:
        moments = optimizer.state.get(p)
        if moments is not None:
            fresh.state[p] = moments
    return fresh


def freeze_critic_paths(agent, scope: str) -> dict:
    """Restrict which value-function parameters a fine-tune is allowed to move.

    `global_mixer` leaves only the mixer path of the global critic adapting, so
    the capacity that is spent on the new day is aimed at the part that sees
    the shift. `global_mixer_strict` additionally holds the local critics,
    which score SoC and departure and are day-invariant for the same reason,
    making the mixer the only part of the value function that can still move.
    """

    scope = str(scope or "off").strip().lower()
    if scope in ("", "off", "all"):
        return {"scope": "off", "frozen": 0, "trainable": 0, "paths": []}
    if scope not in _FREEZE_SCOPES:
        raise ValueError(
            f"unknown critic freeze scope: {scope!r} (expected one of {_FREEZE_SCOPES})"
        )

    strict = scope == "global_mixer_strict"
    frozen_total = 0
    trainable_total = 0
    paths: set[str] = set()

    for online_attr, opt_attr in (
        ("global_critic1", "opt_global_c1"),
        ("global_critic2", "opt_global_c2"),
    ):
        net = getattr(agent, online_attr, None)
        opt = getattr(agent, opt_attr, None)
        if net is None or opt is None:
            continue
        held, keep = [], []
        for name, p in net.named_parameters():
            root = name.split(".", 1)[0]
            if root in _GLOBAL_DAY_INVARIANT_PATHS:
                held.append(p)
                paths.add(root)
            else:
                keep.append(p)
        frozen_total += _freeze_in_place(held)
        trainable_total += len(keep)
        setattr(agent, opt_attr, _rebuild_optimizer(opt, keep))

    if strict:
        for list_attr, opt_attr in (("critics", "opt_c"), ("critics2", "opt_c2")):
            nets = getattr(agent, list_attr, None) or []
            opts = getattr(agent, opt_attr, None) or []
            for i, net in enumerate(nets):
                if net is None:
                    continue
                frozen_total += _freeze_in_place(list(net.parameters()))
                paths.add("local_q_head")
                if i < len(opts) and opts[i] is not None:
                    opts[i] = _rebuild_optimizer(opts[i], [])

    return {
        "scope": scope,
        "frozen": frozen_total,
        "trainable": trainable_total,
        "paths": sorted(paths),
    }
