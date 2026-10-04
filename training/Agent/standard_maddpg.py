"""MADDPG as in Lowe et al. (2017), for comparison with the research learner.

The research learner (training/Agent/maddpg.py) departs from MADDPG in several
ways: each station has a local critic that sees only its own station, a
separate global critic with a QMIX-style mixer sees every station, the two are
trained on different rewards with different discounts, the actor follows a
fixed blend of the two gradients, and critics are TD3-style twins with target
smoothing and a delayed actor. This class keeps everything the comparison
should hold fixed -- actor network, observation, action semantics, physical
action limits, replay memory, learning rates, batch, target rate, network width
and gradient clipping -- and replaces only the learning algorithm:

  critic    one per station, Q_i(o_1..o_N, a_1..a_N): every station's
            observation and executed action, concatenated
  reward    r_i = w_local * r_local_i + w_global * r_global
  target    y_i = r_i + gamma * Q'_i(o', mu'_1(o'_1)..mu'_N(o'_N)) * (1 - d_i),
            one discount, no target-policy smoothing, no twin minimum
  loss      mean squared error
  actor     grad of Q_i with respect to a_i = mu_i(o_i), the other stations'
            actions taken from the replay sample; updated every step
  explore   Gaussian noise on the actor output only

Critics exist only during training (centralized training, decentralized
execution); the saved actors are interchangeable with the research learner's.
"""
from __future__ import annotations

import copy
import math
import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from Config import (
    BATCH_SIZE, LR_ACTOR, LR_CRITIC_LOCAL, TAU,
    STD_MADDPG_CRITIC_HIDDEN, STD_MADDPG_GAMMA, STD_MADDPG_GLOBAL_REWARD_WEIGHT,
    STD_MADDPG_GRAD_CLIP, STD_MADDPG_LOCAL_REWARD_WEIGHT,
    STATION_RULE_ALLOCATION,
)

try:
    from .maddpg import MADDPG, device
except ImportError:
    from maddpg import MADDPG, device


class CentralizedCritic(nn.Module):
    """Q_i over the concatenated observations and actions of every station."""

    def __init__(self, obs_dim: int, act_dim: int, hid: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, 1),
        )

    def forward(self, obs_all: torch.Tensor, act_all: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs_all, act_all], dim=1))


class StandardMADDPG(MADDPG):
    def __init__(self, s_dim, max_evs_per_station, n_agent, *,
                 batch=BATCH_SIZE, tau=TAU, lr_a=LR_ACTOR, lr_c=LR_CRITIC_LOCAL,
                 num_episodes=1, **kwargs):
        del kwargs  # the research learner's critic settings do not apply
        super().__init__(
            s_dim, max_evs_per_station, n_agent,
            gamma=STD_MADDPG_GAMMA, tau=tau, batch=batch,
            lr_a=lr_a, lr_c=lr_c, num_episodes=num_episodes,
        )
        self.local_reward_weight = float(STD_MADDPG_LOCAL_REWARD_WEIGHT)
        self.global_reward_weight = float(STD_MADDPG_GLOBAL_REWARD_WEIGHT)
        self.grad_clip = float(STD_MADDPG_GRAD_CLIP)
        self.critic_hidden = int(STD_MADDPG_CRITIC_HIDDEN)
        self.policy_delay = 1

        # Gaussian noise on the actor output is the only exploration.
        self.epsilon_initial = self.epsilon_final = self.epsilon = 0.0
        self.global_correlated_noise_gain = 0.0

        obs_dim = self.n * int(s_dim)
        act_dim = self.n * int(max_evs_per_station)
        self.critics = [
            CentralizedCritic(obs_dim, act_dim, self.critic_hidden).to(device)
            for _ in range(n_agent)
        ]
        self.t_critics = [copy.deepcopy(c) for c in self.critics]
        self.opt_c = [optim.Adam(c.parameters(), lr=lr_c) for c in self.critics]
        # The research learner's second local critics and global critics are
        # not part of MADDPG.
        self.critics2, self.t_critics2, self.opt_c2 = [], [], []
        self.global_critic1 = self.global_critic2 = None
        self.t_global_critic1 = self.t_global_critic2 = None
        self.opt_global_c1 = self.opt_global_c2 = None

    # --- learning -----------------------------------------------------------

    def _executed_actions(self, raw, current_socs, capacity_kwh, max_power_kw,
                          soc_step_per_kw, kw_per_soc_step, inv_max_power_kw,
                          padding_mask, use_ste, ev_block=None):
        """Map actor outputs in [-1, 1] to the normalized actions the plant executes."""
        kw = raw if STATION_RULE_ALLOCATION else (raw * max_power_kw).masked_fill(padding_mask, 0.0)
        clamped_kw, _ = self._apply_soc_constraint(
            kw, current_socs, padding_mask, use_ste=use_ste,
            capacity_kwh=capacity_kwh, max_power_kw=max_power_kw,
            soc_step_per_kw=soc_step_per_kw, kw_per_soc_step=kw_per_soc_step,
            ev_block=ev_block,
        )
        normalized = self._normalize_power_by_limit(clamped_kw, max_power_kw, inv_max_power_kw)
        return normalized.masked_fill(padding_mask, 0.0)

    def _clip_and_measure(self, module):
        norm = torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm=self.grad_clip)
        return norm.detach() if isinstance(norm, torch.Tensor) else torch.tensor(float(norm), device=device)

    def update(self):
        if self.test_mode:
            self._zero_update_logs()
            return
        if self.buf.pending_size < self.warmup_steps:
            self._zero_update_logs()
            return
        self.update_step += 1

        (s, s2, a, r_local, _r_global, d, actual_station_powers, actual_ev_power_kw,
         r_global_n, _s2_n, _d_n, _n_eff) = self.buf.sample_with_nstep_global(self.batch, 1, self.gamma)
        ctx = self._build_update_ctx(s, s2, a, r_local, d, actual_station_powers, actual_ev_power_kw)
        B = s.size(0)
        obs, obs2 = s.reshape(B, -1), s2.reshape(B, -1)
        a_exec = ctx['a_actual'].masked_fill(ctx['ev_padding_mask'], 0.0)
        r_global_n = r_global_n.reshape(B, 1)

        # Critics.
        with torch.no_grad():
            next_raw = torch.stack([self.t_actors[i](s2[:, i, :]) for i in range(self.n)], dim=1)
            next_exec = self._executed_actions(
                next_raw, ctx['current_socs_s2'], ctx['capacity_kwh_s2'], ctx['max_power_kw_s2'],
                ctx['soc_step_per_kw_s2'], ctx['kw_per_soc_step_s2'], ctx['inv_max_power_kw_s2'],
                ctx['ev_padding_mask_s2'], use_ste=False, ev_block=ctx['ev_block_s2'],
            ).reshape(B, -1)
            targets = []
            for i in range(self.n):
                reward = self.local_reward_weight * r_local[:, i:i + 1] + self.global_reward_weight * r_global_n
                y = reward + self.gamma * self.t_critics[i](obs2, next_exec) * (1.0 - d[:, i:i + 1])
                targets.append(torch.nan_to_num_(y, nan=0.0, posinf=0.0, neginf=0.0))

        a_flat = a_exec.reshape(B, -1)
        critic_losses, critic_norms = [], []
        for i in range(self.n):
            q = self.critics[i](obs, a_flat)
            loss = F.mse_loss(q, targets[i])
            self.opt_c[i].zero_grad()
            loss.backward()
            critic_norms.append(self._clip_and_measure(self.critics[i]))
            critic_losses.append(loss.detach())
        critic_ready = []
        losses_c, norms_c = torch.stack([torch.stack(critic_losses), torch.stack(critic_norms)]).cpu().tolist()
        local_critic_clip_count = 0
        self.local_critic_clip_counts = [0] * self.n
        self.critic_losses = [0.0] * self.n
        for i in range(self.n):
            finite = math.isfinite(losses_c[i]) and math.isfinite(norms_c[i])
            self.critic_losses[i] = losses_c[i] if finite else 0.0
            self.critic_norms_before_clip[i] = norms_c[i] if finite else 0.0
            self.critic_norms[i] = min(norms_c[i], self.grad_clip) if finite else 0.0
            if finite and norms_c[i] > self.grad_clip:
                local_critic_clip_count += 1
                self.local_critic_clip_counts[i] = 1
            if finite:
                critic_ready.append(i)
        for i in critic_ready:
            self.opt_c[i].step()

        # Actors: each station's own action from its current policy, every
        # other station's action as executed in the sample.
        frozen = [(p, p.requires_grad) for c in self.critics for p in c.parameters()]
        for p, _ in frozen:
            p.requires_grad_(False)
        try:
            raw = torch.stack([self.actors[i](s[:, i, :]) for i in range(self.n)], dim=1)
            own = self._executed_actions(
                raw, ctx['current_socs'], ctx['capacity_kwh'], ctx['max_power_kw'],
                ctx['soc_step_per_kw'], ctx['kw_per_soc_step'], ctx['inv_max_power_kw'],
                ctx['ev_padding_mask'], use_ste=True, ev_block=ctx['ev_block'],
            )
            total = torch.zeros((), device=device)
            q_means = []
            for i in range(self.n):
                joint = a_exec.clone()
                joint[:, i, :] = own[:, i, :]
                q_mean = self.critics[i](obs, joint.reshape(B, -1)).mean()
                q_means.append(q_mean.detach())
                self._ep_q_raw_local[i].append(q_mean.detach())
                total = total - q_mean
            for opt in self.opt_a:
                opt.zero_grad()
            total.backward()
            actor_norms = [self._clip_and_measure(actor) for actor in self.actors]
        finally:
            for p, flag in frozen:
                p.requires_grad_(flag)
        q_vals, norms_a = torch.stack([torch.stack(q_means), torch.stack(actor_norms)]).cpu().tolist()
        actor_clip_count = 0
        self.actor_losses = [0.0] * self.n
        self.actor_clip_counts = [0] * self.n
        self.actor_source_local_norms_before_clip = [0.0] * self.n
        self.actor_source_global_norms_before_clip = [0.0] * self.n
        self.actor_source_global_ratio = [0.0] * self.n
        self.actor_source_cos = [0.0] * self.n
        self.actor_source_cos_valid = [0] * self.n
        for i in range(self.n):
            finite = math.isfinite(q_vals[i]) and math.isfinite(norms_a[i])
            self.actor_losses[i] = -q_vals[i] if finite else 0.0
            self.actor_norms_before_clip[i] = norms_a[i] if finite else 0.0
            self.actor_norms[i] = min(norms_a[i], self.grad_clip) if finite else 0.0
            if finite and norms_a[i] > self.grad_clip:
                actor_clip_count += 1
                self.actor_clip_counts[i] = 1
            if finite:
                self.opt_a[i].step()

        self._polyak_update_targets()
        self._aggregate_update_logs(local_critic_clip_count, actor_clip_count)

    def _polyak_update_targets(self):
        src, tgt = [], []
        for online, target in (*zip(self.actors, self.t_actors), *zip(self.critics, self.t_critics)):
            for p, tp in zip(online.parameters(), target.parameters()):
                src.append(p.data)
                tgt.append(tp.data)
        torch._foreach_lerp_(tgt, src, self.tau)

    def set_test_mode(self, mode: bool):
        self.test_mode = mode
        self.training = not mode
        for module in (*self.actors, *self.t_actors, *self.critics, *self.t_critics):
            module.eval() if mode else module.train()

    # --- persistence --------------------------------------------------------

    def _training_resume_compatibility(self):
        return {
            "agent_type": type(self).__name__,
            "s_dim": int(self.s_dim),
            "a_dim": int(self.a_dim),
            "n_agents": int(self.n),
            "max_ev_per_station": int(self.max_ev_per_station),
            "batch": int(self.batch),
            "gamma": float(self.gamma),
            "tau": float(self.tau),
            "local_reward_weight": self.local_reward_weight,
            "global_reward_weight": self.global_reward_weight,
            "grad_clip": self.grad_clip,
            "critic_hidden": self.critic_hidden,
            "replay_capacity": int(self.buf.buf_size),
        }

    def training_resume_state_dict(self):
        noise_state = getattr(self.ou_noise, "state", None)
        return {
            "format_version": 1,
            "compatibility": self._training_resume_compatibility(),
            "models": {
                "actors": [m.state_dict() for m in self.actors],
                "target_actors": [m.state_dict() for m in self.t_actors],
                "critics": [m.state_dict() for m in self.critics],
                "target_critics": [m.state_dict() for m in self.t_critics],
            },
            "optimizers": {
                "actors": [o.state_dict() for o in self.opt_a],
                "critics": [o.state_dict() for o in self.opt_c],
            },
            "scalars": {
                "current_episode": int(self.current_episode),
                "epsilon": float(self.epsilon),
                "ou_noise_scale": float(self.ou_noise_scale),
                "update_step": int(self.update_step),
                "warmup_steps": int(self.warmup_steps),
                "test_mode": bool(self.test_mode),
            },
            "noise_state": noise_state.detach().cpu().clone() if torch.is_tensor(noise_state) else None,
            "replay": self.buf.training_resume_state_dict(),
        }

    def load_training_resume_state_dict(self, state):
        if int(state.get("format_version", -1)) != 1:
            raise ValueError(f"unsupported agent resume format: {state.get('format_version')!r}")
        saved, current = state.get("compatibility") or {}, self._training_resume_compatibility()
        if saved != current:
            detail = ", ".join(
                f"{k}: saved={saved.get(k)!r} current={current.get(k)!r}"
                for k in sorted(set(saved) | set(current)) if saved.get(k) != current.get(k)
            )
            raise ValueError(f"agent resume compatibility mismatch: {detail}")
        models, optimizers, scalars = state["models"], state["optimizers"], state["scalars"]
        self._load_module_list(self.actors, models["actors"], "actor")
        self._load_module_list(self.t_actors, models["target_actors"], "target actor")
        self._load_module_list(self.critics, models["critics"], "critic")
        self._load_module_list(self.t_critics, models["target_critics"], "target critic")
        self._load_optimizer_list(self.opt_a, optimizers["actors"], "actor")
        self._load_optimizer_list(self.opt_c, optimizers["critics"], "critic")
        self.current_episode = int(scalars["current_episode"])
        self.epsilon = float(scalars["epsilon"])
        self.ou_noise_scale = float(scalars["ou_noise_scale"])
        self.update_step = int(scalars["update_step"])
        self.warmup_steps = int(scalars["warmup_steps"])
        self.buf.load_training_resume_state_dict(state["replay"])
        noise_state = state.get("noise_state")
        if noise_state is not None:
            current_noise = getattr(self.ou_noise, "state", None)
            if not torch.is_tensor(current_noise) or current_noise.shape != noise_state.shape:
                raise ValueError("exploration-noise state is incompatible")
            current_noise.copy_(noise_state.to(device=current_noise.device))
        self.set_test_mode(False)
        return {
            "current_episode": int(self.current_episode),
            "update_step": int(self.update_step),
            "replay_size": int(self.buf.size),
            "replay_ptr": int(self.buf.ptr),
        }

    def save_checkpoint(self, path, episode):
        self.save_actors(path, episode)
        torch.save({
            "episode": int(episode),
            "agent_type": type(self).__name__,
            "critics": [c.state_dict() for c in self.critics],
            "t_critics": [c.state_dict() for c in self.t_critics],
            "opt_a": [o.state_dict() for o in self.opt_a],
            "opt_c": [o.state_dict() for o in self.opt_c],
        }, os.path.join(path, f"agent_state_ep{episode}.pth"))

    def load_checkpoint(self, path, episode, map_location=None):
        self.load_actors(path, episode, map_location=map_location)
        loaded = {"path": str(path), "episode": int(episode), "actors": True, "critics": False, "optimizers": False}
        bundle_path = os.path.join(path, f"agent_state_ep{episode}.pth")
        if not os.path.exists(bundle_path):
            return loaded
        target = map_location if map_location is not None else device
        try:
            bundle = torch.load(bundle_path, map_location=target, weights_only=True)
        except (TypeError, RuntimeError, pickle.UnpicklingError):
            bundle = torch.load(bundle_path, map_location=target)
        if bundle.get("agent_type") != type(self).__name__:
            # A research-learner bundle has differently shaped critics; the
            # actors are shared, so only they are restored.
            return loaded
        self._load_module_list(self.critics, bundle["critics"], "critic")
        self._load_module_list(self.t_critics, bundle["t_critics"], "target critic")
        loaded["critics"] = True
        self._load_optimizer_list(self.opt_a, bundle["opt_a"], "actor")
        self._load_optimizer_list(self.opt_c, bundle["opt_c"], "critic")
        loaded["optimizers"] = True
        return loaded


def build_marl_agent(s_dim, max_evs_per_station, n_agent, **kwargs):
    """The learner named by Config.MARL_ALGORITHM, with the given shared settings."""
    from Config import MARL_ALGORITHM

    if MARL_ALGORITHM == "maddpg":
        return StandardMADDPG(s_dim, max_evs_per_station, n_agent, **kwargs)
    if MARL_ALGORITHM == "hybrid":
        return MADDPG(s_dim, max_evs_per_station, n_agent, **kwargs)
    raise ValueError(f"unknown EVMA_MARL_ALGORITHM {MARL_ALGORITHM!r}; use 'hybrid' or 'maddpg'")

