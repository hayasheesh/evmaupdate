"""MAPPO (Yu et al. 2022) on the station actors, for the algorithm comparison.

The actors keep the station actor's network (training/Agent/actor.py) and act on
the same observation; what changes is the learning rule:

  policy   a = tanh(u) per EV slot, u ~ N(mu(o_i), sigma^2); mu is the station
           actor's pre-tanh output, log sigma one learned value per slot.
           Evaluation and the saved actors use the mean, tanh(mu).
  value    one centralized network V(o_1..o_N) giving every station's value
  reward   r_i = w_local * r_local_i + w_global * r_global (STD_MADDPG_*_REWARD_WEIGHT)
  update   on-policy: after each training episode, GAE(gamma, lambda) advantages
           on that episode's 288 steps, then PPO_EPOCHS full-batch epochs of the
           clipped surrogate with an entropy bonus, and a squared-error value loss.

The episode's transitions also go to the replay memory, only so that the
trainer's warmup (Config.WARMUP_STEPS of exploration before learning) ends at
the same point as for the other learners; the warmup episodes are not learned
from. gamma is STD_MADDPG_GAMMA, as for StandardMADDPG. The PPO settings are the
MAPPO paper's defaults for continuous multi-agent tasks and can be overridden
with EVMA_MAPPO_* variables; they are not tuned here.
"""
from __future__ import annotations

import copy
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .actor import Actor
from .maddpg import device
from .standard_maddpg import StandardMADDPG

PPO_LR = float(os.environ.get("EVMA_MAPPO_LR", "5e-4"))
PPO_EPOCHS = int(os.environ.get("EVMA_MAPPO_EPOCHS", "10"))
PPO_CLIP = float(os.environ.get("EVMA_MAPPO_CLIP", "0.2"))
GAE_LAMBDA = float(os.environ.get("EVMA_MAPPO_GAE_LAMBDA", "0.95"))
ENTROPY_COEF = float(os.environ.get("EVMA_MAPPO_ENTROPY_COEF", "0.01"))
MAX_GRAD_NORM = float(os.environ.get("EVMA_MAPPO_MAX_GRAD_NORM", "10.0"))
INITIAL_STD = float(os.environ.get("EVMA_MAPPO_INITIAL_STD", "0.5"))
_LOG_2PI_E = math.log(2.0 * math.pi * math.e)


class GaussianStationActor(Actor):
    """The station actor plus one learned log standard deviation per EV slot."""

    def __init__(self, s_dim, max_evs_per_station, **kwargs):
        super().__init__(s_dim, max_evs_per_station, **kwargs)
        self.log_std = nn.Parameter(torch.full((int(max_evs_per_station),), math.log(INITIAL_STD)))

    def pre_tanh(self, x):
        """The pre-tanh mean per slot and the slot presence, for x shaped (batch, s_dim)."""
        batch = x.shape[0]
        ev_tokens = x[:, :self.ev_state_dim].view(batch, self.max_evs, self.ev_feat_dim)
        presence = (ev_tokens[..., 0:1] > 0.5).float()
        tail = x[:, self.ev_state_dim:self.ev_state_dim + self.tail_dim]
        token_feat = self.ev_token_encoder(ev_tokens)
        denom = presence.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (token_feat * presence).sum(dim=1, keepdim=True) / denom
        count = (denom / float(max(self.max_evs, 1))).expand(-1, self.max_evs, -1)
        token_input = torch.cat([
            token_feat, pooled.expand(-1, self.max_evs, -1),
            tail.unsqueeze(1).expand(-1, self.max_evs, -1), count,
        ], dim=-1)
        return self.ev_action_head(token_input).squeeze(-1), presence.squeeze(-1)


class CentralValue(nn.Module):
    """V_i(o_1..o_N) for every station i."""

    def __init__(self, obs_dim: int, n_agent: int, hid: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim * n_agent, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, n_agent),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs.reshape(obs.shape[0], -1))


class MAPPO(StandardMADDPG):
    def __init__(self, s_dim, max_evs_per_station, n_agent, **kwargs):
        super().__init__(s_dim, max_evs_per_station, n_agent, **kwargs)
        self.actors = [GaussianStationActor(s_dim, max_evs_per_station).to(device) for _ in range(n_agent)]
        self.t_actors = [copy.deepcopy(actor) for actor in self.actors]
        self.opt_a = [optim.Adam(actor.parameters(), lr=PPO_LR) for actor in self.actors]
        value = CentralValue(int(s_dim), n_agent, self.critic_hidden).to(device)
        self.critics = [value]
        self.t_critics = [copy.deepcopy(value)]
        self.opt_c = [optim.Adam(value.parameters(), lr=PPO_LR)]
        self.grad_clip = MAX_GRAD_NORM
        self.ou_noise_scale_initial = self.ou_noise_scale_final = self.ou_noise_scale = 0.0
        self._rollout = self._empty_rollout()
        self._pending = None  # the last sampled action, until its reward arrives

    @staticmethod
    def _empty_rollout():
        return {"obs": [], "u": [], "logp": [], "presence": [], "value": [], "reward": [], "done": [], "next_obs": None}

    # --- acting ---------------------------------------------------------------

    def _distribution(self, obs):
        """Per-station pre-tanh means (n, B, K), log std (n, K) and presence (n, B, K) for obs (B, n, s_dim)."""
        means, presences = zip(*(self.actors[i].pre_tanh(obs[:, i, :]) for i in range(self.n)))
        log_std = torch.stack([actor.log_std for actor in self.actors])
        return torch.stack(means), log_std, torch.stack(presences)

    @staticmethod
    def _log_prob(u, mean, log_std, presence):
        z = (u - mean) * torch.exp(-log_std)
        per_slot = -0.5 * z * z - log_std - 0.5 * math.log(2.0 * math.pi)
        return (per_slot * presence).sum(-1)

    def act(self, state, env=None, noise=True):
        if self.test_mode:
            noise = False
        state = torch.as_tensor(state, dtype=torch.float32, device=device)
        active = state[:, :self.max_evs * self.actors[0].ev_feat_dim].reshape(
            self.n, self.max_evs, self.actors[0].ev_feat_dim)[..., 0] > 0.5
        self.active_slot_mask = active
        self.active_evs = self.active_evs_tensor = active.sum(dim=1).to(dtype=torch.long)
        if env is not None:
            self.env = env
        with torch.no_grad():
            obs = state.unsqueeze(0)
            mean, log_std, presence = self._distribution(obs)
            mean, presence = mean[:, 0], presence[:, 0]
            if not noise:
                return (torch.tanh(mean) * presence).clamp(-1.0, 1.0)
            u = mean + torch.exp(log_std) * torch.randn_like(mean)
            self._pending = {
                "obs": state, "u": u, "logp": self._log_prob(u, mean, log_std, presence),
                "presence": presence, "value": self.critics[0](obs)[0],
            }
            return (torch.tanh(u) * presence).clamp(-1.0, 1.0)

    def cache_experience(self, s, s2, a, r_local, r_global, d, actual_station_powers=None, actual_ev_power_kw=None):
        if self.test_mode:
            return
        super().cache_experience(s, s2, a, r_local, r_global, d, actual_station_powers, actual_ev_power_kw)
        # The trainer caches a step only when its data are finite, so the
        # sampled action joins the rollout here, together with its reward.
        pending, self._pending = self._pending, None
        if pending is None:
            return
        for key, value in pending.items():
            self._rollout[key].append(value)
        r_local = torch.as_tensor(r_local, dtype=torch.float32, device=device).reshape(self.n)
        r_global = torch.as_tensor(r_global, dtype=torch.float32, device=device).reshape(())
        self._rollout["reward"].append(self.local_reward_weight * r_local + self.global_reward_weight * r_global)
        self._rollout["done"].append(torch.as_tensor(d, dtype=torch.float32, device=device).reshape(self.n))
        self._rollout["next_obs"] = torch.as_tensor(s2, dtype=torch.float32, device=device)

    def update(self):
        # Learning happens once per episode, in episode_end.
        self._zero_update_logs()

    def episode_end(self):
        if self.test_mode:
            return
        rollout, self._rollout = self._rollout, self._empty_rollout()
        self._pending = None
        if not rollout["reward"] or self.buf.size < self.warmup_steps:
            return
        self.update_step += 1
        self._ppo(rollout)

    # --- learning -------------------------------------------------------------

    def _advantages(self, rollout):
        values = torch.stack(rollout["value"])  # (T, n)
        rewards = torch.stack(rollout["reward"])
        dones = torch.stack(rollout["done"])
        with torch.no_grad():
            last = self.critics[0](rollout["next_obs"].unsqueeze(0))[0]
        advantages = torch.zeros_like(rewards)
        running = torch.zeros_like(last)
        next_value = last
        for t in range(rewards.shape[0] - 1, -1, -1):
            keep = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * keep - values[t]
            running = delta + self.gamma * GAE_LAMBDA * keep * running
            advantages[t] = running
            next_value = values[t]
        return advantages, advantages + values

    def _ppo(self, rollout):
        obs = torch.stack(rollout["obs"])  # (T, n, s_dim)
        u = torch.stack(rollout["u"])  # (T, n, K)
        old_logp = torch.stack(rollout["logp"])  # (T, n)
        presence = torch.stack(rollout["presence"])
        advantages, returns = self._advantages(rollout)
        acting = presence.sum(-1) > 0  # (T, n): stations with at least one EV
        if acting.any():
            picked = advantages[acting]
            advantages = (advantages - picked.mean()) / (picked.std(unbiased=False) + 1e-8)
        weight = acting.float()
        denom = weight.sum().clamp(min=1.0)
        value_net, opt_v = self.critics[0], self.opt_c[0]
        for _ in range(PPO_EPOCHS):
            mean, log_std, _presence = self._distribution(obs)
            mean = mean.transpose(0, 1)  # (T, n, K)
            logp = self._log_prob(u, mean, log_std.unsqueeze(0), presence)
            ratio = torch.exp(logp - old_logp)
            surrogate = torch.minimum(ratio * advantages, ratio.clamp(1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * advantages)
            entropy = ((0.5 * _LOG_2PI_E + log_std).unsqueeze(0) * presence).sum(-1)
            actor_loss = -((surrogate + ENTROPY_COEF * entropy) * weight).sum() / denom
            value_loss = F.mse_loss(value_net(obs), returns)
            for opt in (*self.opt_a, opt_v):
                opt.zero_grad()
            (actor_loss + value_loss).backward()
            actor_norms = [self._clip_and_measure(actor) for actor in self.actors]
            value_norm = self._clip_and_measure(value_net)
            losses = torch.stack([actor_loss.detach(), value_loss.detach(), value_norm, *actor_norms]).cpu().tolist()
            if all(math.isfinite(x) for x in losses):
                for opt in (*self.opt_a, opt_v):
                    opt.step()
        a_loss, v_loss, v_norm, *a_norms = losses
        self.actor_losses = [a_loss] * self.n
        self.critic_losses = [v_loss] * self.n
        self.critic_norms = [min(v_norm, self.grad_clip)] * self.n
        self.actor_norms = [min(x, self.grad_clip) for x in a_norms]
        clip_count = sum(int(x > self.grad_clip) for x in a_norms)
        with torch.no_grad():
            value_means = returns.mean(0)
        for i in range(self.n):
            self._ep_q_raw_local[i].append(value_means[i])
        self._aggregate_update_logs(int(v_norm > self.grad_clip), clip_count)

    # --- persistence ------------------------------------------------------------

    def _training_resume_compatibility(self):
        return {
            **super()._training_resume_compatibility(),
            "learner": "mappo", "ppo_lr": PPO_LR, "ppo_epochs": PPO_EPOCHS, "ppo_clip": PPO_CLIP,
            "gae_lambda": GAE_LAMBDA, "entropy_coef": ENTROPY_COEF,
        }

    def training_resume_state_dict(self):
        if self._rollout["reward"]:
            raise RuntimeError("MAPPO state is saved between episodes; this episode's rollout is not finished")
        return super().training_resume_state_dict()
