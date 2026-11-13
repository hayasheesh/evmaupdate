"""MADDPG with the attention critic of MAAC (Iqbal and Sha 2019), for the algorithm comparison.

Everything is StandardMADDPG (training/Agent/standard_maddpg.py) -- station
actors, reward, one discount, target, loss, Gaussian exploration, replay --
except the critic. StandardMADDPG gives each station its own critic over the
concatenated observations and actions of every station. Here one critic gives
every station's value at once, as in MAAC:

  e_i  = g_i(o_i, a_i)                    station i's observation-action encoding
  x_i  = sum_{j != i} alpha_ij v(e_j)     attention over the other stations
         alpha_ij = softmax_j( q(e_i) . k(e_j) / sqrt(d) ), per head
  Q_i  = f_i(e_i, x_i)

g_i and f_i are per station; the key, query and value maps are shared by all
stations, with ATTENTION_HEADS heads. MAAC trains stochastic policies with a
soft actor-critic; here the actors stay the deterministic station actors and
follow dQ_i/da_i as in MADDPG, so the critic is the one thing that changes.
"""
from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .maddpg import device
from .standard_maddpg import StandardMADDPG

ATTENTION_HEADS = 4  # MAAC's default


class AttentionCritic(nn.Module):
    """Q_i for every station i from all stations' observations and actions."""

    def __init__(self, obs_dim: int, act_dim: int, n_agent: int, hid: int, heads: int = ATTENTION_HEADS):
        super().__init__()
        if hid % heads:
            raise ValueError(f"hidden width {hid} is not divisible by {heads} heads")
        self.n, self.obs_dim, self.act_dim = int(n_agent), int(obs_dim), int(act_dim)
        self.heads, self.head_dim = int(heads), hid // heads
        self.encoders = nn.ModuleList(
            nn.Sequential(nn.Linear(obs_dim + act_dim, hid), nn.LeakyReLU(0.01)) for _ in range(self.n)
        )
        self.keys = nn.Linear(hid, hid, bias=False)
        self.queries = nn.Linear(hid, hid, bias=False)
        self.values = nn.Sequential(nn.Linear(hid, hid), nn.LeakyReLU(0.01))
        self.heads_out = nn.ModuleList(
            nn.Sequential(nn.Linear(2 * hid, hid), nn.LeakyReLU(0.01), nn.Linear(hid, 1)) for _ in range(self.n)
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """obs (B, n, obs_dim), act (B, n, act_dim) -> Q (B, n)."""
        batch = obs.shape[0]
        e = torch.stack([self.encoders[i](torch.cat([obs[:, i], act[:, i]], dim=-1)) for i in range(self.n)], dim=1)

        def split(x):
            return x.view(batch, self.n, self.heads, self.head_dim).transpose(1, 2)  # (B, H, n, d)

        q, k, v = split(self.queries(e)), split(self.keys(e)), split(self.values(e))
        logits = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)  # (B, H, n, n)
        self_mask = torch.eye(self.n, dtype=torch.bool, device=obs.device)
        weights = torch.softmax(logits.masked_fill(self_mask, float("-inf")), dim=-1)
        x = (weights @ v).transpose(1, 2).reshape(batch, self.n, -1)  # (B, n, hid)
        return torch.cat([self.heads_out[i](torch.cat([e[:, i], x[:, i]], dim=-1)) for i in range(self.n)], dim=1)


class MAACMADDPG(StandardMADDPG):
    def __init__(self, s_dim, max_evs_per_station, n_agent, *, lr_c=None, **kwargs):
        if lr_c is not None:
            kwargs["lr_c"] = lr_c
        super().__init__(s_dim, max_evs_per_station, n_agent, **kwargs)
        critic = AttentionCritic(int(s_dim), int(max_evs_per_station), n_agent, self.critic_hidden).to(device)
        self.critics = [critic]
        self.t_critics = [copy.deepcopy(critic)]
        self.opt_c = [optim.Adam(critic.parameters(), lr=self.opt_c[0].param_groups[0]["lr"])]

    def update(self):
        if self.test_mode or self.buf.size < self.warmup_steps:
            self._zero_update_logs()
            return
        self.update_step += 1

        (s, s2, a, r_local, _r_global, d, actual_station_powers, actual_ev_power_kw,
         r_global_n, _s2_n, _d_n, _n_eff) = self.buf.sample_with_nstep_global(self.batch, 1, self.gamma)
        ctx = self._build_update_ctx(s, s2, a, r_local, d, actual_station_powers, actual_ev_power_kw)
        a_exec = ctx['a_actual'].masked_fill(ctx['ev_padding_mask'], 0.0)
        critic, t_critic, opt_c = self.critics[0], self.t_critics[0], self.opt_c[0]

        with torch.no_grad():
            next_raw = torch.stack([self.t_actors[i](s2[:, i, :]) for i in range(self.n)], dim=1)
            next_exec = self._executed_actions(
                next_raw, ctx['current_socs_s2'], ctx['max_power_kw_s2'],
                ctx['soc_step_per_kw_s2'], ctx['kw_per_soc_step_s2'], ctx['inv_max_power_kw_s2'],
                ctx['ev_padding_mask_s2'], actor_path=False,
            )
            reward = self.local_reward_weight * r_local + self.global_reward_weight * r_global_n.reshape(-1, 1)
            target = reward + self.gamma * t_critic(s2, next_exec) * (1.0 - d)
            target = torch.nan_to_num_(target, nan=0.0, posinf=0.0, neginf=0.0)

        loss = F.mse_loss(critic(s, a_exec), target)
        opt_c.zero_grad()
        loss.backward()
        critic_norm = self._clip_and_measure(critic)

        # Actors: station i's own action from its current policy, every other
        # station's action as executed in the sample.
        frozen = [(p, p.requires_grad) for p in critic.parameters()]
        loss_value, norm_value = torch.stack([loss.detach(), critic_norm]).cpu().tolist()
        critic_ok = math.isfinite(loss_value) and math.isfinite(norm_value)
        if critic_ok:
            opt_c.step()
        for p, _ in frozen:
            p.requires_grad_(False)
        try:
            raw = torch.stack([self.actors[i](s[:, i, :]) for i in range(self.n)], dim=1)
            own = self._executed_actions(
                raw, ctx['current_socs'], ctx['max_power_kw'],
                ctx['soc_step_per_kw'], ctx['kw_per_soc_step'], ctx['inv_max_power_kw'],
                ctx['ev_padding_mask'], actor_path=True,
            )
            total = torch.zeros((), device=device)
            q_means = []
            for i in range(self.n):
                joint = a_exec.clone()
                joint[:, i, :] = own[:, i, :]
                q_mean = critic(s, joint)[:, i].mean()
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

        clipped = int(critic_ok and norm_value > self.grad_clip)
        self.critic_losses = [loss_value if critic_ok else 0.0] * self.n
        self.critic_norms_before_clip = [norm_value if critic_ok else 0.0] * self.n
        self.critic_norms = [min(norm_value, self.grad_clip) if critic_ok else 0.0] * self.n
        self.local_critic_clip_counts = [clipped] * self.n
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
        self._aggregate_update_logs(clipped, actor_clip_count)

    def _training_resume_compatibility(self):
        return {**super()._training_resume_compatibility(), "critic": "attention", "heads": ATTENTION_HEADS}
