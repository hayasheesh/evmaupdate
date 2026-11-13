"""MADDPG with one actor per EV slot, for the comparison with station actors.

Everything is StandardMADDPG (training/Agent/standard_maddpg.py) -- critics,
reward, target, loss, exploration, replay -- except the actor. The station actor
(training/Agent/actor.py) runs one shared network over the station's EV slots
and gives each slot a summary of the other EVs at the station. Here each EV slot
has its own network with its own parameters, and a slot's network sees only

  that slot's EV features, and
  the station's tail features (instruction, bid lookahead, previous fleet residual),

so no actor sees another EV. With 10 slots at 6 stations that is 60 actors.
They are stored as one module per station (``actors[i]`` holds the station's
slot networks as stacked weights), so saving, loading and evaluation work as
for the station actor. Each station's slot networks share one optimizer and one
gradient-norm clip, as the station actor's layers do.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from Config import ACTOR_HIDDEN_SIZE, LR_ACTOR
from environment.observation_config import EV_FEAT_DIM, LOCAL_TAIL_DIM

from .maddpg import device
from .standard_maddpg import StandardMADDPG


class PerEVActor(nn.Module):
    """Independent networks, one per EV slot, each on [slot EV features, station tail]."""

    def __init__(self, s_dim, max_evs_per_station, hid=ACTOR_HIDDEN_SIZE, init_gain=0.1, **_):
        super().__init__()
        self.s_dim = int(s_dim)
        self.max_evs = int(max_evs_per_station)
        self.ev_feat_dim = EV_FEAT_DIM
        self.ev_state_dim = EV_FEAT_DIM * self.max_evs
        self.tail_dim = LOCAL_TAIL_DIM
        self.init_gain = float(init_gain)
        in_dim = self.ev_feat_dim + self.tail_dim
        h = max(int(hid) // 2, 32)  # the station actor's per-slot width
        k = self.max_evs
        self.w1 = nn.Parameter(torch.empty(k, in_dim, h))
        self.b1 = nn.Parameter(torch.zeros(k, h))
        self.n1_weight = nn.Parameter(torch.ones(k, h))
        self.n1_bias = nn.Parameter(torch.zeros(k, h))
        self.w2 = nn.Parameter(torch.empty(k, h, h))
        self.b2 = nn.Parameter(torch.zeros(k, h))
        self.n2_weight = nn.Parameter(torch.ones(k, h))
        self.n2_bias = nn.Parameter(torch.zeros(k, h))
        self.w3 = nn.Parameter(torch.empty(k, h, 1))
        self.b3 = nn.Parameter(torch.zeros(k, 1))
        for weight in (self.w1, self.w2, self.w3):
            for slot in range(k):
                nn.init.xavier_uniform_(weight.data[slot], gain=self.init_gain)

    def _norm(self, x, weight, bias):
        return F.layer_norm(x, x.shape[-1:]) * weight + bias

    def forward(self, x):
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)
        batch = x.shape[0]
        ev = x[:, :self.ev_state_dim].view(batch, self.max_evs, self.ev_feat_dim)
        presence = (ev[..., 0] > 0.5).float()
        tail = x[:, self.ev_state_dim:self.ev_state_dim + self.tail_dim]
        z = torch.cat([ev, tail.unsqueeze(1).expand(-1, self.max_evs, -1)], dim=-1)
        h = torch.einsum('bki,kih->bkh', z, self.w1) + self.b1
        h = F.leaky_relu(self._norm(h, self.n1_weight, self.n1_bias), 0.1)
        h = torch.einsum('bki,kih->bkh', h, self.w2) + self.b2
        h = F.leaky_relu(self._norm(h, self.n2_weight, self.n2_bias), 0.1)
        out = (torch.einsum('bki,kio->bko', h, self.w3) + self.b3).squeeze(-1)
        actions = torch.tanh(out) * presence
        return actions.squeeze(0) if squeeze else actions


class PerEVMADDPG(StandardMADDPG):
    def __init__(self, s_dim, max_evs_per_station, n_agent, *, lr_a=LR_ACTOR, **kwargs):
        super().__init__(s_dim, max_evs_per_station, n_agent, lr_a=lr_a, **kwargs)
        self.actors = [PerEVActor(s_dim, max_evs_per_station).to(device) for _ in range(n_agent)]
        self.t_actors = [copy.deepcopy(actor) for actor in self.actors]
        self.opt_a = [optim.Adam(actor.parameters(), lr=lr_a) for actor in self.actors]

    def _training_resume_compatibility(self):
        return {**super()._training_resume_compatibility(), "actor": "per_ev_slot"}
