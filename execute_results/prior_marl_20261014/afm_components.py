"""MARL-AFMの本文から確認できる演算部品。

対象はSSRN 6229136の3.2節と式13–15。グループ所属、報酬の重み、
ネットワーク幅などは呼出側が与える。原著の一体システムの再現実装ではない。
本線のConfigや学習器は読み込まない。
"""
from __future__ import annotations

import torch
from torch import nn


def group_summary(features: torch.Tensor, group_ids: torch.Tensor,
                  active: torch.Tensor, *, groups: int, reduction: str):
    """接続中のEVを集計し、空のグループを返す。所属の決定は含まない。"""
    if features.ndim != 3 or group_ids.shape != features.shape[:2] or active.shape != group_ids.shape:
        raise ValueError('expected features[B,N,F], group_ids[B,N], active[B,N]')
    if groups < 1 or reduction not in ('mean', 'max'):
        raise ValueError('groups must be positive; reduction must be mean or max')
    if active.dtype != torch.bool or group_ids.dtype != torch.long:
        raise ValueError('active must be bool and group_ids must be int64')
    if not torch.isfinite(features[active]).all():
        raise ValueError('active features must be finite')
    if ((group_ids[active] < 0) | (group_ids[active] >= groups)).any():
        raise ValueError('active group id outside range')
    # 未接続EVのpaddingは、NaNや任意の値でも集計に使わない。
    safe = features.masked_fill(~active.unsqueeze(-1), 0)
    ids = group_ids.masked_fill(~active, 0)
    index = ids.unsqueeze(-1).expand_as(safe)
    counts = features.new_zeros(features.shape[0], groups)
    counts.scatter_add_(1, ids, active.to(features.dtype))
    empty = counts == 0
    if reduction == 'mean':
        summary = features.new_zeros(features.shape[0], groups, features.shape[2])
        summary.scatter_add_(1, index, safe)
        summary = summary / counts.clamp_min(1).unsqueeze(-1)
    else:
        summary = features.new_full((features.shape[0], groups, features.shape[2]), -torch.inf)
        summary.scatter_reduce_(1, index, features.masked_fill(~active.unsqueeze(-1), -torch.inf),
                                reduce='amax', include_self=True)
        summary = summary.masked_fill(empty.unsqueeze(-1), 0)
    return summary, empty


class LocalActor(nn.Module):
    """EV自身と許された共有情報を入力する。MLPの幅は適用側の指定。"""

    def __init__(self, *, observation_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(observation_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, action_dim), nn.Tanh())

    def forward(self, observation: torch.Tensor, active: torch.Tensor):
        if observation.shape[:-1] != active.shape or active.dtype != torch.bool:
            raise ValueError('observation and active dimensions must match')
        if not torch.isfinite(observation[active]).all():
            raise ValueError('active observations must be finite')
        safe = observation.masked_fill(~active.unsqueeze(-1), 0)
        return self.net(safe).masked_fill(~active.unsqueeze(-1), 0)


class AttentionCritic(nn.Module):
    """集約情報とグループ情報に多頭attentionを適用するQ値の部品。

    state_actionとgroup_state_actionに状態と行動を含める設計は適用側の選択。
    原著はQ(s,a)の行動の埋め込み方法と、最終poolingを指定していない。
    この部品では集約tokenのattention出力をQ値に変換する。
    """

    def __init__(self, *, aggregate_dim: int, group_dim: int, hidden_dim: int, heads: int):
        super().__init__()
        if hidden_dim <= 0 or heads <= 0 or hidden_dim % heads:
            raise ValueError('hidden_dim must be a positive multiple of heads')
        self.aggregate = nn.Linear(aggregate_dim, hidden_dim)
        self.group = nn.Linear(group_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.output = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, 1))

    def forward(self, state_action: torch.Tensor, group_state_action: torch.Tensor,
                empty: torch.Tensor):
        if (state_action.ndim != 2 or group_state_action.ndim != 3 or
                empty.shape != group_state_action.shape[:2] or empty.dtype != torch.bool or
                state_action.shape[0] != group_state_action.shape[0]):
            raise ValueError('expected aggregate[B,F], groups[B,K,F], empty[B,K]')
        if (not torch.isfinite(state_action).all() or
                not torch.isfinite(group_state_action[~empty]).all()):
            raise ValueError('nonempty critic inputs must be finite')
        clean = group_state_action.masked_fill(empty.unsqueeze(-1), 0)
        tokens = torch.cat([self.aggregate(state_action).unsqueeze(1), self.group(clean)], dim=1)
        mask = torch.cat([empty.new_zeros(empty.shape[0], 1), empty], dim=1)
        values, weights = self.attention(tokens, tokens, tokens, key_padding_mask=mask,
                                         need_weights=True, average_attn_weights=False)
        return self.output(values[:, 0]), weights[:, :, 0, :]


def ddpg_target(reward: torch.Tensor, terminated: torch.Tensor,
                next_q: torch.Tensor, *, gamma: float):
    """式13の1段先TD target。targetネットワークへの微分を止める。"""
    if not 0 <= gamma <= 1 or reward.shape != next_q.shape or terminated.shape != reward.shape:
        raise ValueError('gamma and target dimensions must be valid')
    if terminated.dtype != torch.bool:
        raise ValueError('terminated must be bool')
    with torch.no_grad():
        # 終端サンプルでは次の状態のpaddingがNaNでもtargetに使わない。
        future = next_q.masked_fill(terminated, 0)
        if not torch.isfinite(reward).all() or not torch.isfinite(future).all():
            raise ValueError('reward and nonterminal next_q must be finite')
        return reward + gamma * future


@torch.no_grad()
def soft_update(target: nn.Module, online: nn.Module, *, tau: float):
    """式15。名前と形状の一致を確認してからtargetを更新する。"""
    if not 0 <= tau <= 1:
        raise ValueError('tau must be between 0 and 1')
    target_state, online_state = target.state_dict(), online.state_dict()
    if target_state.keys() != online_state.keys() or any(
            target_state[k].shape != online_state[k].shape for k in target_state):
        raise ValueError('online and target network structures must match')
    for name, value in target_state.items():
        source = online_state[name]
        if value.is_floating_point():
            value.lerp_(source, tau)
        else:
            value.copy_(source)
