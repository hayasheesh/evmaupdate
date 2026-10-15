"""MARL-AFM networks; unspecified wiring is documented in method_spec.md."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Specification:
    stations: int = 6
    slots: int = 10
    ev_features: int = 5
    tail: int = 77
    forecast_features: int = 14
    blocks: int = 48
    capacity_scale: float = 120.0
    charger_kw: float = 7.7
    dt_hours: float = 5 / 60
    urgent_steps: int = 12
    episode_steps: int = 288
    gamma: float = .99
    tau: float = .005
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    batch: int = 256
    replay: int = 100000
    warmup: int = 7000
    hidden: int = 128
    heads: int = 4
    clip: float = 5.0
    local_scale: float = 25.085 / 7.7 * 2.2
    following_scale: float = .1
    capacity_scale_reward: float = 1.0
    leader_local_scale: float = 1.0
    seed: int = 141026


def mlp(inputs, outputs, hidden=128, tanh=False):
    layers = [nn.Linear(inputs, hidden), nn.ReLU(), nn.Linear(hidden, hidden),
              nn.ReLU(), nn.Linear(hidden, outputs)]
    if tanh:
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


def summarize(features, ids, active, groups):
    safe = features.masked_fill(~active[..., None], 0)
    member = F.one_hot(ids, groups).to(features.dtype) * active[..., None]
    count = member.sum(1)
    mean = member.transpose(1, 2).bmm(safe) / count.clamp_min(1)[..., None]
    maximum = features.new_full((features.shape[0], groups, features.shape[-1]), -torch.inf)
    maximum.scatter_reduce_(1, ids[..., None].expand_as(features),
                            features.masked_fill(~active[..., None], -torch.inf),
                            reduce='amax', include_self=True)
    maximum = maximum.masked_fill((count == 0)[..., None], 0)
    return mean, maximum, count, member


class DynamicGroups:
    """Stable group identities: station x urgency/deficit category, recomputed each step."""
    def __init__(self, spec):
        self.spec = spec
        self.groups = 3 * spec.stations

    def __call__(self, state, bid):
        sp = self.spec
        ev = state[..., :sp.slots * sp.ev_features].reshape(state.shape[0], -1, sp.ev_features)
        active = ev[..., 0] > .5
        ev = ev.masked_fill(~active[..., None], 0)
        category = torch.where((ev[..., 3] > 0) & (ev[..., 2] * sp.episode_steps <= sp.urgent_steps),
                               0, torch.where(ev[..., 3] > 0, 1, 2))
        station = torch.arange(sp.stations, device=state.device).repeat_interleave(sp.slots)
        ids = 3 * station[None] + category
        mean, maximum, count, member = summarize(ev, ids, active, self.groups)
        group_location = torch.arange(self.groups, device=state.device) // 3 / max(sp.stations - 1, 1)
        group_category = torch.arange(self.groups, device=state.device) % 3 / 2
        group = torch.cat([mean, maximum, count[..., None] / sp.slots,
                           group_location[None, :, None].expand(state.shape[0], -1, -1),
                           group_category[None, :, None].expand(state.shape[0], -1, -1)], -1)
        total = active.sum(1).clamp_min(1)
        average = ev.sum(1) / total[:, None]
        soc_variance = ((ev[..., 1] - average[:, 1:2]).square() * active).sum(1) / total
        # The common tail already contains the current command and the committed bid.
        common = torch.cat([state[:, 0, sp.slots * sp.ev_features:], average,
                            soc_variance.sqrt()[:, None],
                            (active.sum(1) / (sp.stations * sp.slots))[:, None], bid], -1)
        selected = group.gather(1, ids[..., None].expand(-1, -1, group.shape[-1]))
        tail = state[..., sp.slots * sp.ev_features:].repeat_interleave(sp.slots, dim=1)
        return dict(ev=ev, active=active, ids=ids, member=member, count=count,
                    incentive=bid[:, 1:2].expand(-1, ev.shape[1]),
                    group=group, common=common, local=torch.cat([ev, tail, selected], -1))


def physical_limits(ev, spec):
    capacity = (ev[..., 4] * spec.capacity_scale).clamp_min(1e-6)
    soc = ev[..., 1].clamp(0, 1)
    lower = torch.maximum(torch.full_like(soc, -1), -soc * capacity / (spec.dt_hours * spec.charger_kw))
    upper = torch.minimum(torch.ones_like(soc), (1 - soc) * capacity / (spec.dt_hours * spec.charger_kw))
    active = ev[..., 0] > .5
    return lower.masked_fill(~active, 0), upper.masked_fill(~active, 0)


class InwardClamp(torch.autograd.Function):
    """Same physical forward and inward gradient convention as the main actuator."""
    @staticmethod
    def forward(ctx, value, lower, upper):
        ctx.save_for_backward(value, lower, upper)
        return value.maximum(lower).minimum(upper)

    @staticmethod
    def backward(ctx, grad):
        value, lower, upper = ctx.saved_tensors
        blocked = ((value > upper) & (grad < 0)) | ((value < lower) & (grad > 0))
        return grad.masked_fill(blocked, 0), None, None


class ControlPolicies(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        self.grouping = DynamicGroups(spec)
        common_dim = spec.tail + spec.ev_features + 2 + 6
        self.reference_actor = mlp(common_dim + 13, 1, spec.hidden)
        # Parameter sharing is explicitly allowed in section 3.2.1.
        self.ev_actor = mlp(spec.ev_features + spec.tail + 13 + 2, 1, 64, tanh=True)

    def forward(self, state, bid, reference=None, noise=0.0):
        f = self.grouping(state, bid)
        if reference is None:
            common = f['common'][:, None].expand(-1, self.grouping.groups, -1)
            reference = torch.sigmoid(self.reference_actor(torch.cat([common, f['group']], -1))).squeeze(-1)
            if noise:
                reference = (reference + noise * torch.randn_like(reference)).clamp(0, 1)
        fraction = reference.gather(1, f['ids'])
        lower, upper = physical_limits(f['ev'], self.spec)
        ev_reference = lower + fraction * (upper - lower)
        inputs = torch.cat([f['local'], ev_reference[..., None], f['incentive'][..., None]], -1)
        raw = self.ev_actor(inputs).squeeze(-1)
        if noise:
            raw = raw + noise * torch.randn_like(raw)
        executed = InwardClamp.apply(raw.clamp(-1, 1), lower, upper).masked_fill(~f['active'], 0)
        f.update(reference=reference, ev_reference=ev_reference, action=executed, inputs=inputs)
        return f


class ControlCritics(nn.Module):
    """Shared group-local utilities -> attention -> unconstrained global mixer."""
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        common_dim = spec.tail + spec.ev_features + 2 + 6
        self.local_critic = mlp(spec.ev_features + spec.tail + 13 + 3, 1, spec.hidden)
        self.global_token = nn.Linear(common_dim, spec.hidden)
        self.group_token = nn.Linear(13 + 3, spec.hidden)
        self.attention = nn.MultiheadAttention(spec.hidden, spec.heads, batch_first=True)
        # No positive-weight constraint: this is the non-monotonic mixer in Fig.4.
        self.mixer = mlp(spec.hidden + 3 * spec.stations, 1, spec.hidden)

    def forward(self, features, action):
        q_ev = self.local_critic(torch.cat([features['inputs'], action[..., None]], -1)).squeeze(-1)
        q_ev = q_ev.masked_fill(~features['active'], 0)
        member = features['member'].transpose(1, 2)
        counts = features['count'].clamp_min(1)[..., None]
        q_group = member.bmm(q_ev[..., None]).squeeze(-1) / counts.squeeze(-1)
        mean_action = member.bmm(action[..., None]) / counts
        groups = torch.cat([features['group'], q_group[..., None], mean_action,
                            features['reference'][..., None]], -1)
        tokens = torch.cat([self.global_token(features['common'])[:, None], self.group_token(groups)], 1)
        empty = features['count'] == 0
        mask = torch.cat([empty.new_zeros(empty.shape[0], 1), empty], 1)
        attended, _ = self.attention(tokens, tokens, tokens, key_padding_mask=mask, need_weights=False)
        global_q = self.mixer(torch.cat([attended[:, 0], q_group], -1))
        return global_q, q_ev, q_group


class BidActor(nn.Module):
    """One leader outputs all six original bid fields per market block."""
    def __init__(self, spec):
        super().__init__()
        self.net = mlp(2 * spec.forecast_features, 6, spec.hidden, tanh=True)

    def forward(self, forecast):
        day = forecast.mean(1, keepdim=True).expand_as(forecast)
        return self.net(torch.cat([forecast, day], -1))


class BidCritic(nn.Module):
    """Day-level upper critic for a committed day-ahead schedule."""
    def __init__(self, spec):
        super().__init__()
        self.token = nn.Linear(spec.forecast_features + 6, spec.hidden)
        self.attention = nn.MultiheadAttention(spec.hidden, spec.heads, batch_first=True)
        self.output = mlp(spec.hidden, 1, spec.hidden)

    def forward(self, forecast, action):
        token = self.token(torch.cat([forecast, action], -1))
        value, _ = self.attention(token, token, token, need_weights=False)
        return self.output(value.mean(1))


class Replay:
    """Lazy CPU ring storage; saved state includes its cursor and sampling RNG."""
    def __init__(self, capacity, seed):
        self.capacity, self.size, self.cursor = capacity, 0, 0
        self.rng = np.random.default_rng(seed)
        self.data = {}

    def add(self, **items):
        values = {k: torch.as_tensor(v).detach().cpu() for k, v in items.items()}
        if not self.data:
            self.data = {k: torch.empty((self.capacity, *v.shape), dtype=v.dtype) for k, v in values.items()}
        if self.data.keys() != values.keys():
            raise ValueError('replay field mismatch')
        for key, value in values.items():
            if self.data[key].shape[1:] != value.shape:
                raise ValueError(f'replay shape mismatch: {key}')
            self.data[key][self.cursor].copy_(value)
        self.cursor = (self.cursor + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch, device):
        ix = torch.from_numpy(self.rng.integers(self.size, size=batch))
        return {k: v[ix].to(device) for k, v in self.data.items()}

    def state_dict(self):
        return dict(capacity=self.capacity, size=self.size, cursor=self.cursor,
                    rng=copy.deepcopy(self.rng.bit_generator.state),
                    data={k: v[:self.size].clone() for k, v in self.data.items()})

    def load_state_dict(self, saved):
        if saved['capacity'] != self.capacity:
            raise ValueError('replay capacity mismatch')
        self.size, self.cursor = saved['size'], saved['cursor']
        self.rng.bit_generator.state = saved['rng']
        self.data = {}
        for key, value in saved['data'].items():
            target = torch.empty((self.capacity, *value.shape[1:]), dtype=value.dtype)
            target[:self.size].copy_(value)
            self.data[key] = target


@torch.no_grad()
def soft_update(target, online, tau):
    for t, s in zip(target.parameters(), online.parameters(), strict=True):
        t.lerp_(s, tau)


def continued_ev_value(next_q, correspondence):
    """An EV's target follows its identity, even when slots or groups change."""
    return next_q.gather(1, correspondence.clamp_min(0)).masked_fill(correspondence < 0, 0)


class AFMLearner:
    def __init__(self, spec=Specification(), device='cpu'):
        self.spec, self.device = spec, torch.device(device)
        self.policies = ControlPolicies(spec).to(self.device)
        self.critics = ControlCritics(spec).to(self.device)
        self.bid_actor = BidActor(spec).to(self.device)
        self.bid_critic = BidCritic(spec).to(self.device)
        self.targets = {name: copy.deepcopy(getattr(self, name)) for name in
                        ('policies', 'critics', 'bid_actor', 'bid_critic')}
        self.optimizers = {name: torch.optim.Adam(getattr(self, name).parameters(),
                            lr=spec.actor_lr if name in ('policies', 'bid_actor') else spec.critic_lr)
                           for name in self.targets}
        self.replay = Replay(spec.replay, spec.seed + 1)
        self.bid_replay = Replay(10000, spec.seed + 2)
        self.updates = self.bid_updates = 0
        self.last = {}

    @torch.no_grad()
    def choose_bid(self, forecast, noise=0):
        forecast = torch.as_tensor(forecast, dtype=torch.float32, device=self.device)[None]
        action = self.bid_actor(forecast)[0]
        if noise:
            action = action + noise * torch.randn_like(action)
        return action.clamp(-1, 1).cpu().numpy()

    @torch.no_grad()
    def act(self, state, bid, noise=0):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)[None]
        bid = torch.as_tensor(bid, dtype=torch.float32, device=self.device)[None]
        f = self.policies(state, bid, noise=noise)
        return f['action'][0].reshape(self.spec.stations, self.spec.slots), f['reference'][0], f['ev_reference'][0]

    def _step(self, name, loss):
        if not torch.isfinite(loss):
            raise FloatingPointError(f'AFM {name} loss is not finite')
        optimizer = self.optimizers[name]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(getattr(self, name).parameters(), self.spec.clip)
        if not torch.isfinite(norm):
            raise FloatingPointError(f'AFM {name} gradient is not finite')
        optimizer.step()
        self.last[name + '_loss'] = float(loss.detach())
        self.last[name + '_grad_before_clip'] = float(norm)

    def update_control(self):
        sp = self.spec
        if self.replay.size < max(sp.warmup, sp.batch):
            return False
        b = self.replay.sample(sp.batch, self.device)
        with torch.no_grad():
            future = self.targets['policies'](b['s2'], b['bid2'])
            next_global, next_ev, _ = self.targets['critics'](future, future['action'])
            ev_future = continued_ev_value(next_ev, b['correspondence'])
            local_target = b['local_reward'] + sp.gamma * ev_future * (1 - b['done'])
            global_target = b['global_reward'] + sp.gamma * next_global * (1 - b['done'])
        current = self.policies(b['s'], b['bid'], reference=b['reference'])
        global_q, ev_q, _ = self.critics(current, b['action'])
        mask = current['active']
        local_loss = ((ev_q - local_target).square() * mask).sum() / mask.sum().clamp_min(1)
        self._step('critics', F.mse_loss(global_q, global_target) + local_loss)
        # DDPG through the complete leader-reference/follower response path.
        for p in self.critics.parameters():
            p.requires_grad_(False)
        try:
            proposed = self.policies(b['s'], b['bid'])
            objective, own_q, _ = self.critics(proposed, proposed['action'])
            own_value = (own_q * proposed['active']).sum() / proposed['active'].sum().clamp_min(1)
            self._step('policies', -objective.mean() - own_value)
        finally:
            for p in self.critics.parameters():
                p.requires_grad_(True)
        for name in ('policies', 'critics'):
            soft_update(self.targets[name], getattr(self, name), sp.tau)
        self.updates += 1
        return True

    def update_bid(self):
        # A daily bid is held for the entire rollout. The daily transition is terminal.
        if self.bid_replay.size < 16:
            return False
        b = self.bid_replay.sample(min(self.spec.batch, self.bid_replay.size), self.device)
        q = self.bid_critic(b['forecast'], b['action'])
        self._step('bid_critic', F.mse_loss(q, b['reward']))
        for p in self.bid_critic.parameters():
            p.requires_grad_(False)
        try:
            action = self.bid_actor(b['forecast'])
            self._step('bid_actor', -self.bid_critic(b['forecast'], action).mean())
        finally:
            for p in self.bid_critic.parameters():
                p.requires_grad_(True)
        for name in ('bid_actor', 'bid_critic'):
            soft_update(self.targets[name], getattr(self, name), self.spec.tau)
        self.bid_updates += 1
        return True

    def state_dict(self):
        return dict(spec=asdict(self.spec), networks={n: getattr(self, n).state_dict() for n in self.targets},
                    targets={n: v.state_dict() for n, v in self.targets.items()},
                    optimizers={n: v.state_dict() for n, v in self.optimizers.items()},
                    replay=self.replay.state_dict(), bid_replay=self.bid_replay.state_dict(),
                    updates=self.updates, bid_updates=self.bid_updates,
                    torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)

    def load_state_dict(self, saved):
        if saved['spec'] != asdict(self.spec):
            raise ValueError('method specification mismatch')
        for name in self.targets:
            getattr(self, name).load_state_dict(saved['networks'][name])
            self.targets[name].load_state_dict(saved['targets'][name])
            self.optimizers[name].load_state_dict(saved['optimizers'][name])
        self.replay.load_state_dict(saved['replay'])
        self.bid_replay.load_state_dict(saved['bid_replay'])
        self.updates, self.bid_updates = saved['updates'], saved['bid_updates']
        torch.set_rng_state(saved['torch_rng'])
        if saved['cuda_rng'] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
