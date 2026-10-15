"""未接続EVのpaddingとattentionの対称性・学習勾配を確認する。"""
import copy
import torch
import pytest

from afm_components import AttentionCritic, LocalActor, ddpg_target, group_summary, soft_update


@pytest.mark.parametrize('reduction', ['mean', 'max'])
def test_group_padding_and_empty_group(reduction):
    x = torch.tensor([[[2., 4.], [6., 8.], [float('nan'), 999.]]])
    active = torch.tensor([[True, True, False]])
    ids = torch.tensor([[1, 1, -999]])
    out, empty = group_summary(x, ids, active, groups=3, reduction=reduction)
    expected = torch.tensor([4., 6.] if reduction == 'mean' else [6., 8.])
    torch.testing.assert_close(out[0, 1], expected)
    assert empty.tolist() == [[True, False, True]]
    assert torch.count_nonzero(out[0, [0, 2]]) == 0


def test_group_ev_permutation_and_invalid_membership():
    torch.manual_seed(10)
    x = torch.randn(2, 6, 3)
    ids = torch.tensor([[0, 1, 0, 2, 1, 2]]).expand(2, -1)
    active = torch.ones(2, 6, dtype=torch.bool)
    out, mask = group_summary(x, ids, active, groups=3, reduction='mean')
    order = torch.tensor([4, 0, 5, 2, 1, 3])
    perm, perm_mask = group_summary(x[:, order], ids[:, order], active[:, order],
                                   groups=3, reduction='mean')
    torch.testing.assert_close(out, perm)
    assert torch.equal(mask, perm_mask)
    with pytest.raises(ValueError):
        group_summary(x, ids + 1, active, groups=3, reduction='mean')


def test_attention_masks_empty_groups_and_ignores_group_order():
    torch.manual_seed(11)
    critic = AttentionCritic(aggregate_dim=3, group_dim=4, hidden_dim=16, heads=4)
    agg = torch.randn(2, 3)
    groups = torch.randn(2, 3, 4)
    empty = torch.tensor([[False, True, False], [True, True, True]])
    groups[empty] = float('nan')
    q, weights = critic(agg, groups, empty)
    assert torch.isfinite(q).all()
    torch.testing.assert_close(weights[0, :, 2], torch.zeros(4))
    torch.testing.assert_close(weights[1, :, 1:], torch.zeros(4, 3))
    order = torch.tensor([2, 0, 1])
    perm, _ = critic(agg, groups[:, order], empty[:, order])
    torch.testing.assert_close(q, perm)


def test_actor_receives_gradient_through_group_summary():
    torch.manual_seed(12)
    actor = LocalActor(observation_dim=3, hidden_dim=8, action_dim=1)
    obs = torch.randn(1, 4, 3)
    active = torch.tensor([[True, True, True, False]])
    obs[~active] = float('nan')
    action = actor(obs, active)
    assert action[0, 3].item() == 0
    summary, empty = group_summary(torch.cat([obs.masked_fill(~active.unsqueeze(-1), 0), action], -1),
                                   torch.tensor([[0, 0, 1, -1]]), active,
                                   groups=3, reduction='mean')
    critic = AttentionCritic(aggregate_dim=2, group_dim=4, hidden_dim=8, heads=2)
    q, _ = critic(torch.zeros(1, 2), summary, empty)
    (-q.mean()).backward()
    gradients = [p.grad for p in actor.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in gradients)
    assert sum(g.abs().sum().item() for g in gradients) > 0


def test_terminal_target_and_target_parameter_interpolation():
    reward = torch.tensor([[3.], [4.]], requires_grad=True)
    next_q = torch.tensor([[2.], [float('nan')]], requires_grad=True)
    result = ddpg_target(reward, torch.tensor([[False], [True]]), next_q, gamma=.99)
    torch.testing.assert_close(result, torch.tensor([[4.98], [4.]]))
    assert not result.requires_grad
    online = torch.nn.Linear(2, 1)
    target = copy.deepcopy(online)
    old = {k: v.clone() for k, v in target.state_dict().items()}
    with torch.no_grad():
        for p in online.parameters():
            p.add_(10)
    soft_update(target, online, tau=.005)
    for k, v in target.state_dict().items():
        torch.testing.assert_close(v, old[k] + .05)
    with pytest.raises(ValueError):
        soft_update(target, torch.nn.Linear(3, 1), tau=.005)
