"""Bid/control/update/resume and an actual EVEnv closed loop."""
import copy
from dataclasses import replace
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pytest
import torch
torch.set_num_threads(1)
from settings import configure, banks
config = configure('aemo')
from networks import AFMLearner, Specification, continued_ev_value
from data import forecast_features, decode_bid, correspondence, committed_action


def learner_and_state():
    spec = replace(Specification(), hidden=16, batch=4, replay=32, warmup=4)
    torch.manual_seed(5)
    learner = AFMLearner(spec)
    state = torch.zeros(6, 127)
    ev = state[:, :50].reshape(6, 10, 5)
    ev[:, :2] = torch.tensor([1., .5, .1, .2, .6])
    return learner, state


def fill(learner, state):
    for _ in range(8):
        action, reference, _ = learner.act(state, np.zeros(6), noise=.1)
        learner.replay.add(s=state, s2=state, bid=torch.zeros(6), bid2=torch.zeros(6),
                           action=action.flatten(), reference=reference,
                           local_reward=torch.full((60,), .7), global_reward=torch.tensor([.2]),
                           done=torch.zeros(1), correspondence=torch.arange(60))
    for i in range(16):
        learner.bid_replay.add(forecast=torch.randn(48, 14), action=torch.randn(48, 6).tanh(),
                               reward=torch.tensor([.1 + i / 20]))


def changed(before, module):
    return any(not torch.equal(x, y) for x, y in zip(before, module.parameters()))


def test_all_networks_and_targets_receive_updates():
    learner, state = learner_and_state()
    fill(learner, state)
    modules = dict(reference=learner.policies.reference_actor, ev=learner.policies.ev_actor,
                   local=learner.critics.local_critic, attention=learner.critics.attention,
                   mixer=learner.critics.mixer, bid_actor=learner.bid_actor,
                   bid_critic=learner.bid_critic)
    before = {k: [p.detach().clone() for p in m.parameters()] for k, m in modules.items()}
    targets = {k: [p.detach().clone() for p in m.parameters()] for k, m in learner.targets.items()}
    assert learner.update_control() and learner.update_bid()
    assert all(changed(before[k], m) for k, m in modules.items())
    assert all(changed(targets[k], m) for k, m in learner.targets.items())
    assert all(np.isfinite(x) for x in learner.last.values())
    assert config.Q_MIX_GLOBAL_WEIGHT == .5 and config.GRAD_CLIP_MAX == 5


def test_resume_reproduces_both_next_updates_exactly():
    a, state = learner_and_state()
    fill(a, state)
    a.update_control(); a.update_bid()
    saved = copy.deepcopy(a.state_dict())
    b, _ = learner_and_state()
    b.load_state_dict(saved)
    a.update_control(); a.update_bid()
    b.update_control(); b.update_bid()
    for name in a.targets:
        for x, y in zip(getattr(a, name).parameters(), getattr(b, name).parameters()):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        for x, y in zip(a.targets[name].parameters(), b.targets[name].parameters()):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
    wrong = AFMLearner(replace(a.spec, local_scale=2 * a.spec.local_scale))
    with pytest.raises(ValueError, match='specification mismatch'):
        wrong.load_state_dict(saved)


def test_ev_identity_survives_reordering_but_not_replacement():
    before = [(0, 1, 10), (0, 2, 20), (0, 3, 30), None]
    after = [(0, 2, 20), (0, 1, 40), (0, 1, 10), None]
    mapping = correspondence(before, after)
    assert mapping.tolist() == [2, 0, -1, -1]
    torch.testing.assert_close(continued_ev_value(torch.tensor([[3., 1000., 7., 99.]]), mapping[None]),
                               torch.tensor([[7., 3., 0., 0.]]))


def test_padding_permutation_and_empty_groups():
    learner, state = learner_and_state()
    baseline, _, _ = learner.act(state, np.zeros(6))
    assert torch.count_nonzero(baseline[:, 2:]) == 0
    order = torch.tensor([1, 0, 2, 3, 4, 5, 6, 7, 8, 9])
    perm = state.clone()
    perm[:, :50] = state[:, :50].reshape(6, 10, 5)[:, order].reshape(6, 50)
    torch.testing.assert_close(learner.act(perm, np.zeros(6))[0], baseline[:, order])
    empty = torch.zeros_like(state)
    f = learner.policies(empty[None], torch.zeros(1, 6))
    q = learner.critics(f, f['action'])[0]
    assert torch.isfinite(q).all() and torch.count_nonzero(f['action']) == 0


def test_mixer_can_have_negative_derivative():
    learner, _ = learner_and_state()
    mixer = learner.critics.mixer
    for p in mixer.parameters():
        torch.nn.init.zeros_(p)
    with torch.no_grad():
        mixer[0].weight[0, -18] = -1
        mixer[0].bias[0] = 2
        mixer[2].weight[0, 0] = 1
        mixer[4].weight[0, 0] = 1
    inputs = torch.zeros(1, 16 + 18, requires_grad=True)
    mixer(inputs).sum().backward()
    assert inputs.grad[0, -18] == -1


def payload():
    from training.bid_bank import BidBank
    bank = BidBank(banks('aemo')[0])
    return bank.load_entry(bank.entries[0])


def test_learned_bid_uses_forecasts_not_solved_plan_and_obeys_envelope():
    learner, _ = learner_and_state()
    p = payload()
    a = forecast_features(p, learner.spec)
    poisoned = dict(p, baseline_plan=np.full(48, 1e9), up_plan=np.full(48, 1e9), down_plan=np.full(48, 1e9))
    np.testing.assert_array_equal(a, forecast_features(poisoned, learner.spec))
    action = learner.choose_bid(a)
    bid = decode_bid(poisoned, action, learner.spec)
    assert not np.array_equal(bid['up_plan'], p['up_plan'])
    assert not bid['result'].feasible and 'certified' not in bid['result'].method
    live = (bid['up_plan'] > 0) | (bid['down_plan'] > 0)
    low = p['assessment_i_sustained_power_min_kw'] + p['assessment_i_control_margin_kw']
    high = p['assessment_i_sustained_power_max_kw'] - p['assessment_i_control_margin_kw']
    assert np.all((bid['baseline_plan'] - bid['up_plan'])[live] >= low[live] - 1e-6)
    assert np.all((bid['baseline_plan'] + bid['down_plan'])[live] <= high[live] + 1e-6)
    quantity = action.copy(); quantity[:, 0] = .4
    assert not np.array_equal(decode_bid(p, quantity, learner.spec)['baseline_plan'], bid['baseline_plan'])
    prices = action.copy(); prices[:, [1, 4, 5]] = .7
    price_bid = decode_bid(p, prices, learner.spec)
    np.testing.assert_array_equal(price_bid['up_plan'], bid['up_plan'])
    assert not np.array_equal(committed_action(price_bid, learner.spec), committed_action(bid, learner.spec))


def test_actual_ev_env_closed_loop_has_control_feedback_and_replay():
    from driver import episode
    from environment.normalize import configure_observation_normalization
    configure_observation_normalization({'command_scale': 'bank', 'demand_center_kw': 0,
                                         'demand_scale_kw': 145.219, 'instruction_scale_scale_kw': 308.7})
    learner, _ = learner_and_state()
    p = payload()
    row = episode(learner, p, p['activation_scenario_payload'][0], seed=140001,
                  train=True)
    assert row['bid_mode'] == 'learned' and row['capacity_kw_block'] >= 0
    assert learner.updates > 0 and learner.bid_replay.size == 1
    assert all(np.isfinite(value) for value in learner.last.values())
    assert row['local_rate'] is not None and row['tracking_steps'] > 0


def test_common_bid_final_inputs_match_existing_main_case():
    from driver import episode
    from evaluate import final_row
    from training.bid_bank import BidBank
    from training.lower_bid_training import _activation_scenarios_for_day
    from environment.normalize import configure_observation_normalization
    import json
    from settings import ROOT
    model = sorted((ROOT / 'archive').glob('wr_cap77wd_aemoplan_6station_resbid_tanh_ls717_2*'))[-1]
    profile = json.loads((model / 'input/observation_normalization.json').read_text())
    configure_observation_normalization(profile)
    bank = BidBank(banks('aemo')[1]); p = bank.load_entry(bank.entries[0])
    commands, _ = _activation_scenarios_for_day(p['service_date'], p['forecast_seed'] + 512209,
                                               n_scenarios=4, scenario_partition='holdout')
    learner, _ = learner_and_state()
    row = episode(learner, p, commands[0], seed=4000000, train=False, common_bid=True)
    row = final_row(row, 0, 0, 0, 'test', 10000)
    file = ROOT / 'execute_results/paper_eval_20261014/main6_aemo_v2/result_aemo_marl_4cmd_2ev.jsonl'
    ref = next(json.loads(line) for line in file.read_text().splitlines()
               if (json.loads(line)['day'], json.loads(line)['command'], json.loads(line)['ev_seed']) == (0, 0, 0))
    assert row['departed_evs'] == ref['departed_evs']
    for field in ('target_kw', 'tol_kw', 'tracking', 'baseline_kw', 'up_kw', 'down_kw'):
        np.testing.assert_array_equal(row['traces'][field], ref['traces'][field])
    assert row['offered_capacity_kw_block'] == ref['offered_capacity_kw_block']
