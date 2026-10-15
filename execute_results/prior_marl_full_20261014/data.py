"""Forecast-to-learned-bid and EV identity adapters for the accepted EVEnv."""
from __future__ import annotations

import copy
import numpy as np
import torch


def forecast_features(payload, spec):
    """Read forecasts and Assessment-I envelopes, never the solved Benders plan."""
    block_rows = []
    ev_bank = payload['bid_ev_scenario_bank']
    lower = np.asarray(payload['assessment_i_sustained_power_min_kw'], float)
    upper = np.asarray(payload['assessment_i_sustained_power_max_kw'], float)
    physical = spec.stations * spec.slots * spec.charger_kw
    for block in range(spec.blocks):
        t = block * (spec.episode_steps // spec.blocks)
        rows = []
        for scenario in ev_bank:
            online = [ev for ev in scenario if ev.arrival_t <= t < ev.departure_t]
            if online:
                rows.append([len(online) / (spec.stations * spec.slots),
                             np.mean([ev.initial_soc for ev in online]),
                             np.std([ev.initial_soc for ev in online]),
                             np.mean([max(ev.target_soc - ev.initial_soc, 0) for ev in online]),
                             np.mean([min(ev.departure_t - t, spec.episode_steps) / spec.episode_steps for ev in online]),
                             np.mean([ev.capacity_kwh / spec.capacity_scale for ev in online])])
            else:
                rows.append([0] * 6)
        average = np.mean(rows, axis=0)
        # The accepted task has fixed awards; its price fields are exogenous constants.
        # They remain in the leader's six-dimensional action, without invented clearing.
        clock = 2 * np.pi * block / spec.blocks
        block_rows.append([*average, lower[block] / physical, upper[block] / physical,
                           np.sin(clock), np.cos(clock), 1.0, 1.0, 1.0,
                           float(np.min(np.asarray(rows)[:, 0]))])
    result = np.asarray(block_rows, np.float32)
    if result.shape != (spec.blocks, spec.forecast_features):
        raise ValueError('forecast feature layout mismatch')
    return result


def decode_bid(payload, raw, spec):
    """Market quantities are bounded by the same forecast Assessment-I envelope."""
    from training.blockwise_bid import assessment_i_bounds_with_control_margin, minimum_bid_quantity_kw
    from training.lower_bid_training import SubmittedBidResult
    raw = np.asarray(raw, float)
    if raw.shape != (spec.blocks, 6) or not np.isfinite(raw).all():
        raise ValueError('leader must emit six finite bid fields per block')
    lower, upper = assessment_i_bounds_with_control_margin(
        payload['assessment_i_sustained_power_min_kw'], payload['assessment_i_sustained_power_max_kw'],
        payload['assessment_i_control_margin_kw'])
    unit = (np.clip(raw, -1, 1) + 1) / 2
    baseline = lower + unit[:, 0] * (upper - lower)
    up = unit[:, 2] * (baseline - lower)
    down = unit[:, 3] * (upper - baseline)
    minimum = float(minimum_bid_quantity_kw())
    up = np.where(up >= minimum, up, 0)
    down = np.where(down >= minimum, down, 0)
    baseline = np.where((up > 0) | (down > 0), baseline, 0)
    # Minimal metadata: never retain the source solver's feasibility certification.
    result = SubmittedBidResult(solved=True, feasible=False, status='learned_bid_not_lp_certified',
                            objective_capacity_kw_block=float(np.sum(up + down)),
                            method='marl_afm_learned_bid', n_scenarios=len(payload.get('bid_ev_scenario_bank', [])),
                            mean_baseline_kw=float(baseline.mean()), mean_up_kw=float(up.mean()),
                            mean_down_kw=float(down.mean()), up_pass_rate=float('nan'),
                            down_pass_rate=float('nan'), global_tracking_rate=float('nan'), soc_hit_rate=float('nan'))
    bid = {key: copy.deepcopy(payload[key]) for key in
           ('service_date', 'forecast_seed', 'assessment_band_fraction', 'apply_transition_band',
            'arrival_probabilities_by_station', 'activation_mode') if key in payload}
    bid.update(result=result, baseline_plan=baseline, up_plan=up, down_plan=down,
               submitted_up_plan=up.copy(), submitted_down_plan=down.copy(),
               activation_scenario_payload=[], learned_bid_action=raw.copy(),
               learned_price_fields=unit[:, [1, 4, 5]],
               award_metadata={'mode': 'fixed_full_award', 'price_clearing': 'exogenous_task'})
    return bid


def committed_action(bid, spec):
    """Executed bid context, independent of any unexecuted raw action."""
    physical = spec.stations * spec.slots * spec.charger_kw
    prices = bid.get('learned_price_fields', np.ones((spec.blocks, 3)))
    return np.column_stack([np.asarray(bid['baseline_plan']) / physical, prices[:, 0],
                            np.asarray(bid['up_plan']) / physical,
                            np.asarray(bid['down_plan']) / physical, prices[:, 1], prices[:, 2]]).astype(np.float32)


def identities(env):
    """Use station, EV id and arrival together; EV ids may be recycled."""
    rows = []
    for station in range(env.num_stations):
        order = env._get_sorted_active_evs(station)
        ids = env.ev_ids[station, order].cpu().tolist()
        arrivals = env.arrival_step[station, order].cpu().tolist()
        rows.extend([(station, int(eid), int(arrival)) for eid, arrival in zip(ids, arrivals)])
        rows.extend([None] * (env.max_ev_per_station - len(ids)))
    return rows


def correspondence(before, after):
    index = {identity: slot for slot, identity in enumerate(after) if identity is not None}
    return torch.tensor([index.get(identity, -1) if identity is not None else -1 for identity in before], dtype=torch.long)


def individual_rewards(state, executed, reference, spec):
    """Native SoC reward functions applied to individual EVs, plus group following."""
    from environment.EVEnv import soc_progress_shaping, departure_rewards
    sp = spec
    ev = state[:, :sp.slots * sp.ev_features].reshape(-1, sp.ev_features)
    active = ev[:, 0] > .5
    old_soc = ev[:, 1] * 100
    capacity = (ev[:, 4] * sp.capacity_scale).clamp_min(1e-6)
    new_soc = (old_soc + executed.reshape(-1) * sp.charger_kw * sp.dt_hours / capacity * 100).clamp(0, 100)
    target = old_soc + ev[:, 3] * 100
    remaining = ev[:, 2] * sp.episode_steps
    # Invoke the same helper for each EV. Its mean then contains only that EV.
    import EnvConfig as ec
    urgency = 1 + ec.LOCAL_DEFICIT_SHAPING_URGENCY_GAIN * ((ec.LOCAL_DEFICIT_SHAPING_URGENCY_STEPS - remaining) / max(ec.LOCAL_DEFICIT_SHAPING_URGENCY_STEPS, 1)).clamp(0, 1)
    progress = ((target - old_soc).clamp_min(0) - (target - new_soc).clamp_min(0)) / 100
    shaping = (ec.LOCAL_DEFICIT_SHAPING_COEF * progress * urgency).clamp(-ec.LOCAL_DEFICIT_SHAPING_CLIP, ec.LOCAL_DEFICIT_SHAPING_CLIP)
    departure = departure_rewards(new_soc, target).masked_fill(remaining > 1.0001, 0)
    reward = sp.local_scale * (shaping + departure) - sp.following_scale * (executed.reshape(-1) - reference).abs()
    return reward.masked_fill(~active, 0)


def make_episode_inputs(bid, command):
    from training.lower_bid_training import build_fixed_upper_bid_training_episode
    return build_fixed_upper_bid_training_episode(bid, activation_scenario=command)
