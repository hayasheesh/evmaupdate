import numpy as np
import pytest
import torch

from EnvConfig import (
    BESS_CHARGE_EFFICIENCY,
    BESS_DISCHARGE_EFFICIENCY,
    BESS_ENERGY_KWH,
    BESS_INITIAL_SOC_PCT,
    BESS_MAX_SOC_PCT,
    BESS_MIN_SOC_PCT,
    BESS_POWER_KW,
    BESS_TARGET_SOC_PCT,
    BESS_TARGET_POWER_CAP_KW,
    EPISODE_STEPS,
    NUM_STATIONS,
    POWER_TO_ENERGY,
    TARGET_DEPLOYMENT_STATIONS,
)
from environment.EVEnv import EVEnv
from environment.central_residual_allocator import (
    StationFlexibilityEnvelope,
    allocate_central_station_targets,
    capped_weighted_waterfill,
)
from environment.normalize import normalize_observation
from environment.observation_config import (
    BESS_CONTEXT_FEATURES,
    EV_FEAT_DIM,
    LOCAL_TAIL_FEATURE_NAMES,
)


@pytest.fixture(scope="module")
def env():
    return EVEnv()


def _configure_bess(
    env,
    *,
    power_kw=BESS_POWER_KW,
    energy_kwh=BESS_ENERGY_KWH,
    initial_soc_pct=BESS_INITIAL_SOC_PCT,
    target_soc_pct=BESS_TARGET_SOC_PCT,
    min_soc_pct=BESS_MIN_SOC_PCT,
    max_soc_pct=BESS_MAX_SOC_PCT,
):
    env.use_residual_bess = True
    env.bess_power_limit_kw = float(power_kw)
    env.bess_energy_capacity_kwh = float(energy_kwh)
    env.bess_initial_soc_pct = float(initial_soc_pct)
    env.bess_target_soc_pct = float(target_soc_pct)
    env.bess_min_soc_pct = float(min_soc_pct)
    env.bess_max_soc_pct = float(max_soc_pct)
    env.bess_charge_efficiency = float(BESS_CHARGE_EFFICIENCY)
    env.bess_discharge_efficiency = float(BESS_DISCHARGE_EFFICIENCY)


def _reset_empty(env, requests, tracking=None, tol_kw=1.0, initial_counts=None):
    request_series = np.zeros(EPISODE_STEPS, dtype=np.float32)
    request_values = np.asarray(requests, dtype=np.float32).reshape(-1)
    request_series[: request_values.size] = request_values
    tracking_series = np.ones(EPISODE_STEPS, dtype=bool)
    if tracking is not None:
        tracking_values = np.asarray(tracking, dtype=bool).reshape(-1)
        tracking_series[: tracking_values.size] = tracking_values
    arrival_counts = np.zeros((NUM_STATIONS, EPISODE_STEPS), dtype=np.int64)
    if initial_counts is None:
        initial_counts = np.zeros(NUM_STATIONS, dtype=np.int64)
    return env.reset(
        net_demand_series=request_series,
        tracking_enabled_series=tracking_series,
        tol_narrow_series=np.full(EPISODE_STEPS, tol_kw, dtype=np.float32),
        arrival_counts_by_station_step=arrival_counts,
        initial_evs_by_station=np.asarray(initial_counts, dtype=np.int64),
    )


def _zero_actions(env):
    return torch.zeros(
        (env.num_stations, env.max_ev_per_station), dtype=torch.float32
    )


def test_default_bess_respects_500_station_power_cap_and_30min_energy() -> None:
    at_target_kw = (
        BESS_POWER_KW * TARGET_DEPLOYMENT_STATIONS / NUM_STATIONS
    )
    assert at_target_kw == pytest.approx(BESS_TARGET_POWER_CAP_KW)
    usable_discharge_kwh = (
        BESS_ENERGY_KWH
        * (BESS_TARGET_SOC_PCT - BESS_MIN_SOC_PCT)
        / 100.0
        * BESS_DISCHARGE_EFFICIENCY
    )
    assert usable_discharge_kwh == pytest.approx(BESS_POWER_KW * 0.5)


def test_actor_reward_and_marl_metrics_stay_pre_bess(env):
    _configure_bess(env)
    _reset_empty(env, [BESS_POWER_KW], tol_kw=1.0)
    env.begin_step()
    obs, _local_reward, global_reward, _done, info = env.apply_action(
        _zero_actions(env)
    )

    assert info["total_ev_transport"] == pytest.approx(0.0)
    assert info["pre_bess_residual_kw"] == pytest.approx(BESS_POWER_KW)
    assert info["bess_power_kw"] == pytest.approx(BESS_POWER_KW)
    assert info["pcc_power_kw"] == pytest.approx(BESS_POWER_KW)
    assert info["post_bess_residual_kw"] == pytest.approx(0.0, abs=1e-6)
    assert info["bess_soc_pct"] > BESS_INITIAL_SOC_PCT
    assert global_reward == pytest.approx(
        env._calculate_balance_reward(BESS_POWER_KW)
    )
    assert info["system_global_reward"] == pytest.approx(
        env._calculate_balance_reward(0.0)
    )

    metrics = env.get_metrics()
    assert metrics["tracking_success_rate"] == pytest.approx(0.0)
    assert metrics["system_tracking_success_rate"] == pytest.approx(100.0)
    assert metrics["pre_bess_mae_kw"] == pytest.approx(BESS_POWER_KW)
    assert metrics["post_bess_mae_kw"] == pytest.approx(0.0, abs=1e-6)

    # Grid-side residual telemetry remains in ``info``/metrics, not in the
    # default actor state. The actor still sees centrally corrected EV SoC.
    assert not set(BESS_CONTEXT_FEATURES).intersection(LOCAL_TAIL_FEATURE_NAMES)
    normalized = normalize_observation(obs)
    assert normalized.shape == obs.shape


def test_central_ev_layer_changes_soc_but_reward_stays_on_raw_actor(env):
    _configure_bess(env)
    # The allocator is off by default now that the proposed system is fully
    # decentralised, so a test about the allocator has to ask for it.
    env.use_central_ev_residual_allocator = True
    initial_counts = np.zeros(NUM_STATIONS, dtype=np.int64)
    initial_counts[0] = 1
    request_kw = BESS_POWER_KW + 11.0
    _reset_empty(env, [request_kw], tol_kw=1.0, initial_counts=initial_counts)
    env.begin_step()
    active_slot = int(torch.nonzero(env.ev_mask[0], as_tuple=False)[0].item())
    soc_before = float(env.soc[0, active_slot].item())

    next_obs, local_reward, _global_reward, _done, info = env.apply_action(
        _zero_actions(env), build_info=False
    )

    assert float(env.soc[0, active_slot].item()) > soc_before
    assert float(info["raw_actor_ev_power_kw"].abs().sum().item()) == pytest.approx(0.0)
    assert float(info["actual_ev_power_kw"].abs().sum().item()) > 0.0
    assert info["raw_actor_total_power_kw"] == pytest.approx(0.0)
    assert info["central_correction_power_kw"] > 0.0
    assert float(local_reward[0].item()) == pytest.approx(0.0, abs=1e-7)
    # The corrected physical result, rather than the raw zero proposal, is the
    # SoC exposed to the actor on the next step.
    compact_slot = 0
    soc_col = compact_slot * EV_FEAT_DIM + 1
    assert float(next_obs[0, soc_col]) == pytest.approx(
        float(env.soc[0, active_slot].item())
    )
    assert info["bess_power_kw"] == pytest.approx(
        request_kw - float(info["total_ev_transport"]), abs=5e-5
    )
    assert info["pcc_power_kw"] == pytest.approx(request_kw)
    assert _global_reward == pytest.approx(
        env._calculate_balance_reward(request_kw)
    )


def test_bess_power_and_energy_constraints_are_physical(env):
    _configure_bess(env, power_kw=10.0, energy_kwh=20.0)
    _reset_empty(env, [100.0], tol_kw=1.0)
    env.begin_step()
    _obs, _lr, _gr, _done, info = env.apply_action(_zero_actions(env))
    assert info["bess_power_kw"] == pytest.approx(10.0)
    assert info["post_bess_residual_kw"] == pytest.approx(90.0)
    assert info["bess_power_limit_hit"] is True
    assert info["bess_energy_limit_hit"] is False

    _configure_bess(
        env,
        power_kw=1000.0,
        energy_kwh=1.0,
        initial_soc_pct=11.0,
        target_soc_pct=50.0,
        min_soc_pct=10.0,
        max_soc_pct=90.0,
    )
    _reset_empty(env, [-100.0], tol_kw=1.0)
    env.begin_step()
    _obs, _lr, _gr, _done, info = env.apply_action(_zero_actions(env))
    expected_kw = (
        0.01 * BESS_DISCHARGE_EFFICIENCY / float(POWER_TO_ENERGY)
    )
    assert info["bess_power_kw"] == pytest.approx(-expected_kw, rel=1e-6)
    assert info["bess_soc_pct"] == pytest.approx(10.0, abs=1e-6)
    assert info["bess_energy_limit_hit"] is True


def test_central_ev_allocator_closes_feasible_residual_without_bess(env):
    _configure_bess(env)
    env.use_residual_bess = False
    # The allocator is off by default now that the proposed system is fully
    # decentralised, so a test about the allocator has to ask for it.
    env.use_central_ev_residual_allocator = True
    initial_counts = np.zeros(NUM_STATIONS, dtype=np.int64)
    initial_counts[0] = 1
    _reset_empty(env, [5.0], tol_kw=0.1, initial_counts=initial_counts)
    env.begin_step()
    _obs, _lr, global_reward, _done, info = env.apply_action(_zero_actions(env))
    assert info["raw_actor_total_power_kw"] == pytest.approx(0.0)
    assert info["total_ev_transport"] == pytest.approx(5.0, abs=1e-4)
    assert info["bess_power_kw"] == pytest.approx(0.0)
    assert info["pcc_power_kw"] == pytest.approx(5.0, abs=1e-4)
    assert global_reward == pytest.approx(env._calculate_balance_reward(5.0))
    metrics = env.get_metrics()
    assert metrics["tracking_success_rate"] == pytest.approx(0.0)
    assert metrics["central_tracking_success_rate"] == pytest.approx(100.0)
    assert metrics["system_tracking_success_rate"] == pytest.approx(100.0)


def test_central_allocator_accepts_station_aggregates_only():
    raw = torch.tensor([2.0, -1.0, 0.0])
    target_up = torch.tensor([3.0, 5.0, 1.0])
    surplus_up = torch.tensor([2.0, 1.0, 4.0])
    remove_charge = torch.tensor([2.0, 0.0, 0.0])
    safe_discharge = torch.tensor([4.0, 3.0, 2.0])
    priority = torch.tensor([1.0, 2.0, 0.5])
    envelope = StationFlexibilityEnvelope(
        raw_power_kw=raw,
        safe_min_power_kw=raw - remove_charge - safe_discharge,
        safe_max_power_kw=raw + target_up + surplus_up,
        toward_target_charge_headroom_kw=target_up,
        surplus_charge_headroom_kw=surplus_up,
        remove_charge_headroom_kw=remove_charge,
        safe_discharge_headroom_kw=safe_discharge,
        toward_target_priority=priority,
        surplus_charge_priority=priority,
        remove_charge_priority=priority,
        safe_discharge_priority=priority,
    )

    station_targets, info = allocate_central_station_targets(
        envelope,
        request_kw=8.0,
        tracking_enabled=True,
    )

    assert station_targets.ndim == 1
    assert station_targets.numel() == 3
    assert float(station_targets.sum().item()) == pytest.approx(8.0, abs=1e-5)
    assert torch.all(station_targets >= envelope.safe_min_power_kw - 1e-6)
    assert torch.all(station_targets <= envelope.safe_max_power_kw + 1e-6)
    assert info["corrected_station_count"] > 0
    # The public central interface consists solely of one-dimensional station
    # tensors: no EV-slot state can cross this boundary.
    assert all(value.ndim == 1 for value in envelope.__dict__.values())


def test_station_local_dispatch_realizes_central_station_target(env):
    _configure_bess(env)
    env.use_residual_bess = False
    initial_counts = np.zeros(NUM_STATIONS, dtype=np.int64)
    initial_counts[:2] = 2
    _reset_empty(env, [17.0], tol_kw=0.1, initial_counts=initial_counts)
    env.begin_step()

    _obs, _lr, _gr, _done, info = env.apply_action(
        _zero_actions(env), build_info=False
    )

    station_targets = info["central_station_target_powers"]
    assert station_targets.shape == (NUM_STATIONS,)
    assert torch.allclose(
        info["station_powers"], station_targets, atol=1e-4, rtol=0.0
    )
    assert info["central_station_target_max_abs_error_kw"] < 1e-4
    assert info["central_allocator_architecture"] == "station_envelope_hierarchical"


def test_free_step_recentres_bess_without_creating_actor_reward(env):
    _configure_bess(env)
    _reset_empty(env, [100.0, 0.0], tracking=[True, False], tol_kw=1.0)
    env.begin_step()
    env.apply_action(_zero_actions(env))
    charged_soc = env._bess_soc_pct()
    assert charged_soc > env.bess_target_soc_pct

    env.begin_step()
    _obs, _lr, global_reward, _done, info = env.apply_action(_zero_actions(env))
    assert global_reward == pytest.approx(0.0)
    assert info["system_global_reward"] == pytest.approx(0.0)
    assert info["bess_power_kw"] < 0.0
    assert env.bess_target_soc_pct <= info["bess_soc_pct"] < charged_soc
    metrics = env.get_metrics()
    assert metrics["tracking_steps"] == 1
    assert metrics["free_steps"] == 1


def test_sort_free_waterfill_is_capped_weighted_and_exact():
    headroom = torch.tensor([1.0, 2.0, 3.0, 4.0])
    weights = torch.tensor([1.0, 4.0, 1.0, 2.0])
    allocation = capped_weighted_waterfill(
        headroom, weights, 7.25, iterations=16
    )
    assert float(allocation.sum().item()) == pytest.approx(7.25, abs=1e-5)
    assert torch.all(allocation >= 0.0)
    assert torch.all(allocation <= headroom + 1e-7)
    assert allocation[1] >= allocation[0]


def test_waterfill_scales_to_5000_slots_without_fleet_sorting():
    headroom = torch.linspace(0.1, 20.0, 5000)
    weights = torch.linspace(0.2, 2.0, 5000)
    target = float(headroom.sum().item()) * 0.61
    allocation = capped_weighted_waterfill(
        headroom, weights, target, iterations=16
    )
    assert float(allocation.sum().item()) == pytest.approx(target, rel=1e-5)
