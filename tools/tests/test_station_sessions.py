"""EVs come from each station's own measured sessions."""

from datetime import date
import random

import numpy as np
import pytest
import torch


def test_calendars_follow_each_country():
    from environment.calendars import HOLIDAY, WEEKDAY, day_class, is_public_holiday

    assert is_public_holiday(date(2024, 11, 4), "JP")          # substitute holiday
    assert day_class("2024-11-28", "JP") == WEEKDAY            # not a Japanese holiday
    assert day_class("2024-11-28", "US") == HOLIDAY            # Thanksgiving
    assert not is_public_holiday(date(2020, 6, 19), "US")      # Juneteenth only from 2021
    assert is_public_holiday(date(2021, 6, 18), "US")          # observed on the Friday
    assert is_public_holiday(date(2019, 4, 19), "NO")          # Good Friday 2019
    assert is_public_holiday(date(2019, 5, 17), "NO")
    assert day_class("2024-08-03", "JP") == HOLIDAY            # Saturday
    with pytest.raises(ValueError):
        is_public_holiday(date(2030, 1, 1), "JP")


def _pool(station_id):
    from EnvConfig import SESSION_MATCH_MIN_CANDIDATES, STATION_SESSION_DIR
    from environment.station_sessions import load_station_pool

    return load_station_pool(str(STATION_SESSION_DIR), station_id, int(SESSION_MATCH_MIN_CANDIDATES))


def test_pool_rate_and_hourly_shape_set_the_daily_arrivals():
    from environment.station_sessions import arrival_probabilities

    pool = _pool("ACN_JPL_ARROYO1")
    for cls, class_pool in pool.pools.items():
        assert class_pool.hour_share.sum() == pytest.approx(1.0)
        probs = arrival_probabilities(pool, cls, growth=3.0, steps=288)
        # One charger sees rate x growth arrivals a day on average.
        assert probs.sum() == pytest.approx(class_pool.rate_per_port_day * 3.0)
    # A workplace garage is much busier on weekdays.
    assert pool.pools["weekday"].rate_per_port_day > 3 * pool.pools["holiday"].rate_per_port_day


def test_session_draw_matches_the_arrival_hour():
    from environment.station_sessions import draw_session

    pool = _pool("COMM VITALITY _ 1104 SPRUCE1")
    rng = random.Random(3)
    sessions = pool.pools["weekday"]
    for step in (100, 150, 200):          # 08:20, 12:30, 16:40
        hour = step // 12
        hours = {int(sessions.arrival_minute[draw_session(pool, "weekday", step, 288, rng)] // 60)
                 for _ in range(50)}
        window = len(sessions.candidates_by_hour[hour])
        assert window >= 20
        assert all(abs(h - hour) <= 2 for h in hours)


def test_only_residential_sessions_carry_their_own_plug_in_soc():
    assert _pool("RESIDENTIAL NORWAY _ OSL_S").has_arrival_soc
    assert not _pool("ACN_CALTECH_GARAGE1").has_arrival_soc


def _env(service_date="2024-04-02", seed=11):
    from environment.EVEnv import EVEnv
    from tools.evaluator import set_env_seed

    set_env_seed(seed)
    env = EVEnv()
    env.reset(net_demand_series=np.zeros(288, dtype=np.float32), service_date=service_date)
    return env


def test_session_attributes_use_delivered_energy_over_the_evs_own_battery():
    env = _env()
    station = env.station_ids.index("ACN_JPL_ARROYO1")
    sessions = env.station_pools[station].pools["weekday"]
    for seed in range(40):
        ev = env._session_attributes(station, "weekday", 96, random.Random(seed))
        assert 0.0 <= ev["init_soc"] <= ev["target_soc"] <= 100.0
        delivered = ev["needed_soc"] * ev["capacity_kwh"] / 100.0
        # The target never asks for more than some session of this station delivered.
        assert delivered <= float(sessions.energy_kwh.max()) + 1e-6
        assert ev["dwell_steps"] >= 1


def test_day_class_changes_the_arrival_rate():
    weekday = _env("2024-04-02")
    holiday = _env("2024-04-06")       # Saturday
    assert weekday.day_class == "weekday" and holiday.day_class == "holiday"
    jpl = weekday.station_ids.index("ACN_JPL_ARROYO1")
    assert weekday.arrival_profiles_by_station[jpl].sum() > 3 * holiday.arrival_profiles_by_station[jpl].sum()


def test_evs_present_at_midnight_come_from_the_previous_days():
    env = _env()
    residential = [i for i, s in enumerate(env.station_ids) if s.startswith("RESIDENTIAL")]
    downtown = [i for i, s in enumerate(env.station_ids) if s.startswith("COMM VITALITY")]
    # Home chargers are occupied overnight; short downtown stays are not.
    assert sum(env.initial_evs_by_station[i] for i in residential) > 0
    assert sum(env.initial_evs_by_station[i] for i in downtown) <= sum(
        env.initial_evs_by_station[i] for i in residential
    )
    assert int(env.ev_mask.sum().item()) == sum(env.initial_evs_by_station)


def test_no_ev_departs_after_the_last_step_and_overnight_targets_are_reachable():
    from EnvConfig import POWER_TO_ENERGY

    env = _env()
    actions = torch.zeros((env.num_stations, env.max_ev_per_station))
    for _ in range(288):
        env.begin_step()
        active = env.ev_mask
        assert int(env.depart[active].max().item() if active.any() else 0) <= 288
        env.apply_action(actions, build_info=False, return_observation=False)
    # Every EV, including the ones staying overnight, has been judged by the
    # last step, so none is left connected.
    assert not bool(env.ev_mask.any().item())

    # The day-end obligation is the SoC the next day assumes at 00:00: charged
    # at the rated power from arrival until the target.
    per_step = 10.0 * POWER_TO_ENERGY * 100.0 / 50.0
    # Arrives at step 283 with 20 %; its 40 % target is reachable in the stay.
    dep, target = env._horizon_obligation(40.0, 300, 50.0, 10.0, 283, 20.0)
    assert dep == 288
    assert target == pytest.approx(20.0 + 6 * per_step, abs=2e-3)   # six steps, 283..288
    assert env._horizon_obligation(80.0, 300, 50.0, 10.0, 100, 20.0) == (288, 80.0)
    # An EV from a previous day counts its charging since that arrival.
    _dep, early = env._horizon_obligation(80.0, 300, 50.0, 10.0, -5, 20.0)
    assert early == pytest.approx(80.0)
    assert env._horizon_obligation(80.0, 200, 50.0, 10.0, 100, 20.0) == (200, 80.0)
    # Never below what rated-power charging after the day needs.
    assert target >= 40.0 - 12 * per_step


def test_bank_days_follow_the_calendar_mix():
    from training.run_after_day_ahead_bid import service_day_payloads, stratified_bank_day_selection

    train, test, info = stratified_bank_day_selection(
        service_day_payloads(2024)["train"], train_count=25, test_count=5
    )
    assert info["train_class_counts"] == {"weekday": 17, "saturday": 3, "sunday_holiday": 5}
    assert info["test_class_counts"] == {"weekday": 3, "saturday": 1, "sunday_holiday": 1}
    assert not {p["date"] for p in train} & {p["date"] for p in test}


def test_evaluation_assessment_i_counts_only_evs_staying_through_the_block():
    from environment.physical_capability import aggregate_physical_capability

    env = _env()
    env.ev_mask.fill_(False)
    env.ev_mask[0, 0] = True
    env.ev_mask[0, 1] = True
    env.ev_capacity_kwh[0, :2] = 60.0
    env.ev_max_power_kw[0, :2] = 11.0
    env.soc[0, :2] = 50.0
    env.target[0, :2] = 50.0
    env.depart[0, 0] = 6        # connected through the block's last step
    env.depart[0, 1] = 3        # leaves before the block ends
    charge, discharge = aggregate_physical_capability(env, block_end_step=6)
    assert charge == pytest.approx(11.0)
    # It must keep its 30 kWh target at the block end, so it cannot discharge.
    assert discharge == pytest.approx(0.0)
    env.target[0, 0] = 10.0
    _charge, discharge = aggregate_physical_capability(env, block_end_step=6)
    assert discharge == pytest.approx(11.0)
