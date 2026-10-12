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
        assert 0.0 <= ev["init_soc"] <= 100.0
        assert ev["target_soc"] <= 80.0
        delivered = max(ev["needed_soc"], 0.0) * ev["capacity_kwh"] / 100.0
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


def test_an_ev_leaving_after_the_last_step_is_never_judged():
    env = _env()
    actions = torch.zeros((env.num_stations, env.max_ev_per_station))
    for _ in range(288):
        env.begin_step()
        env.apply_action(actions, build_info=False, return_observation=False)
    still = env.ev_mask
    # The overnight EVs are still connected, each with its own later departure.
    assert bool(still.any().item())
    assert int(env.depart[still].min().item()) > 288
    # Every placed EV either left during the day and was judged, or is still here.
    assert env.metrics["departing_evs"] + int(still.sum().item()) == len(env.arrival_soc_log)


def test_the_target_is_capped_at_80_percent_even_below_the_plug_in_soc():
    env = _env()
    above = below = 0
    for station in range(env.num_stations):
        for seed in range(60):
            ev = env._session_attributes(station, "weekday", 120, random.Random(1000 * station + seed))
            assert ev["target_soc"] <= 80.0 + 1e-9
            if ev["init_soc"] > 80.0:
                assert ev["target_soc"] == pytest.approx(80.0)
                above += 1
            else:
                assert ev["target_soc"] >= ev["init_soc"] - 1e-9
                below += 1
    assert above > 0 and below > 0


def test_plug_in_soc_follows_the_ev_project_bins_by_location():
    env = _env()
    bins = env.plugin_soc_bins
    assert [s.startswith("RESIDENTIAL") for s in env.station_ids] == env.station_is_residential
    rng = random.Random(5)
    home = np.array([bins.draw(rng, home=True) for _ in range(20000)])
    away = np.array([bins.draw(rng, home=False) for _ in range(20000)])
    for draws, column in ((home, bins.home_cdf), (away, bins.away_cdf)):
        assert draws.min() >= 0.0 and draws.max() < 90.0      # the >= 90 % bin is empty
        shares = np.histogram(draws, bins=np.r_[bins.low, bins.high[-1]])[0] / len(draws)
        assert np.allclose(shares, np.diff(np.r_[0.0, column]), atol=0.01)
    assert 45.0 < home.mean() < 52.0 and 45.0 < away.mean() < 52.0


def test_remaining_time_reads_one_for_a_stay_past_the_day():
    from environment.normalize import normalize_observation
    from environment.observation_config import EV_FEATURE_NAMES

    env = _env()
    obs = env.begin_step()
    norm = normalize_observation(obs)
    k = len(EV_FEATURE_NAMES)
    j = EV_FEATURE_NAMES.index("remaining_time")
    ev_block = norm[:, : env.max_ev_per_station * k].reshape(env.num_stations, env.max_ev_per_station, k)
    assert float(ev_block[..., j].max()) <= 1.0
    assert bool((env.depart[env.ev_mask] > 288).any().item())


def test_bank_days_follow_the_calendar_mix():
    from training.run_after_day_ahead_bid import service_day_payloads, stratified_bank_day_selection

    train, test, info = stratified_bank_day_selection(
        service_day_payloads(2024)["train"], train_count=25, test_count=5,
        day_classes=("weekday", "saturday", "sunday_holiday"),
    )
    assert info["mode"] == "stratified_proportional"
    # 2024 under the Japanese calendar with its customary off periods: 237 / 50 / 79 days.
    assert info["train_class_counts"] == {"weekday": 16, "saturday": 3, "sunday_holiday": 6}
    assert info["test_class_counts"] == {"weekday": 3, "saturday": 1, "sunday_holiday": 1}
    assert not {p["date"] for p in train} & {p["date"] for p in test}


def test_bank_days_are_weekdays_only_by_default():
    from training.run_after_day_ahead_bid import service_day_payloads, stratified_bank_day_selection

    train, test, info = stratified_bank_day_selection(
        service_day_payloads(2024)["train"], train_count=25, test_count=5
    )
    assert info["mode"] == "stratified_proportional[weekday]"
    assert info["train_class_counts"] == {"weekday": 25, "saturday": 0, "sunday_holiday": 0}
    assert info["test_class_counts"] == {"weekday": 5, "saturday": 0, "sunday_holiday": 0}
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


def test_customary_off_periods_are_days_off():
    from environment.calendars import HOLIDAY, WEEKDAY, day_class

    assert day_class("2024-01-02", "JP") == HOLIDAY     # Tuesday, not a national holiday
    assert day_class("2024-12-30", "JP") == HOLIDAY     # Monday
    assert day_class("2024-01-04", "JP") == WEEKDAY
    assert day_class("2024-05-01", "JP") == HOLIDAY     # Golden Week
    assert day_class("2024-08-14", "JP") == HOLIDAY     # Obon
    assert day_class("2024-05-07", "JP") == WEEKDAY
    assert day_class("2019-11-29", "US") == HOLIDAY     # Friday after Thanksgiving
    assert day_class("2019-12-26", "US") == HOLIDAY
    assert day_class("2019-12-02", "US") == WEEKDAY
    assert day_class("2019-07-10", "NO") == HOLIDAY     # fellesferie
    assert day_class("2019-04-15", "NO") == HOLIDAY     # Monday of Holy Week
    assert day_class("2019-08-06", "NO") == WEEKDAY


def test_weekday_banks_skip_the_year_end_and_drop_the_holiday_observation():
    from environment.observation_config import LOCAL_TAIL_FEATURE_NAMES, GLOBAL_TAIL_FEATURE_NAMES
    from training.run_after_day_ahead_bid import service_day_payloads, stratified_bank_day_selection

    train, test, info = stratified_bank_day_selection(
        service_day_payloads(2024)["train"], train_count=25, test_count=5
    )
    for d in info["train_dates"] + info["test_dates"]:
        month, day = int(d[5:7]), int(d[8:10])
        assert not ((month == 12 and day >= 29) or (month == 1 and day <= 3)), d
    assert "service_day_holiday" not in LOCAL_TAIL_FEATURE_NAMES
    assert "service_day_holiday" not in GLOBAL_TAIL_FEATURE_NAMES


@pytest.mark.skipif(
    "service_day_holiday" not in __import__("environment.observation_config", fromlist=["x"]).LOCAL_TAIL_FEATURE_NAMES,
    reason="the holiday observation exists only when the banks hold non-weekdays",
)
def test_the_observation_tells_a_holiday_from_a_weekday():
    from environment.normalize import normalize_observation
    from environment.observation_config import LOCAL_TAIL_FEATURE_NAMES

    position = len(LOCAL_TAIL_FEATURE_NAMES) - 1 - LOCAL_TAIL_FEATURE_NAMES[::-1].index("service_day_holiday")
    offset = len(LOCAL_TAIL_FEATURE_NAMES) - position
    for service_date, expected in (("2024-04-02", 0.0), ("2024-08-03", 1.0)):
        obs = normalize_observation(_env(service_date)._get_obs())
        assert torch.all(obs[:, -offset] == expected)


def test_the_observation_carries_the_current_pass_band():
    from environment.normalize import AG_REQUEST_SCALE, normalize_observation
    from environment.observation_config import LOCAL_TAIL_FEATURE_NAMES
    from environment.EVEnv import EVEnv
    from tools.evaluator import set_env_seed

    set_env_seed(11)
    env = EVEnv()
    band = np.full(288, 30.0, dtype=np.float32)
    band[6:] = 60.0
    env.reset(net_demand_series=np.zeros(288, dtype=np.float32), tol_narrow_series=band, service_date="2024-04-02")
    offset = len(LOCAL_TAIL_FEATURE_NAMES) - LOCAL_TAIL_FEATURE_NAMES.index("tracking_band_kw")
    for step, kw in ((1, 30.0), (7, 60.0)):
        while env.step_count < step:
            obs = env.begin_step()
            if env.step_count < step:
                env.apply_action(np.zeros((env.num_stations, env.max_ev_per_station), dtype=np.float32))
        assert torch.allclose(normalize_observation(obs)[:, -offset], torch.tensor(kw / AG_REQUEST_SCALE))


def test_the_difference_reward_scores_a_station_against_its_share():
    env = _env()
    n = env.num_stations
    safe_min = torch.full((n,), -10.0)
    safe_max = torch.full((n,), 10.0)
    request = float(n)                   # every station's share is 1 kW
    station = torch.ones(n)
    station[0] = 3.0                     # station 0 overshoots by 2 kW
    total = float(station.sum())
    fleet = env._calculate_balance_reward(abs(request - total))
    rewards = env._tracking_difference_rewards(request, total, station, safe_min, safe_max, fleet).cpu()
    # Station 0 made the fleet worse than its share would have; the others did their share.
    assert rewards[0] < 0
    assert torch.allclose(rewards[1:], torch.zeros(n - 1), atol=1e-6)


def test_the_fleet_residual_can_be_shown_per_station(monkeypatch):
    import environment.EVEnv as ev_module
    from environment.observation_config import LOCAL_TAIL_FEATURE_NAMES

    env = _env()
    env.last_raw_actor_residual_kw = 70.0
    offset = len(LOCAL_TAIL_FEATURE_NAMES) - LOCAL_TAIL_FEATURE_NAMES.index("fleet_residual_prev")
    full = float(env._get_obs()[0, -offset])
    monkeypatch.setattr(ev_module, "FLEET_RESIDUAL_OBS_PER_STATION", True)
    assert float(env._get_obs()[0, -offset]) == pytest.approx(full / env.num_stations)
