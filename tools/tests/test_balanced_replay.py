"""Balanced replay: offline (pretrain) region is preserved and mixed in."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.Agent.replay_buffer import ReplayBuffer


N_AGENTS = 3
MAX_EVS = 4


def _cache_one(buf: ReplayBuffer, value: float) -> None:
    """Store one transition whose reward encodes which region it came from."""
    s = torch.full((N_AGENTS, 6), value, dtype=torch.float32)
    a = torch.full((N_AGENTS, MAX_EVS), value, dtype=torch.float32)
    buf.cache(
        s,
        s.clone(),
        a,
        torch.full((N_AGENTS,), value, dtype=torch.float32),
        torch.tensor([value], dtype=torch.float32),
        torch.zeros(N_AGENTS, dtype=torch.float32),
        actual_station_powers=torch.zeros(N_AGENTS, dtype=torch.float32),
        actual_ev_soc_changes=a.clone(),
    )


def _filled_buffer(cap: int, n_offline: int) -> ReplayBuffer:
    buf = ReplayBuffer(cap=cap)
    for _ in range(n_offline):
        _cache_one(buf, 1.0)  # offline transitions carry reward 1.0
    return buf


def test_offline_region_is_not_overwritten_by_online_writes():
    buf = _filled_buffer(cap=64, n_offline=20)
    assert buf.mark_offline_region(ratio_initial=0.5, ratio_final=0.5, decay_steps=0)
    assert buf.offline_size == 20

    # Write more online transitions than the remaining capacity so the online
    # region has to wrap; the offline slots must survive.
    for _ in range(200):
        _cache_one(buf, 0.0)

    offline_rewards = buf.r_global[: buf.offline_size]
    assert torch.all(offline_rewards == 1.0), "offline region was overwritten"
    assert buf.ptr >= buf.offline_size, "write pointer re-entered the offline region"


def test_minibatch_mixes_both_regions_at_the_configured_ratio():
    buf = _filled_buffer(cap=4096, n_offline=500)
    buf.mark_offline_region(ratio_initial=0.5, ratio_final=0.5, decay_steps=0)
    for _ in range(500):
        _cache_one(buf, 0.0)

    batch = 512
    _s, _s2, _a, _rl, r_global, *_rest = buf.sample(batch)
    offline_share = float((r_global.flatten() == 1.0).float().mean())
    assert 0.4 < offline_share < 0.6, f"offline share {offline_share:.3f} off target 0.5"


def test_offline_share_decays_toward_final_ratio():
    buf = _filled_buffer(cap=8192, n_offline=200)
    buf.mark_offline_region(ratio_initial=0.8, ratio_final=0.2, decay_steps=1000)
    assert buf.offline_sample_ratio == pytest.approx(0.8)

    for _ in range(500):
        _cache_one(buf, 0.0)
    midpoint = buf.offline_sample_ratio
    assert 0.4 < midpoint < 0.6, f"ratio {midpoint:.3f} should be about halfway"

    for _ in range(600):
        _cache_one(buf, 0.0)
    assert buf.offline_sample_ratio == pytest.approx(0.2, abs=1e-6)


def test_plain_buffer_behaviour_is_unchanged_without_an_offline_region():
    buf = _filled_buffer(cap=128, n_offline=50)
    assert buf.offline_size == 0
    assert buf._mixed_start_indices(32, 1) is None

    before_ptr = buf.ptr
    for _ in range(100):
        _cache_one(buf, 0.0)
    # Plain wrap-around: pointer cycles through the whole buffer.
    assert buf.size == 128
    assert buf.ptr == (before_ptr + 100) % 128


def test_snapshot_round_trip_through_maddpg(tmp_path):
    """save_replay_snapshot -> load_replay_snapshot seeds the offline region."""
    from EnvConfig import MAX_EV_PER_STATION, NUM_STATIONS
    from environment.observation_config import EV_FEAT_DIM, LOCAL_TAIL_DIM
    from training.Agent.maddpg import MADDPG

    s_dim = MAX_EV_PER_STATION * EV_FEAT_DIM + LOCAL_TAIL_DIM

    def store(buf, value):
        s = torch.full((NUM_STATIONS, s_dim), value, dtype=torch.float32)
        a = torch.full((NUM_STATIONS, MAX_EV_PER_STATION), value, dtype=torch.float32)
        buf.cache(
            s,
            s.clone(),
            a,
            torch.full((NUM_STATIONS,), value, dtype=torch.float32),
            torch.tensor([value], dtype=torch.float32),
            torch.zeros(NUM_STATIONS, dtype=torch.float32),
            actual_station_powers=torch.zeros(NUM_STATIONS, dtype=torch.float32),
            actual_ev_soc_changes=a.clone(),
        )

    saver = MADDPG(s_dim=s_dim, max_evs_per_station=MAX_EV_PER_STATION, n_agent=NUM_STATIONS)
    for _ in range(120):
        store(saver.buf, 1.0)
    ckpt = tmp_path / "TEST120"
    ckpt.mkdir()
    assert saver.save_replay_snapshot(str(ckpt), 120, max_transitions=120)

    loader = MADDPG(s_dim=s_dim, max_evs_per_station=MAX_EV_PER_STATION, n_agent=NUM_STATIONS)
    assert loader.load_replay_snapshot(str(ckpt), 120) == 120
    loader.buf.mark_offline_region(ratio_initial=0.5, ratio_final=0.5, decay_steps=0)
    for _ in range(120):
        store(loader.buf, 0.0)

    assert torch.all(loader.buf.r_global[:120] == 1.0), "inherited data was overwritten"
    _s, _s2, _a, _rl, r_global, *_rest = loader.buf.sample(256)
    share = float((r_global.flatten() == 1.0).float().mean())
    assert 0.4 < share < 0.6, f"offline share {share:.3f} off target 0.5"


def test_missing_snapshot_returns_zero_instead_of_raising(tmp_path):
    """A pretrain archive without a snapshot must fall back, not crash."""
    from EnvConfig import MAX_EV_PER_STATION, NUM_STATIONS
    from environment.observation_config import EV_FEAT_DIM, LOCAL_TAIL_DIM
    from training.Agent.maddpg import MADDPG

    s_dim = MAX_EV_PER_STATION * EV_FEAT_DIM + LOCAL_TAIL_DIM
    agent = MADDPG(s_dim=s_dim, max_evs_per_station=MAX_EV_PER_STATION, n_agent=NUM_STATIONS)
    empty = tmp_path / "TEST999"
    empty.mkdir()
    assert agent.load_replay_snapshot(str(empty), 999) == 0
    assert agent.buf.offline_size == 0


def test_nstep_sampling_windows_stay_inside_their_region():
    buf = _filled_buffer(cap=2048, n_offline=300)
    buf.mark_offline_region(ratio_initial=0.5, ratio_final=0.5, decay_steps=0)
    for _ in range(300):
        _cache_one(buf, 0.0)

    idxs = buf._mixed_start_indices(256, 4)
    assert idxs is not None
    offline_idx = idxs[idxs < buf.offline_size]
    online_idx = idxs[idxs >= buf.offline_size]
    # An n-step window must not run off the end of its own region.
    assert torch.all(offline_idx + 4 <= buf.offline_size + 1)
    assert torch.all(online_idx + 4 <= buf.size + 1)
