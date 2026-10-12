from __future__ import annotations


def test_a_cached_ev_bank_matches_a_fresh_rollout(tmp_path, monkeypatch) -> None:
    import training.lower_bid_training as lbt

    monkeypatch.setattr(lbt, "EV_CANDIDATE_CACHE_DIR", tmp_path)
    kwargs = dict(count=2, seed=73000, seed_offset=0, arrival_probs=None, label="t", service_date="2024-04-02")
    monkeypatch.delenv("EVMA_EV_CANDIDATE_CACHE", raising=False)
    fresh = lbt._sample_ev_scenario_bank(**kwargs)
    assert not list(tmp_path.glob("*.pkl"))

    monkeypatch.setenv("EVMA_EV_CANDIDATE_CACHE", "1")
    first = lbt._sample_ev_scenario_bank(**kwargs)
    assert len(list(tmp_path.glob("*.pkl"))) == 1
    assert not list(tmp_path.glob("*.claim"))

    calls = []
    monkeypatch.setattr(lbt, "_sample_ev_candidate", lambda payload: calls.append(payload) or [])
    second = lbt._sample_ev_scenario_bank(**kwargs)
    assert calls == []
    assert repr(first) == repr(fresh) == repr(second)
