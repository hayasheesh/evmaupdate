"""The EV candidate bank does not depend on how many processes draw it."""

from training.lower_bid_training import _sample_ev_scenario_bank


def test_parallel_draw_matches_serial_draw():
    kwargs = dict(count=3, seed=4242, seed_offset=0, arrival_probs=None, day_context=None,
                  label="test", service_date="2024-04-02")
    serial = _sample_ev_scenario_bank(workers=1, **kwargs)
    parallel = _sample_ev_scenario_bank(workers=3, **kwargs)
    assert [len(evs) for evs in serial] == [len(evs) for evs in parallel]
    assert serial == parallel
