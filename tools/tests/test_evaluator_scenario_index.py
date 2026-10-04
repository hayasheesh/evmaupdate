from tools.evaluator import _evaluation_episode_index


def test_evaluation_scenario_index_is_zero_based_and_checkpoint_independent():
    assert [_evaluation_episode_index(i) for i in range(1, 6)] == [0, 1, 2, 3, 4]
