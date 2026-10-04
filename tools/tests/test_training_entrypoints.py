from __future__ import annotations

import EnvConfig
import pre_train


def test_pretrain_entrypoint_accepts_no_arguments() -> None:
    args = pre_train.build_parser().parse_args([])

    assert args.episodes > 0
    assert args.train_days > 0
    assert EnvConfig.LOWER_TRAIN_UPPER_BID_ACTIVATION_SCENARIOS == 128
