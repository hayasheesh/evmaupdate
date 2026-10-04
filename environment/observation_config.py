"""
Observation feature configuration.
Defines the features included in the observation space, their order, and the
derived observation dimensions used by the actor, critics, and normalizer.
"""

from EnvConfig import LOWER_BID_LOOKAHEAD_BLOCKS


LOCAL_DEMAND_STEPS = 1

# ``demand_0`` is a ratio against a day-specific instruction envelope.  The
# envelope is required to recover physical kW; without it, identical normalized
# observations can require different aggregate actions and yield different
# absolute-kW rewards. The known bid-schedule features follow this scalar.
BID_LOOKAHEAD_FIELDS = ("baseline", "up", "down")
BID_LOOKAHEAD_FEATURES = tuple(
    f"bid_{field}_lookahead_{offset}"
    for offset in range(int(LOWER_BID_LOOKAHEAD_BLOCKS))
    for field in BID_LOOKAHEAD_FIELDS
)
BID_CONTEXT_FEATURES = ("instruction_scale_kw", *BID_LOOKAHEAD_FEATURES)

GLOBAL_DEMAND_STEPS = 1


EV_FEATURE_NAMES = (
    "presence",
    "soc",
    "remaining_time",
    "needed_soc",
    "battery_capacity_kwh",
    "max_power_kw",
)

# The fleet's own miss on the previous step (``fleet_residual_prev``) is one
# broadcast scalar, the same quantity the point-of-coupling battery reads.
LOCAL_TAIL_FEATURE_NAMES = (
    *(f"demand_{i}" for i in range(int(LOCAL_DEMAND_STEPS))),
    "market_tracking_enabled",
    "fleet_residual_prev",
    *BID_CONTEXT_FEATURES,
    "step",
)
GLOBAL_TAIL_FEATURE_NAMES = (
    "total_power",
    "step",
    *(f"demand_{i}" for i in range(int(GLOBAL_DEMAND_STEPS))),
    "market_tracking_enabled",
    "fleet_residual_prev",
    *BID_CONTEXT_FEATURES,
)


def ev_block_dim(max_evs):
    return int(max_evs) * EV_FEAT_DIM


def local_obs_dim(max_evs):
    return ev_block_dim(max_evs) + LOCAL_TAIL_DIM


def global_obs_dim(max_evs, n_agent):
    return int(n_agent) * ev_block_dim(max_evs) + GLOBAL_TAIL_DIM


EV_FEAT_DIM = len(EV_FEATURE_NAMES)
LOCAL_TAIL_DIM = len(LOCAL_TAIL_FEATURE_NAMES)
GLOBAL_TAIL_DIM = len(GLOBAL_TAIL_FEATURE_NAMES)
