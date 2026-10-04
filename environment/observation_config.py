"""
Observation feature configuration.
Defines the features included in the observation space, their order, and the
derived observation dimensions used by the actor, critics, and normalizer.
"""

try:
    from EnvConfig import (
        USE_SWITCHING_CONSTRAINTS,
        USE_HETEROGENEOUS_EV_PHYSICS,
        DAY_CONTEXT_USE_OBS,
        DAY_CONTEXT_INCLUDE_WEATHER,
        LOWER_BID_CONTEXT_USE_OBS,
        LOWER_BID_LOOKAHEAD_BLOCKS,
        BESS_CONTEXT_USE_OBS,
        LOCAL_USE_FLEET_RESIDUAL,
    )
except ImportError:
    USE_SWITCHING_CONSTRAINTS = False
    USE_HETEROGENEOUS_EV_PHYSICS = False
    DAY_CONTEXT_USE_OBS = False
    DAY_CONTEXT_INCLUDE_WEATHER = True
    LOWER_BID_CONTEXT_USE_OBS = False
    LOWER_BID_LOOKAHEAD_BLOCKS = 0
    BESS_CONTEXT_USE_OBS = False
    LOCAL_USE_FLEET_RESIDUAL = False


LOCAL_DEMAND_STEPS = 1
LOCAL_USE_TRACKING_ENABLED = True
LOCAL_USE_STEP = True

# ``demand_0`` is a ratio against a day-specific instruction envelope.  The
# envelope is required to recover physical kW; without it, identical normalized
# observations can require different aggregate actions and yield different
# absolute-kW rewards. Optional known bid-schedule features follow this scalar.
BID_LOOKAHEAD_FIELDS = ("baseline", "up", "down")
BID_LOOKAHEAD_FEATURES = tuple(
    f"bid_{field}_lookahead_{offset}"
    for offset in range(int(LOWER_BID_LOOKAHEAD_BLOCKS))
    for field in BID_LOOKAHEAD_FIELDS
)
BID_CONTEXT_FEATURES = ("instruction_scale_kw", *BID_LOOKAHEAD_FEATURES)

# Optional previous-step grid-side diagnostics.  These signals are deliberately
# separated when an ablation explicitly enables them, but they are off for the
# learning policy by default: corrected EV SoC already carries the station-side
# physical transition, while these aggregate post-controller values are not
# controlled by any one station actor.
BESS_CONTEXT_FEATURES = (
    "last_raw_actor_total_power_kw",
    "last_raw_actor_residual_kw",
    "last_central_correction_power_kw",
    "last_central_ev_total_power_kw",
    "last_pre_bess_residual_kw",
    "last_bess_power_kw",
    "last_pcc_power_kw",
    "last_post_bess_residual_kw",
    "bess_soc_pct",
)

GLOBAL_DEMAND_STEPS = 1
GLOBAL_USE_STEP = True
GLOBAL_USE_TOTAL_POWER = True


BASE_EV_FEATURES = (
    "presence",
    "soc",
    "remaining_time",
    "needed_soc",
)
PHYSICAL_EV_FEATURES = (
    "battery_capacity_kwh",
    "max_power_kw",
)
SWITCH_EV_FEATURES = (
    "switch_count",
    "last_direction",
)
CALENDAR_CONTEXT_FEATURES = (
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
    "is_holiday",
)
WEATHER_CONTEXT_FEATURES = (
    "temp_norm",
    "apparent_temp_norm",
    "precip_norm",
    "rain_norm",
    "snow_norm",
    "wind_norm",
)


def get_ev_feature_names():
    names = list(BASE_EV_FEATURES)
    if bool(USE_HETEROGENEOUS_EV_PHYSICS):
        names.extend(PHYSICAL_EV_FEATURES)
    if bool(USE_SWITCHING_CONSTRAINTS):
        names.extend(SWITCH_EV_FEATURES)
    return tuple(names)


def get_local_tail_feature_names():
    names = [f"demand_{i}" for i in range(int(LOCAL_DEMAND_STEPS))]
    if LOCAL_USE_TRACKING_ENABLED:
        names.append("market_tracking_enabled")
    if bool(LOCAL_USE_FLEET_RESIDUAL):
        # One broadcast scalar: how far the fleet as a whole missed last step.
        names.append("fleet_residual_prev")
    if bool(LOWER_BID_CONTEXT_USE_OBS):
        names.extend(BID_CONTEXT_FEATURES)
    if bool(BESS_CONTEXT_USE_OBS):
        names.extend(BESS_CONTEXT_FEATURES)
    if LOCAL_USE_STEP:
        names.append("step")
    if bool(DAY_CONTEXT_USE_OBS):
        names.extend(CALENDAR_CONTEXT_FEATURES)
        if bool(DAY_CONTEXT_INCLUDE_WEATHER):
            names.extend(WEATHER_CONTEXT_FEATURES)
    return tuple(names)


def get_global_tail_feature_names():
    names = []
    if GLOBAL_USE_TOTAL_POWER:
        names.append("total_power")
    if GLOBAL_USE_STEP:
        names.append("step")
    names.extend(f"demand_{i}" for i in range(int(GLOBAL_DEMAND_STEPS)))
    if LOCAL_USE_TRACKING_ENABLED:
        names.append("market_tracking_enabled")
    if bool(LOCAL_USE_FLEET_RESIDUAL):
        # One broadcast scalar: how far the fleet as a whole missed last step.
        names.append("fleet_residual_prev")
    if bool(LOWER_BID_CONTEXT_USE_OBS):
        names.extend(BID_CONTEXT_FEATURES)
    if bool(BESS_CONTEXT_USE_OBS):
        names.extend(BESS_CONTEXT_FEATURES)
    return tuple(names)


def ev_block_dim(max_evs):
    return int(max_evs) * EV_FEAT_DIM


def local_obs_dim(max_evs):
    return ev_block_dim(max_evs) + LOCAL_TAIL_DIM


def global_obs_dim(max_evs, n_agent):
    return int(n_agent) * ev_block_dim(max_evs) + GLOBAL_TAIL_DIM


EV_FEATURE_NAMES = get_ev_feature_names()
LOCAL_TAIL_FEATURE_NAMES = get_local_tail_feature_names()
GLOBAL_TAIL_FEATURE_NAMES = get_global_tail_feature_names()

EV_FEAT_DIM = len(EV_FEATURE_NAMES)
LOCAL_TAIL_DIM = len(LOCAL_TAIL_FEATURE_NAMES)
GLOBAL_TAIL_DIM = len(GLOBAL_TAIL_FEATURE_NAMES)
