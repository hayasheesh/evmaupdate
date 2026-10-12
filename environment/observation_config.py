"""
Observation feature configuration.
Defines the features included in the observation space, their order, and the
derived observation dimensions used by the actor, critics, and normalizer.
"""

import os

from EnvConfig import BANK_DAY_CLASSES, EV_CHARGER_MAX_POWER_OPTIONS_KW, LOWER_BID_LOOKAHEAD_BLOCKS


def _flag(name: str) -> bool:
    return os.environ.get(name, "1").strip().lower() not in ("0", "false", "no", "off")


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


# ``max_power_kw`` is observed only when EVs can have different ratings; with
# one rating for every EV (the 7.7 kW cap) it would always read 1, and the
# learner takes the rating from EnvConfig instead (EV_FIXED_MAX_POWER_KW).
OBSERVE_EV_MAX_POWER = len(set(float(p) for p in EV_CHARGER_MAX_POWER_OPTIONS_KW)) > 1
EV_FIXED_MAX_POWER_KW = None if OBSERVE_EV_MAX_POWER else float(EV_CHARGER_MAX_POWER_OPTIONS_KW[0])
EV_FEATURE_NAMES = (
    "presence",
    "soc",
    "remaining_time",
    "needed_soc",
    "battery_capacity_kwh",
    *(("max_power_kw",) if OBSERVE_EV_MAX_POWER else ()),
)

# The fleet's own miss on the previous step (``fleet_residual_prev``) is one
# broadcast scalar, the same quantity the point-of-coupling battery reads.
# ``service_day_holiday`` is 1 on a weekend or holiday of the service calendar:
# the class that sets every station's arrival rate, known a day ahead. It is
# observed only when the bid banks hold days other than weekdays
# (EnvConfig.BANK_DAY_CLASSES); with weekdays only it would always read 0.
# ``tracking_band_kw`` is the current block's pass band (10 % of the award) in
# kW, the width a step has to land in.
OBSERVE_SERVICE_DAY_HOLIDAY = set(BANK_DAY_CLASSES) != {"weekday"}
_HOLIDAY_FEATURE = ("service_day_holiday",) if OBSERVE_SERVICE_DAY_HOLIDAY else ()
# Three groups can be left out of the observation of every network (actor,
# local critic, global critic); each defaults to observed.
#   EVMA_OBSERVE_FLEET_RESIDUAL  fleet_residual_prev
#   EVMA_OBSERVE_BID_CONTEXT     instruction_scale_kw and the bid lookahead
#   EVMA_OBSERVE_TRACKING_BAND   tracking_band_kw
OBSERVE_FLEET_RESIDUAL = _flag("EVMA_OBSERVE_FLEET_RESIDUAL")
OBSERVE_BID_CONTEXT = _flag("EVMA_OBSERVE_BID_CONTEXT")
OBSERVE_TRACKING_BAND = _flag("EVMA_OBSERVE_TRACKING_BAND")
_RESIDUAL_FEATURE = ("fleet_residual_prev",) if OBSERVE_FLEET_RESIDUAL else ()
OBSERVED_BID_CONTEXT_FEATURES = BID_CONTEXT_FEATURES if OBSERVE_BID_CONTEXT else ()
_BAND_FEATURE = ("tracking_band_kw",) if OBSERVE_TRACKING_BAND else ()
LOCAL_TAIL_FEATURE_NAMES = (
    *(f"demand_{i}" for i in range(int(LOCAL_DEMAND_STEPS))),
    "market_tracking_enabled",
    *_RESIDUAL_FEATURE,
    *OBSERVED_BID_CONTEXT_FEATURES,
    "step",
    *_HOLIDAY_FEATURE,
    *_BAND_FEATURE,
)
GLOBAL_TAIL_FEATURE_NAMES = (
    "total_power",
    "step",
    *(f"demand_{i}" for i in range(int(GLOBAL_DEMAND_STEPS))),
    "market_tracking_enabled",
    *_RESIDUAL_FEATURE,
    *OBSERVED_BID_CONTEXT_FEATURES,
    *_HOLIDAY_FEATURE,
    *_BAND_FEATURE,
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
