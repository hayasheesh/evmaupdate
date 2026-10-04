"""
EnvConfig.py - environment and EV-scenario configuration.

Keep simulator, demand-scaling, arrival, day-context, and station-scenario
settings here.  `Config.py` re-exports these names for older modules, but new
environment/market code should import from `EnvConfig` directly.
"""

from __future__ import annotations

import os


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "yes", "on")


# The study targets a 500-station deployment and scales market parameters by
# NUM_STATIONS / TARGET_DEPLOYMENT_STATIONS, so the station count has to be
# settable to sweep it. Raising it reuses the station roster in cycle (see
# PER_STATION_SESSION_IDS below), so added stations repeat existing stations.
NUM_STATIONS = int(os.environ.get("EVMA_NUM_STATIONS", 7))
TARGET_DEPLOYMENT_STATIONS = int(os.environ.get(
    "EVMA_TARGET_DEPLOYMENT_STATIONS",
    500,
))
# The tracking-reward kW constants below were set at 7 stations. The fleet's
# command and band grow with the station count, so they scale by N / 7.
_REWARD_KW_SCALE = NUM_STATIONS / 7.0


# --- Environment rewards and metrics ---
LOCAL_R_SOC_HIT = 1.0
GLOBAL_BALANCE_REWARD = 1
# Continuous bounded tracking reward used by the lower controller. Assessment
# pass/fail still uses the per-step market tolerance; this reward only supplies
# a stable learning signal toward smaller residual power error. At zero error it
# is +GLOBAL_BALANCE_REWARD and it approaches -GLOBAL_BALANCE_REWARD smoothly.
GLOBAL_BALANCE_REWARD_MODE = os.environ.get(
    "EVMA_GLOBAL_BALANCE_REWARD_MODE", "bounded_absolute_error"
).strip().lower()
GLOBAL_BALANCE_REWARD_ERROR_SCALE_KW = float(os.environ.get(
    "EVMA_GLOBAL_BALANCE_REWARD_ERROR_SCALE_KW", str(150.0 * _REWARD_KW_SCALE)
))
# Where the bounded tracking reward stops bending and continues as a straight
# line, in kW of residual. The tanh form gives gradient inside the tolerance
# band but flattens where large misses are: a 300 kW miss gets about a
# fourteenth of the slope at the origin. Past this point the reward keeps the
# slope it had here, so a deep miss is paid for in proportion to its depth.
# 0 keeps the pure tanh. The origin slope and the zero crossing do not change;
# the cost is that the reward is no longer bounded below.
GLOBAL_BALANCE_REWARD_LINEAR_TAIL_KW = float(os.environ.get(
    "EVMA_GLOBAL_BALANCE_REWARD_LINEAR_TAIL_KW", str(60.0 * _REWARD_KW_SCALE)
))
# Show each station the fleet's total residual from the previous step, as one
# scalar. Stations cannot see each other, so when every one of them hedges,
# the sum falls short and nothing in the observation says so. This is the same
# quantity the point-of-coupling battery reads and carries no per-EV state.
# With it on, a station no longer acts on purely local information: one
# broadcast scalar reaches every actor.
LOCAL_USE_FLEET_RESIDUAL = _env_flag("EVMA_LOCAL_FLEET_RESIDUAL_OBS", "1")
# Which residual the tracking reward is computed on.
#
# Off (default): the raw actor residual, i.e. the EV-fleet power before the
# central residual allocator and the battery. On: the residual at the system
# output, which is the quantity Assessment II judges. Both are computed every
# step; this only decides which one the learner is paid for. The two can move
# in opposite directions.
GLOBAL_REWARD_ON_SYSTEM_OUTPUT = _env_flag(
    "EVMA_GLOBAL_REWARD_ON_SYSTEM_OUTPUT", "0"
)

SOC_HIT_BONUS = 0.5
LOCAL_DEPARTURE_MISS_PENALTY = 0.5
LOCAL_DEPARTURE_DEFICIT_PENALTY_LINEAR = 0.05
# Departure reward shape. "step": LOCAL_R_SOC_HIT + SOC_HIT_BONUS at or above
# target, otherwise -(LOCAL_DEPARTURE_MISS_PENALTY + LINEAR * shortfall in SoC
# points), a 2.05 drop at the target and only 0.05 per point below it.
# "smooth": the hit reward minus LINEAR_F * f + QUADRATIC_F * f**2 for a
# shortfall fraction f = points / 100, continuous at the target and steeper the
# deeper the shortfall, so draining an EV that will miss anyway is not free.
LOCAL_DEPARTURE_REWARD_MODE = os.environ.get("EVMA_LOCAL_DEPARTURE_REWARD_MODE", "step").strip().lower()
LOCAL_DEPARTURE_SMOOTH_LINEAR = float(os.environ.get("EVMA_LOCAL_DEPARTURE_SMOOTH_LINEAR", "6.0"))
LOCAL_DEPARTURE_SMOOTH_QUADRATIC = float(os.environ.get("EVMA_LOCAL_DEPARTURE_SMOOTH_QUADRATIC", "12.0"))
# Apply the execution-time departure force-charging floor inside the training
# and interim-test environment, before anything else sees the actions, so the
# learner trains on the controller it will run with. The floor and its slack
# are those of training.system_controller.apply_force_charging (the evaluation
# pipeline passes 0.1 kWh). The per-point penalty is charged to the station for
# every SoC point the floor adds on top of what its actor asked for.
TRAIN_FORCE_CHARGING = _env_flag("EVMA_TRAIN_FORCE_CHARGING", "0")
TRAIN_FORCE_SLACK_KWH = float(os.environ.get("EVMA_TRAIN_FORCE_SLACK_KWH", "0.1"))
LOCAL_FORCED_PENALTY_PER_POINT = float(os.environ.get("EVMA_LOCAL_FORCED_PENALTY_PER_POINT", "0.0"))
# Learn each station's local value as the sum of per-EV values. The environment
# splits the station's local reward over the EVs present at the step (each EV's
# SoC shaping share and its own departure reward; station-level terms equally),
# and each EV's value is bootstrapped from the same EV at the next step, so an
# EV's action is credited with that EV's outcome only. Actors, the global
# critic and execution are unchanged.
LOCAL_CRITIC_PER_EV = _env_flag("EVMA_LOCAL_CRITIC_PER_EV", "0")
# Local SoC reward. "legacy": the station shaping (soc_progress_shaping) plus the
# departure hit/miss reward. "potential": every EV is paid, every step, the fall
# of its urgency-weighted shortfall,
#   r = coef * (u(k_t) * d_t - gamma * u(k_t - 1) * d_{t+1}),
# d the shortfall below target as a fraction, k the steps left before
# departure and u(k) = 1 + gain * clamp((window - k) / window, 0, 1). Closing
# the shortfall pays more the nearer departure is, and a shortfall left
# standing inside the window costs more each step, so over an EV's stay the
# rewards add up (discounted) to coef * u * d at arrival minus
# coef * (1 + gain) * d at departure: waiting cannot earn more than charging.
# The departure keeps only miss_penalty for leaving below target at all, since
# the SoC hit rate counts misses, not points. gamma is the local critics'
# discount (Config.GAMMA, EVMA_GAMMA).
LOCAL_REWARD_MODE = os.environ.get("EVMA_LOCAL_REWARD_MODE", "legacy").strip().lower()
LOCAL_POTENTIAL_COEF = float(os.environ.get("EVMA_LOCAL_POTENTIAL_COEF", "1.5"))
LOCAL_POTENTIAL_URGENCY_GAIN = float(os.environ.get("EVMA_LOCAL_POTENTIAL_URGENCY_GAIN", "3.0"))
LOCAL_POTENTIAL_WINDOW_STEPS = int(os.environ.get("EVMA_LOCAL_POTENTIAL_WINDOW_STEPS", "48"))
LOCAL_POTENTIAL_MISS_PENALTY = float(os.environ.get("EVMA_LOCAL_POTENTIAL_MISS_PENALTY", "0.5"))
LOCAL_POTENTIAL_GAMMA = float(os.environ.get("EVMA_GAMMA", "0.985"))
if LOCAL_REWARD_MODE not in ("legacy", "potential"):
    raise ValueError(f"EVMA_LOCAL_REWARD_MODE must be 'legacy' or 'potential', got {LOCAL_REWARD_MODE!r}")
# Each station's actor emits one normalized scalar for its feasible total
# power. environment/station_allocation splits that total by laxity (charge
# the least slack first, discharge the most slack first). The split uses only
# that station's EVs. Acting and learning apply the same split; force charging
# may then override individual EV actions at evaluation time.
STATION_RULE_ALLOCATION = _env_flag("EVMA_STATION_RULE_ALLOCATION", "0")
# SoC is kept by the station rule instead of being learned: every EV is first
# given the departure force-charging floor (the rule evaluation applies,
# training.system_controller.apply_force_charging, with TRAIN_FORCE_SLACK_KWH),
# and the actor's scalar only moves the station's charging or discharging above
# those floors. The local critics are then not trained; actors learn from the
# global critic alone. Q_MIX_GLOBAL_WEIGHT is left as it is.
STATION_SOC_FLOOR = _env_flag("EVMA_STATION_SOC_FLOOR", "0")
if STATION_SOC_FLOOR and not STATION_RULE_ALLOCATION:
    raise ValueError("EVMA_STATION_SOC_FLOOR needs EVMA_STATION_RULE_ALLOCATION=1")
if LOCAL_DEPARTURE_REWARD_MODE not in ("step", "smooth"):
    raise ValueError(f"EVMA_LOCAL_DEPARTURE_REWARD_MODE must be 'step' or 'smooth', got {LOCAL_DEPARTURE_REWARD_MODE!r}")

LOCAL_DEFICIT_SHAPING_COEF = 0.50
LOCAL_DEFICIT_SHAPING_CLIP = float(os.environ.get("EVMA_LOCAL_SHAPING_CLIP", "0.08"))
LOCAL_DEFICIT_SHAPING_URGENCY_GAIN = float(os.environ.get("EVMA_LOCAL_URGENCY_GAIN", "1.0"))
LOCAL_DEFICIT_SHAPING_URGENCY_STEPS = 48
# What the urgency weight on a shortfall is measured against. "time": steps
# left before departure. "laxity": steps left minus the steps full-power
# charging would need to reach the target, so an EV with hours to spare weighs
# little however soon it leaves, and one that can barely still make it weighs
# the most. Laxity uses only the EV's own SoC, target, capacity, rating and
# departure, all in the station's observation.
LOCAL_URGENCY_BASIS = os.environ.get("EVMA_LOCAL_URGENCY_BASIS", "time").strip().lower()
if LOCAL_URGENCY_BASIS not in ("time", "laxity"):
    raise ValueError(f"EVMA_LOCAL_URGENCY_BASIS must be 'time' or 'laxity', got {LOCAL_URGENCY_BASIS!r}")
# How a station's per-EV SoC progress is combined into its shaping reward.
# "mean" divides by the number of EVs present, so one EV's progress weighs
# 1/N; "sum" gives every EV the same weight however many share the station.
LOCAL_SHAPING_REDUCTION = os.environ.get("EVMA_LOCAL_SHAPING_REDUCTION", "mean").strip().lower()
# Weight on the SoC above target. The deficit term is blind to an EV that has
# reached its target, so charging it further or discharging it back toward the
# target changes nothing. With a positive weight, growing that surplus costs
# and shrinking it pays, so the station takes flexibility from satisfied EVs
# before short ones. 0 leaves the reward as before.
LOCAL_SURPLUS_SHAPING_COEF = float(os.environ.get("EVMA_LOCAL_SURPLUS_SHAPING_COEF", "0.0"))
if LOCAL_SHAPING_REDUCTION not in ("mean", "sum"):
    raise ValueError(f"EVMA_LOCAL_SHAPING_REDUCTION must be 'mean' or 'sum', got {LOCAL_SHAPING_REDUCTION!r}")

TOL_NARROW_METRICS = float(os.environ.get("EVMA_TOL_NARROW", 150.0))
# Constant per-kW slope of the global balance reward outside the tolerance band.
# Same as the published paper: slope = 2 * GLOBAL_BALANCE_REWARD / D with a
# fixed D = 150 kW, so the reward falls +1 -> -1 over 150 kW past the band edge.
# It stays constant even though D is now set per step: 2 * 1 / 150 ~= 0.01333/kW.
GLOBAL_BALANCE_REWARD_SLOPE = float(os.environ.get(
    "EVMA_BALANCE_REWARD_SLOPE", 2.0 * float(GLOBAL_BALANCE_REWARD) / 150.0))
SOC_WIDE = 20


# --- EV physics and episode geometry ---
EV_CAPACITY = 100.0
EPISODE_STEPS = 288

# Per-EV battery capacity and charger rating. When enabled, each arriving EV
# draws its own capacity/power, and the upper bidder picks these up
# automatically (evenv_adapter reads env.ev_capacity_kwh / ev_max_power_kw into
# each EVSpec), so bid feasibility is computed against the same heterogeneous
# fleet the lower controller sees.
#
# The distributions target the fleet an EV VPP would actually dispatch once
# this market participation is routine, not today's installed base. Sales-
# weighted average BEV pack size rose from ~48 kWh (2018) to ~62-63 kWh (2025)
# and is still growing ~5 %/year; the EU average is already ~70 kWh and the US
# ~90 kWh. Dropping the 40 kWh first-generation tier and adding a 120 kWh tier
# gives a mean of ~81 kWh, a defensible near-future fleet.
# Charger ratings likewise assume high-power AC / DC destination charging
# rather than today's 6-7 kW workplace AC, keeping mean power near the
# homogeneous 27.5 kW value so switching regimes does not silently halve the
# fleet's biddable capability.
USE_HETEROGENEOUS_EV_PHYSICS = _env_flag("EVMA_HETEROGENEOUS_EV_PHYSICS", "1")

# --- Target market product -------------------------------------------------
# The product is SECONDARY-2, not tertiary-2. The two differ in how
# assessment-II is judged, which changes every number the bidder produces.
#   secondary-2: per measurement point, >=90 % of points inside the band in
#                each 30-minute block (取引規程 第39条(3))
#   tertiary-2 : the 30-minute AVERAGE output, one check per block
# Do not read the assessment method out of OCCTO committee papers -- the widely
# cited ones are tertiary-2 context. See docs/市場規定と海外事例.md.
# The band is +-10 % of the AWARDED delta-kW, so scaling the bid scales the
# band with it: a narrower bid is not an easier bid. OCCTO WG49 (2024) records
# this as a known market-design issue ("ΔkW落札量が小さいことによるペナルティ
# リスク"), and its proposed relief explicitly excludes DR resources with no
# nameplate rating -- which is what an EV aggregator is.
# Instruction cadence: 5-minute commands on the market evaluation grid.
EV_BATTERY_CAPACITY_OPTIONS_KWH = (50.0, 60.0, 75.0, 90.0, 100.0, 120.0)
EV_BATTERY_CAPACITY_PROBS = (0.10, 0.18, 0.25, 0.22, 0.15, 0.10)
EV_CHARGER_MAX_POWER_OPTIONS_KW = (11.0, 19.2, 27.5, 50.0)
EV_CHARGER_MAX_POWER_PROBS = (0.20, 0.30, 0.35, 0.15)
MAX_EV_POWER_KW = 27.5
EV_CAPACITY_OBS_SCALE_KWH = max(max(EV_BATTERY_CAPACITY_OPTIONS_KWH), EV_CAPACITY)
EV_CHARGER_POWER_OBS_SCALE_KW = max(max(EV_CHARGER_MAX_POWER_OPTIONS_KW), MAX_EV_POWER_KW)
TIME_STEP_MINUTES = 5
POWER_TO_ENERGY = TIME_STEP_MINUTES / 60.0



MAX_EV_PER_STATION = 10
NUM_EVS = 10000

# --- EV sessions ---
# Every station draws its EVs from its own measured charging sessions. One row
# of data/input_EVinfo/station_sessions/<station>.csv is one real session, and
# its arrival time, dwell time and delivered energy are used together, so the
# relation between when a car arrives, how long it stays and how much it takes
# is the station's own. The tables, their sources and the arrival rates are
# built by data/input_EVinfo/build_station_sessions.py; docs/EVデータ.md
# describes them.
STATION_SESSION_DIR = os.path.join(PROJECT_ROOT, "data", "input_EVinfo", "station_sessions")
# Station roster in EVEnv order. NUM_STATIONS takes the first entries and
# repeats the list when it needs more.
STATION_ROSTER = (
    "COMM VITALITY _ 1400 WALNUT1",
    "COMM VITALITY _ 1104 SPRUCE1",
    "COMM VITALITY _ 1500PEARL1",
    "ACN_JPL_ARROYO1",
    "ACN_CALTECH_GARAGE1",
    "RESIDENTIAL NORWAY _ OSL_S",
    "RESIDENTIAL NORWAY _ TRO_R",
    "BOULDER _ CARPENTER PARK1",
    "BOULDER _ N BOULDER REC 1",
)
PER_STATION_SESSION_IDS = tuple(
    STATION_ROSTER[index % len(STATION_ROSTER)] for index in range(NUM_STATIONS)
)
# Arrival rate: the station's sessions per port and day in the service day's
# class (weekday, or weekend/holiday), times this growth factor for a future EV
# population, on each of MAX_EV_PER_STATION chargers. An arrival that finds
# every charger taken is lost.
FUTURE_EV_ARRIVAL_GROWTH_MULTIPLIER = float(
    os.environ.get("EVMA_FUTURE_EV_ARRIVAL_GROWTH", "3.0")
)
# The service day is a day of the Japanese market, so its weekday/holiday class
# follows the Japanese calendar. Each source dataset was classified with its own
# country's calendar when the session tables were built.
SERVICE_CALENDAR_COUNTRY = "JP"
# Bid-bank days are drawn from this calendar year, weekday / Saturday /
# Sunday-or-holiday in the proportions the year has
# (training.run_after_day_ahead_bid.stratified_bank_day_selection).
SERVICE_YEAR = int(os.environ.get("EVMA_SERVICE_YEAR", "2024"))
# An arrival at step t draws a session of the same day class that started in
# the same clock hour; when that hour holds fewer sessions than this, the
# window widens one hour each side until it does.
SESSION_MATCH_MIN_CANDIDATES = int(os.environ.get("EVMA_SESSION_MATCH_MIN_CANDIDATES", "20"))
# The EVs plugged in at 00:00 are the ones the same arrival process leaves
# connected after running this many previous days, without control: each
# charged at its rated power from arrival until its target or midnight.
EV_PREROLL_DAYS = int(os.environ.get("EVMA_EV_PREROLL_DAYS", "2"))
# Plug-in SoC. Residential sessions carry their own (Norway Dataset3
# SoC_start). Every other session draws from this distribution: SoC on arrival
# at the DC fast chargers of the EPFL DESL Level-3 dataset
# (github.com/DESL-EPFL/Level-3-EV-charging-dataset, 1,878 sessions, Apr 2022 -
# Jul 2023), built by data/input_EVinfo/Arrivesoc.py.
EV_SOC_ARRIVAL_DISTRIBUTION_PATH = os.path.join(
    PROJECT_ROOT, "data", "input_EVinfo", "soc_arrival_distribution.csv"
)
# Departure target: plug-in SoC plus the session's delivered energy over the
# EV's own battery, capped at a full battery and at what this fraction of the
# EV's rated power can add during its stay.
EV_TARGET_REACHABLE_POWER_FRACTION = 0.8


# --- Demand input and command scaling ---
DEMAND_ADJUSTMENT_DIR = os.path.join(
    PROJECT_ROOT, "data", "input.demand_fromPJM", "output_5min"
)
# Empirical load-following command sets sharing one scenario interface. What
# each contains and how it is normalised: docs/指令データ.md. An explicit
# EVMA_ACTIVATION_SCENARIO_DIR takes precedence, for reproducing an archived
# experiment.
ACTIVATION_SIGNAL_SET = os.environ.get(
    "EVMA_ACTIVATION_SIGNAL_SET", "aemo_plan_deviation"
).strip().lower()
_ACTIVATION_SIGNAL_DIRS = {
    "aemo_plan_deviation": os.path.join(
        PROJECT_ROOT,
        "data",
        "aemo",
        "nem",
        "command_libraries",
        "plan_deviation_calendar_day",
    ),
    "aemo_plan_deviation_gc": os.path.join(
        PROJECT_ROOT,
        "data",
        "aemo",
        "nem",
        "command_libraries",
        "plan_deviation_gc_calendar_day",
    ),
    "nem": os.path.join(PROJECT_ROOT, "data", "aemo", "nem", "processed_5min", "archive"),
    "aemo_bess_dispatch": os.path.join(
        PROJECT_ROOT,
        "data",
        "aemo",
        "nem",
        "command_libraries",
        "bess_dispatch_calendar_day",
    ),
    "aemo_wdru": os.path.join(
        PROJECT_ROOT,
        "data",
        "aemo",
        "nem",
        "command_libraries",
        "wdru_activation_calendar_day",
    ),
    "elexon_plan_deviation": os.path.join(
        PROJECT_ROOT,
        "data",
        "elexon",
        "command_libraries",
        "battery_plan_deviation_calendar_day",
    ),
    "ercot_plan_deviation": os.path.join(
        PROJECT_ROOT,
        "data",
        "ercot",
        "sced",
        "command_libraries",
        "esr_plan_deviation_calendar_day",
    ),
    "ercot": os.path.join(
        PROJECT_ROOT, "data", "ercot", "sced", "processed_5min", "rtc_b"
    ),
    # AGC 型の比較用。PJM の RegD（tools/build_pjm_regd_library.py）。
    # pjm_regd は実指令で、validation と test の日を最終評価にだけ使う。入札と学習（学習の途中テストを含む）は、
    # 実指令の train の日だけから作った疑似指令 pjm_regd_phase_shift を使う。
    "pjm_regd": os.path.join(
        PROJECT_ROOT,
        "data",
        "pjm",
        "command_libraries",
        "regd_calendar_day",
    ),
    "pjm_regd_phase_shift": os.path.join(
        PROJECT_ROOT,
        "data",
        "pjm",
        "command_libraries",
        "regd_phase_shift_train_calendar_day",
    ),
}
if ACTIVATION_SIGNAL_SET not in _ACTIVATION_SIGNAL_DIRS:
    raise ValueError(
        "EVMA_ACTIVATION_SIGNAL_SET must be one of "
        f"{sorted(_ACTIVATION_SIGNAL_DIRS)}, got "
        f"{ACTIVATION_SIGNAL_SET!r}"
    )
ACTIVATION_SCENARIO_DIR = os.environ.get("EVMA_ACTIVATION_SCENARIO_DIR", "").strip() or (
    _ACTIVATION_SIGNAL_DIRS[ACTIVATION_SIGNAL_SET]
)
# Fleet power ceiling used as the bid cap and demand-target clamp. Under
# heterogeneous physics each EV draws its own rating, so neither the
# homogeneous MAX_EV_POWER_KW product nor the max-rating product describes the
# fleet: with the current mix the expected aggregate is ~1756 kW, while
# "every EV draws the 50 kW tier" (3500 kW) has effectively zero probability
# over 70 slots. Scale by the mean rating with a headroom factor so the cap is
# a loose-but-reachable bound; the bidder then shrinks it to the deliverable
# width through its own feasibility check.
_MEAN_PER_EV_POWER_KW = sum(
    w * p for w, p in zip(EV_CHARGER_MAX_POWER_OPTIONS_KW, EV_CHARGER_MAX_POWER_PROBS)
)
_PER_EV_POWER_CAP_KW = (
    min(1.25 * _MEAN_PER_EV_POWER_KW, max(EV_CHARGER_MAX_POWER_OPTIONS_KW))
    if USE_HETEROGENEOUS_EV_PHYSICS
    else MAX_EV_POWER_KW
)
PHYSICAL_MAX_POWER_KW = float(NUM_STATIONS * MAX_EV_PER_STATION * _PER_EV_POWER_CAP_KW)
DEMAND_TARGET_MIN_KW = float(os.environ.get("EVMA_DEMAND_MIN_KW", -PHYSICAL_MAX_POWER_KW))
DEMAND_TARGET_MAX_KW = float(os.environ.get("EVMA_DEMAND_MAX_KW", PHYSICAL_MAX_POWER_KW))

# --- Grid-side residual BESS ------------------------------------------------
# This BESS is an independent PCC actuator. It never rewrites an EV action and
# never changes the EV-side learning reward. The physical PCC output and a
# separate set of system metrics use the post-BESS result. During free steps a
# deterministic controller recentres stored energy for the next dispatch block.
USE_RESIDUAL_BESS = _env_flag("EVMA_USE_RESIDUAL_BESS", "1")
# Off while training, and while scoring during training.  The battery never
# rewrites an EV action and never enters the EV-side reward -- the tracking
# reward is the raw actor deviation and the departure reward reads the raw
# actor SoC trajectory -- so on a pretrain or a fine-tune it is arithmetic whose
# only consumer is a reported column.  The final precision evaluation builds its
# own environment and is untouched, and a pipeline that wants the battery still
# asks for it per pipeline.
TRAIN_USE_RESIDUAL_BESS = _env_flag("EVMA_TRAIN_RESIDUAL_BESS", "0")
# The lower policy already observes the EV-side consequence of the rule layer
# through the corrected physical SoC on the next step.  Grid-side residual
# telemetry is not controlled by an individual station actor, and it would
# take 9 of the 13 local tail features for a post-controller that cannot earn
# actor reward.  Available for explicit ablations only.
BESS_CONTEXT_USE_OBS = _env_flag("EVMA_BESS_CONTEXT_OBS", "0")
BESS_TARGET_POWER_CAP_KW = float(os.environ.get(
    "EVMA_BESS_TARGET_POWER_CAP_KW",
    # Keep the 500-station equivalent below the 2 MW special-high-voltage
    # application boundary used by the study.  1999.4 kW also stays below the
    # 1999.5 kW maximum-receiving-power example in TEPCO's application guide.
    "1999.4",
))
_BESS_DEFAULT_POWER_KW = (
    BESS_TARGET_POWER_CAP_KW
    * NUM_STATIONS
    / max(TARGET_DEPLOYMENT_STATIONS, 1)
)
BESS_POWER_KW = float(os.environ.get(
    "EVMA_BESS_POWER_KW",
    str(_BESS_DEFAULT_POWER_KW),
))
BESS_INITIAL_SOC_PCT = float(os.environ.get("EVMA_BESS_INITIAL_SOC_PCT", "50.0"))
BESS_TARGET_SOC_PCT = float(os.environ.get("EVMA_BESS_TARGET_SOC_PCT", "50.0"))
BESS_MIN_SOC_PCT = float(os.environ.get("EVMA_BESS_MIN_SOC_PCT", "10.0"))
BESS_MAX_SOC_PCT = float(os.environ.get("EVMA_BESS_MAX_SOC_PCT", "90.0"))
BESS_CHARGE_EFFICIENCY = float(os.environ.get("EVMA_BESS_CHARGE_EFFICIENCY", "0.95"))
BESS_DISCHARGE_EFFICIENCY = float(os.environ.get("EVMA_BESS_DISCHARGE_EFFICIENCY", "0.95"))
# Default energy is the smallest nameplate capacity that can discharge at the
# full power rating for one 30-minute block from target SoC to minimum SoC.
_BESS_DEFAULT_DISCHARGE_SOC_FRACTION = max(
    (BESS_TARGET_SOC_PCT - BESS_MIN_SOC_PCT) / 100.0,
    1e-9,
)
_BESS_DEFAULT_ENERGY_KWH = (
    BESS_POWER_KW
    * 0.5
    / (_BESS_DEFAULT_DISCHARGE_SOC_FRACTION * BESS_DISCHARGE_EFFICIENCY)
)
BESS_ENERGY_KWH = float(os.environ.get(
    "EVMA_BESS_ENERGY_KWH",
    str(_BESS_DEFAULT_ENERGY_KWH),
))

# Central residual redistribution is not part of the proposed system. It is
# retained only as a central-observation rule baseline: every station obeys a
# centrally assigned power target. The proposed MARL system instead assumes
# that each station owns its connected EV load and makes its own action.
USE_CENTRAL_EV_RESIDUAL_ALLOCATOR = _env_flag(
    "EVMA_USE_CENTRAL_EV_RESIDUAL_ALLOCATOR", "0"
)
CENTRAL_EV_ALLOCATOR_WATERFILL_ITERS = int(os.environ.get(
    "EVMA_CENTRAL_EV_ALLOCATOR_WATERFILL_ITERS", "16"
))
# Where the central layer may push a station down to.
#
# Off (default): the bound is min(actor power, departure-safe power). When the
# actor has already discharged past its own departure-reachability floor, that
# minimum equals the actor power and the central layer gets no downward
# headroom at all, so a down command that the fleet could still have served is
# missed.
#
# On: the bound is the departure-safe power itself. The central layer may pull
# a station below where the actor put it, but never below what departure
# reachability allows.
CENTRAL_EV_ALLOCATOR_DOWN_TO_SAFE_FLOOR = _env_flag(
    "EVMA_CENTRAL_ALLOCATOR_DOWN_TO_SAFE_FLOOR", "0"
)
CENTRAL_EV_ALLOCATOR_DEPARTURE_SLACK_KWH = float(os.environ.get(
    "EVMA_CENTRAL_EV_ALLOCATOR_DEPARTURE_SLACK_KWH", "0.1"
))

if BESS_POWER_KW <= 0.0 or BESS_ENERGY_KWH <= 0.0:
    raise ValueError("BESS power and energy ratings must be positive")
if BESS_TARGET_POWER_CAP_KW <= 0.0:
    raise ValueError("target-deployment BESS power cap must be positive")
_BESS_POWER_AT_TARGET_KW = (
    BESS_POWER_KW
    * TARGET_DEPLOYMENT_STATIONS
    / max(NUM_STATIONS, 1)
)
if _BESS_POWER_AT_TARGET_KW > BESS_TARGET_POWER_CAP_KW + 1e-9:
    raise ValueError(
        "BESS power exceeds the target-deployment cap: "
        f"{_BESS_POWER_AT_TARGET_KW:.6g} kW at "
        f"{TARGET_DEPLOYMENT_STATIONS} stations > "
        f"{BESS_TARGET_POWER_CAP_KW:.6g} kW"
    )
if not 0.0 <= BESS_MIN_SOC_PCT < BESS_MAX_SOC_PCT <= 100.0:
    raise ValueError("BESS SoC bounds must satisfy 0 <= min < max <= 100")
if not BESS_MIN_SOC_PCT <= BESS_INITIAL_SOC_PCT <= BESS_MAX_SOC_PCT:
    raise ValueError("BESS initial SoC must lie inside the configured SoC bounds")
if not BESS_MIN_SOC_PCT <= BESS_TARGET_SOC_PCT <= BESS_MAX_SOC_PCT:
    raise ValueError("BESS target SoC must lie inside the configured SoC bounds")
if not 0.0 < BESS_CHARGE_EFFICIENCY <= 1.0:
    raise ValueError("BESS charge efficiency must be in (0, 1]")
if not 0.0 < BESS_DISCHARGE_EFFICIENCY <= 1.0:
    raise ValueError("BESS discharge efficiency must be in (0, 1]")
if CENTRAL_EV_ALLOCATOR_WATERFILL_ITERS <= 0:
    raise ValueError("central EV allocator water-fill iterations must be positive")
if CENTRAL_EV_ALLOCATOR_DEPARTURE_SLACK_KWH < 0.0:
    raise ValueError("central EV allocator departure slack must be non-negative")


# --- Day context and bid observation ---
# Calendar and weather features of the service day, for the observation only
# (off by default). EV arrivals depend on the day only through its weekday /
# holiday class (SERVICE_CALENDAR_COUNTRY).
USE_DAY_CONTEXT_ARRIVALS = _env_flag("EVMA_USE_DAY_CONTEXT_ARRIVALS")
DAY_CONTEXT_WEATHER_CSV = os.environ.get(
    "EVMA_DAY_CONTEXT_WEATHER_CSV",
    os.path.join(PROJECT_ROOT, "data", "weather", "open_meteo_boulder_forecast_2024_demand_days.csv"),
)
DAY_CONTEXT_USE_OBS = _env_flag("EVMA_DAY_CONTEXT_OBS")
DAY_CONTEXT_INCLUDE_WEATHER = _env_flag("EVMA_DAY_CONTEXT_INCLUDE_WEATHER", "1")
# Expose the physical instruction envelope to the lower policy.  ``demand_0``
# is divided by this day-specific value, so omitting it aliases (for example)
# 0.5 * 300 kW and 0.5 * 1000 kW even though the required fleet power and the
# absolute-kW tracking reward are different. This flag supplies the missing
# unit conversion; the separate lookahead setting below optionally adds the
# known baseline and award-width schedule.
LOWER_BID_CONTEXT_USE_OBS = _env_flag("EVMA_LOWER_BID_CONTEXT_OBS", "1")
# Number of known day-ahead bid blocks exposed from the current 30-minute
# block onward. The values are baseline/up/down schedules fixed before the
# operating day; realized future activation is never included. Zero keeps the
# compact current-instruction observation for ablations.
LOWER_BID_LOOKAHEAD_BLOCKS = int(os.environ.get(
    "EVMA_LOWER_BID_LOOKAHEAD_BLOCKS", "24"
))
if not 0 <= LOWER_BID_LOOKAHEAD_BLOCKS <= 48:
    raise ValueError("EVMA_LOWER_BID_LOOKAHEAD_BLOCKS must be in [0, 48]")


# --- Switching constraints ---
USE_SWITCHING_CONSTRAINTS = False
MAX_SWITCH_COUNT = 10 ** 9
LOCAL_SWITCH_PENALTY = 0.0


# --- Station power limits ---
USE_STATION_TOTAL_POWER_LIMIT = False
STATION_MAX_TOTAL_POWER_KW = float(MAX_EV_POWER_KW * MAX_EV_PER_STATION)
LOCAL_STATION_LIMIT_PENALTY = 0.0


# Wider submitted bids must let bid magnitude through the observation clamp.
OBS_DEMAND_CLAMP = float(os.environ.get(
    "EVMA_OBS_DEMAND_CLAMP",
    "2.5",
))
# --- Lower-controller training from an actual day-ahead bid -----------------
# This is the intended "forecast -> submitted bid -> lower control" data path.
# Each episode first computes a submitted day-ahead bid robust to minimum, median, and
# maximum EV-count realizations sampled from the day's forecast, then trains
# execution on realizations from that forecast distribution.
# Defaults consumed by the shared pipeline and advanced tools. Normal training
# starts from root-level pre_train.py. Legacy fine-tune reads the same values.
#   DAY=None -> pick by SPLIT + INDEX; set e.g. "2025-12-01" to fix one day.
LOWER_TRAIN_UPPER_BID_DAY = (os.environ.get("EVMA_LOWER_TRAIN_UPPER_BID_DAY", "").strip() or None)
LOWER_TRAIN_UPPER_BID_SPLIT = os.environ.get("EVMA_LOWER_TRAIN_UPPER_BID_SPLIT", "test").strip().lower()
LOWER_TRAIN_UPPER_BID_INDEX = int(os.environ.get("EVMA_LOWER_TRAIN_UPPER_BID_INDEX", 0))
# Warmup collection is counted separately by training/train.py.
# Controller precision is also reported at each interim interval.
LOWER_TRAIN_UPPER_BID_EPISODES = int(os.environ.get("EVMA_LOWER_TRAIN_UPPER_BID_EPISODES", 0))
LOWER_TRAIN_UPPER_BID_MODEL_NAME = os.environ.get(
    "EVMA_LOWER_TRAIN_UPPER_BID_MODEL_NAME",
    (
        "direct_bid_boa_2000ep_7station"
        if ACTIVATION_SIGNAL_SET == "nem"
        else f"direct_bid_{ACTIVATION_SIGNAL_SET}_2000ep_7station"
    ),
)
LOWER_TRAIN_UPPER_BID_TRAIN_SPLIT_COUNT = int(os.environ.get("EVMA_LOWER_TRAIN_UPPER_BID_TRAIN_SPLIT_COUNT", 25))
LOWER_TRAIN_UPPER_BID_USE_TRAIN_BANK = _env_flag("EVMA_LOWER_TRAIN_USE_BID_BANK", "1")
LOWER_TRAIN_UPPER_BID_BANK_BUILD_MISSING = _env_flag("EVMA_LOWER_TRAIN_BUILD_BID_BANK", "1")
# Train on a complete bank whose recorded upper-bid settings differ from the
# current code, without rebuilding it. Only for comparing a new controller
# setting against a run that was trained on that same bank; the bank's own
# settings stay what they were and are printed at start-up.
LOWER_TRAIN_ACCEPT_BANK_AS_IS = _env_flag("EVMA_LOWER_TRAIN_ACCEPT_BANK_AS_IS", "0")
LOWER_TRAIN_UPPER_BID_ACTIVATION_SCENARIOS = int(os.environ.get(
    "EVMA_LOWER_TRAIN_UPPER_BID_ACTIVATION_SCENARIOS", 128
))
LOWER_TRAIN_UPPER_BID_EV_SCENARIO_CANDIDATES = int(os.environ.get(
    "EVMA_LOWER_TRAIN_UPPER_BID_EV_SCENARIO_CANDIDATES", 128
))
if (
    LOWER_TRAIN_UPPER_BID_EV_SCENARIO_CANDIDATES < 1
    or LOWER_TRAIN_UPPER_BID_EV_SCENARIO_CANDIDATES == 2
):
    raise ValueError(
        "EV scenario candidates must be 1 for a single-realization diagnostic "
        "or at least 3 for min/median/max robustness"
    )
LOWER_TRAIN_UPPER_BID_EV_SCENARIO_LAYOUT = (
    "single_1of1"
    if LOWER_TRAIN_UPPER_BID_EV_SCENARIO_CANDIDATES == 1
    else f"minmedmax_3of{LOWER_TRAIN_UPPER_BID_EV_SCENARIO_CANDIDATES}"
)
LOWER_TRAIN_UPPER_BID_BANK_DIR = os.environ.get(
    "EVMA_LOWER_TRAIN_BID_BANK_DIR",
    os.path.join(
        PROJECT_ROOT,
        "execute_results",
        "bid_banks",
        (
            f"train_25_{LOWER_TRAIN_UPPER_BID_EV_SCENARIO_LAYOUT}ev_"
            f"{LOWER_TRAIN_UPPER_BID_ACTIVATION_SCENARIOS}cmd_all_commands"
            if ACTIVATION_SIGNAL_SET == "nem"
            else (
                f"train_25_{LOWER_TRAIN_UPPER_BID_EV_SCENARIO_LAYOUT}ev_"
                f"{LOWER_TRAIN_UPPER_BID_ACTIVATION_SCENARIOS}cmd_all_commands_"
                f"{ACTIVATION_SIGNAL_SET}"
            )
        ),
    ),
)
LOWER_TRAIN_UPPER_BID_TEST_BANK_COUNT = int(os.environ.get(
    "EVMA_LOWER_TRAIN_TEST_BID_BANK_COUNT",
    5,
))
LOWER_TRAIN_UPPER_BID_TEST_BANK_SEED_OFFSET = int(os.environ.get(
    "EVMA_LOWER_TRAIN_TEST_BID_BANK_SEED_OFFSET",
    1_000_003,
))
LOWER_TRAIN_UPPER_BID_TEST_BANK_DIR = os.environ.get(
    "EVMA_LOWER_TRAIN_TEST_BID_BANK_DIR",
    os.path.join(
        PROJECT_ROOT,
        "execute_results",
        "bid_banks",
        (
            f"validation_5_{LOWER_TRAIN_UPPER_BID_EV_SCENARIO_LAYOUT}ev_"
            f"{LOWER_TRAIN_UPPER_BID_ACTIVATION_SCENARIOS}cmd_all_commands"
            if ACTIVATION_SIGNAL_SET == "nem"
            else (
                f"validation_5_{LOWER_TRAIN_UPPER_BID_EV_SCENARIO_LAYOUT}ev_"
                f"{LOWER_TRAIN_UPPER_BID_ACTIVATION_SCENARIOS}cmd_all_commands_"
                f"{ACTIVATION_SIGNAL_SET}"
            )
        ),
    ),
)
# Training and validation dates are selected deterministically to cover
# weekday-class and month cells.
# How many processes solve the per-scenario feasibility columns for one day's
# bid. Every entry point that solves a bid must pass this through.
LOWER_TRAIN_BID_SCENARIO_WORKERS = int(os.environ.get(
    "EVMA_LOWER_TRAIN_SCENARIO_WORKERS", "1"
))
# --- Legacy fine-tune reproduction only ------------------------------------
# These settings are not used by the proposed system. They remain so the
# archived experiment under legacy/finetune can be reproduced.
FINETUNE_ENABLE = _env_flag("EVMA_FINETUNE_ENABLE", "0")
FINETUNE_EPISODES = int(os.environ.get("EVMA_FINETUNE_EPISODES", 300))
# Multiplies the normal initial/final exploration schedules during fine-tune.
FINETUNE_NOISE_SCALE = float(os.environ.get("EVMA_FINETUNE_NOISE_SCALE", 0.3))
# Config.WARMUP_STEPS (7000, ~24 episodes) is sized for learning an actor from
# scratch and wastes ~8% of the fine-tune budget (EVMA_FINETUNE_EPISODES,
# default 300) on pure buffer-fill. But the opposite extreme is worse: the
# warm-started critic is refit on whatever the buffer holds when updates
# begin, so starting at 3 episodes (864 steps, ~1.7 x BATCH_SIZE) refits it on
# a nearly degenerate sample and the restored policy degrades before it
# improves (measured: ~200 episodes to climb back). 8 episodes (2304 steps, 4.5 x
# BATCH_SIZE) keeps the buffer-fill cost under 3 % of the budget while giving
# the first critic updates a non-degenerate sample.
FINETUNE_WARMUP_STEPS = int(os.environ.get("EVMA_FINETUNE_WARMUP_STEPS", 8 * EPISODE_STEPS))
# The next-day activation is unknown when the specialist is trained.  Load a
# a broad historical command set independently of the design partition used to
# construct the already-fixed upper bid.
FINETUNE_ACTIVATION_SCENARIOS = int(os.environ.get(
    "EVMA_FINETUNE_ACTIVATION_SCENARIOS",
    "256",
))
# Keep command-shape evaluation disjoint from the fine-tune command pool.
FINETUNE_EVAL_ACTIVATION_SCENARIOS = int(os.environ.get(
    "EVMA_FINETUNE_EVAL_ACTIVATION_SCENARIOS", 512
))
# Optional existing pretrained results dir (or its results/TEST* checkpoint
# dir) used as the warm start, so fine-tune can run without redoing pretrain.
FINETUNE_WARMSTART_DIR = os.environ.get("EVMA_FINETUNE_WARMSTART_DIR", "").strip()
# Reinitialize the critic output heads after the warm start, before the
# fine-tune loop. Off by default.
#
# The restored critic carries a value function fitted to the other days' bids.
# On a new day's bid those estimates are miscalibrated, and the actor is
# updated through them. Nikishin et al. (ICML 2022) report that resetting the
# critic is the component that matters, with the replay buffer kept so the
# critic refits quickly; here the warm start also restores a replay snapshot,
# so there is data to refit on from the first update.
#
#   "off"         : no reset (default)
#   "last_linear" : only the final Linear of each value head
#   "head"        : the whole output head (q_head / per_agent_head /
#                   mixer_w / mixer_b)
#
# Online and target critics are both reset, and the Adam moments for the reset
# parameters are dropped; leaving either in place would pull the fresh head
# back toward the value it was supposed to forget.
FINETUNE_RESET_CRITIC_SCOPE = os.environ.get(
    "EVMA_FINETUNE_RESET_CRITIC_SCOPE", "off"
).strip().lower()
if FINETUNE_RESET_CRITIC_SCOPE not in ("off", "last_linear", "head"):
    raise ValueError(
        "EVMA_FINETUNE_RESET_CRITIC_SCOPE must be off, last_linear or head"
    )

# What a fixed bid makes new about a day is its award band and the command
# profile that follows from it.  In the global critic that reaches the value
# function only through the mixer, which reads the demand/time context and the
# realized station powers; the per-station embedding and the per-agent utility
# head describe station physics, which the day does not change.  Restricting
# adaptation to the mixer spends the capacity on the part that sees the shift.
#
#   "all"                 : every critic parameter adapts (the default)
#   "global_mixer"        : in the global critics only mixer_w / mixer_b /
#                           common_mode_gain adapt
#   "global_mixer_strict" : additionally hold the local critics, leaving the
#                           mixer as the only part of the value function that
#                           can still move
#
# Freezing is enforced by dropping the parameters from the optimizer, because
# requires_grad does not survive the actor update and a zero gradient alone
# would still be carried by the Adam moments.
FINETUNE_CRITIC_ADAPT_SCOPE = os.environ.get(
    "EVMA_FINETUNE_CRITIC_ADAPT_SCOPE", "all"
).strip().lower()
if FINETUNE_CRITIC_ADAPT_SCOPE not in ("all", "global_mixer", "global_mixer_strict"):
    raise ValueError(
        "EVMA_FINETUNE_CRITIC_ADAPT_SCOPE must be all, global_mixer "
        "or global_mixer_strict"
    )
# Fine-tuning a converged policy can lose. Candidates are scored on held-out
# realizations and the best is adopted, the warm start among them, so
# "no adaptation" is a reachable answer.
FINETUNE_SELECT_ENABLE = _env_flag("EVMA_FINETUNE_SELECT", "1")
# EV realizations used to choose. Disjoint from the final evaluation's seeds so
# the number that gets reported is not the number that picked the model.
# Evaluating every held-out command for every checkpoint would make choosing
# the checkpoint dominate the fine-tune cost. Two EV seeds retain a paired
# comparison while the command subset below bounds that cost.
FINETUNE_SELECTION_SEEDS = int(os.environ.get("EVMA_FINETUNE_SELECTION_SEEDS", 2))
FINETUNE_SELECTION_BASE_SEED = int(os.environ.get(
    "EVMA_FINETUNE_SELECTION_BASE_SEED", 610_000
))
# Selection uses a fixed subset of the final holdout command bank. The final
# report evaluates every holdout command; evaluating all 512 for every
# checkpoint would make model selection several times more expensive than the
# fine-tune itself.
FINETUNE_SELECTION_ACTIVATION_SCENARIOS = int(os.environ.get(
    "EVMA_FINETUNE_SELECTION_ACTIVATION_SCENARIOS", 96
))
# Every saved checkpoint would be evaluated otherwise. Candidates are spread
# evenly over the run so both an early peak and a late one stay reachable.
FINETUNE_SELECTION_MAX_CANDIDATES = int(os.environ.get(
    "EVMA_FINETUNE_SELECTION_MAX_CANDIDATES", 3
))
# After training, evaluate the trained controller under fresh stochastic EV
# realizations drawn from the same day-ahead arrival-probability forecast used
# for training. Arrival-probability forecast error is a separate robustness test.
# EVAL_SEEDS = EV realizations per scenario.
LOWER_TRAIN_UPPER_BID_EVAL_CONTROLLER_PRECISION = _env_flag(
    "EVMA_LOWER_TRAIN_UPPER_BID_EVAL_CONTROLLER_PRECISION", "1"
)
LOWER_TRAIN_UPPER_BID_EVAL_SEEDS = int(os.environ.get("EVMA_LOWER_TRAIN_UPPER_BID_EVAL_SEEDS", 5))
LOWER_TRAIN_UPPER_BID_UP_MAX_KW = float(os.environ.get("EVMA_LOWER_TRAIN_UPPER_BID_UP_MAX_KW", PHYSICAL_MAX_POWER_KW))
LOWER_TRAIN_UPPER_BID_DOWN_MAX_KW = float(os.environ.get("EVMA_LOWER_TRAIN_UPPER_BID_DOWN_MAX_KW", PHYSICAL_MAX_POWER_KW))
LOWER_TRAIN_UPPER_BID_BASELINE_MIN_KW = float(os.environ.get("EVMA_LOWER_TRAIN_UPPER_BID_BASELINE_MIN_KW", -PHYSICAL_MAX_POWER_KW))
LOWER_TRAIN_UPPER_BID_BASELINE_MAX_KW = float(os.environ.get("EVMA_LOWER_TRAIN_UPPER_BID_BASELINE_MAX_KW", PHYSICAL_MAX_POWER_KW))
# Assessment-I is an EV-fleet 30-minute sustained-capability bound computed
# across three selected EV realizations. Rank a reproducible candidate pool by
# accepted EV-session count, then retain minimum, median, and maximum.
# Jointly choose one day-ahead baseline/up/down profile against every selected
# EV realization and all 128 design commands.
# Departure SoC and every assessed 5-minute tracking point are hard constraints.
LOWER_TRAIN_UPPER_BID_REWARD_BAND_FRAC = float(os.environ.get("EVMA_LOWER_TRAIN_UPPER_BID_REWARD_BAND_FRAC", 0.10))
# Participation floor. Two separate requirements, and the binding one is not
# the market's.
#
# Market: EPRX sets a 1 MW minimum per non-zero direction. The 7-station rig is
# a scaled-down slice of the 500-station deployment this study targets, so that
# scales to 1 MW x 7/500 = 14 kW.
#
# Controller: the Assessment-II band is 10% of the award, so a small award
# demands a precision the lower controller cannot hold. A block whose band is
# narrower than the controller's tracking error cannot be tracked at any amount
# of training, so it only adds guaranteed Assessment-II failures.
#
# LOWER_CONTROLLER_MIN_TRACKING_BAND_KW sets the narrowest band worth bidding
# into, and the floor follows from it (floor = band / band fraction). Dropping
# the blocks below it costs little physical regulation capacity because their
# directional awards are small.
#
# The band is set per station count from the trained lower controller's raw
# tracking error (interim-test MAE of the MARL output):
#   7 stations : 25 kW (minimum bid 250 kW), set by the user.
#   20 stations: 80 kW (minimum bid 800 kW). Interim-test MAE over the last
#                400 episodes of the 20-station AB runs (EV model before the
#                station-session tables): AEMO 68.3 kW, ERCOT 87.0 kW; their
#                mean 77.7 kW times 10 rounded to 800 kW.
# Another station count has no value until its controller error is measured;
# set EVMA_LOWER_CONTROLLER_MIN_TRACKING_BAND_KW for it.
#
# Note the floor still has to stay under the fleet's sustained capability. If
# it exceeds it, the "0 or at least the floor" constraint has no feasible
# non-zero point and the bid comes out empty rather than conservative --
# _sustainable_plan_bounds warns when that happens.
_MIN_TRACKING_BAND_KW_BY_STATIONS = {7: 25.0, 20: 80.0}
_min_band_env = os.environ.get("EVMA_LOWER_CONTROLLER_MIN_TRACKING_BAND_KW", "").strip()
LOWER_CONTROLLER_MIN_TRACKING_BAND_KW = (
    float(_min_band_env) if _min_band_env
    else _MIN_TRACKING_BAND_KW_BY_STATIONS.get(NUM_STATIONS)
)
MARKET_MINIMUM_BID_QUANTITY_KW = float(os.environ.get(
    "EVMA_MARKET_MINIMUM_BID_QUANTITY_KW",
    1000.0,
))
# The initial bid is also the per-block upper limit of the Benders search. With
# this on, it comes from an aggregate LP that checks the fleet's cumulative
# energy over the day in every EV realization x design command, not only the
# 30-minute Assessment-I capability of each block. 0 keeps the Assessment-I
# seed.
LOWER_TRAIN_UPPER_BID_ENERGY_INITIAL_BID = _env_flag(
    "EVMA_LOWER_TRAIN_UPPER_BID_ENERGY_INITIAL", "1"
)
LOWER_TRAIN_UPPER_BID_ENERGY_INITIAL_TIME_LIMIT_S = float(os.environ.get(
    "EVMA_LOWER_TRAIN_UPPER_BID_ENERGY_INITIAL_TIME_LIMIT_S",
    "1800",
))
# Objective weight on the baseline step between adjacent participating blocks,
# in kW-block per kW of step, in the seed LP and the Benders master. Capacity
# alone leaves the baseline free wherever Assessment I and the recourse do not
# pin it, and the solver then returns an arbitrary vertex: the baseline jumps to
# the Assessment-I ceiling for one block and back. At 1e-3 removing a 1000 kW
# step can cost at most 1 kW-block of capacity. 0 leaves the baseline to the
# solver.
LOWER_TRAIN_UPPER_BID_BASELINE_STEP_WEIGHT = float(os.environ.get(
    "EVMA_LOWER_TRAIN_UPPER_BID_BASELINE_STEP_WEIGHT",
    "1e-3",
))
# Farkas cuts added per inner Benders iteration. Every iteration checks all
# 3 x 128 EV/command combinations; this cap limits cuts entering the master.
LOWER_TRAIN_UPPER_BID_BENDERS_CUTS_PER_ROUND = int(os.environ.get(
    "EVMA_LOWER_TRAIN_UPPER_BID_BENDERS_CUTS_PER_ROUND",
    10,
))
# With a per-iteration cap the master needs more iterations to accumulate the
# same evidence, so the work is bounded by total cuts and the iteration count
# is left generous enough that the cut bound is the one that binds. Exhausting
# the budget is reported as its own stop reason; it is never read as proof that
# a participation pattern is infeasible.
LOWER_TRAIN_UPPER_BID_BENDERS_MAX_ROUNDS = int(os.environ.get(
    "EVMA_LOWER_TRAIN_UPPER_BID_BENDERS_MAX_ROUNDS",
    200,
))
# The budget bounds work, not conclusions: exhausting it is reported as an
# incomplete solve and is never interpreted as infeasibility.
LOWER_TRAIN_UPPER_BID_BENDERS_MAX_TOTAL_CUTS = int(os.environ.get(
    "EVMA_LOWER_TRAIN_UPPER_BID_BENDERS_MAX_TOTAL_CUTS",
    4000,
))
LOWER_TRAIN_RESEARCH_MINIMUM_BID_QUANTITY_KW = os.environ.get(
    "EVMA_LOWER_TRAIN_RESEARCH_MINIMUM_BID_QUANTITY_KW",
    # The larger of the scaled market minimum and the award whose 10% band the
    # controller can actually hold. Empty when the band is not set for this
    # station count; training.blockwise_bid.minimum_bid_quantity_kw then refuses
    # to build a bid.
    str(
        max(
            MARKET_MINIMUM_BID_QUANTITY_KW
            * NUM_STATIONS
            / max(TARGET_DEPLOYMENT_STATIONS, 1),
            LOWER_CONTROLLER_MIN_TRACKING_BAND_KW
            / max(LOWER_TRAIN_UPPER_BID_REWARD_BAND_FRAC, 1e-9),
        )
    ) if LOWER_CONTROLLER_MIN_TRACKING_BAND_KW is not None else "",
)
# The literal EPRX minimum is 1,000 kW per non-zero direction. Keep zero as
# the research-scale default; the solver still enforces the 1 kW bid unit.
# Set this to 1000 for a market-eligibility run.
LOWER_TRAIN_UPPER_BID_PHYSICAL_LP_MIN_DIRECTION_BID_KW = float(os.environ.get(
    "EVMA_LOWER_TRAIN_UPPER_BID_PHYSICAL_LP_MIN_DIRECTION_BID_KW",
    "0.0",
))
LOWER_TRAIN_UPPER_BID_PHYSICAL_LP_TIME_LIMIT_S = float(os.environ.get(
    "EVMA_LOWER_TRAIN_UPPER_BID_PHYSICAL_LP_TIME_LIMIT_S",
    "180",
))
# A direct sparse phase-I LP is substantially faster for today's roughly
# 90-EV scenarios, while column generation overtakes it as the fleet grows.
# The measured crossover on the fixed 256-command bank is about 350 EVs; keep a
# margin so the 50-station path uses the better-scaling decomposition.
LOWER_TRAIN_UPPER_BID_DIRECT_ORACLE_MAX_EVS = int(os.environ.get(
    "EVMA_LOWER_TRAIN_UPPER_BID_DIRECT_ORACLE_MAX_EVS",
    "300",
))
# Worker-local path columns are warm starts, not part of the certificate.
# Bounding them prevents later commands/Benders rounds from inheriting an
# ever-growing restricted master. Zero restores the unbounded cache.
LOWER_TRAIN_UPPER_BID_COLGEN_CACHE_COLUMNS_PER_EV = int(os.environ.get(
    "EVMA_LOWER_TRAIN_UPPER_BID_COLGEN_CACHE_COLUMNS_PER_EV",
    "12",
))
# With unit charge/discharge efficiency the scenario recourse is a
# prefix-bounded matrix feasibility problem, so one feasible circulation
# answers it.  The screen rounds the arc bounds in both directions and asks a
# compiled max flow: a feasible answer comes from a restriction of the real
# problem and an infeasible one from a relaxation, so neither can be wrong, and
# the exact LP still runs whenever the two roundings straddle.  Set to 0 to
# take the LP on every scenario.
LOWER_TRAIN_UPPER_BID_CIRCULATION_SCREEN = _env_flag(
    "EVMA_LOWER_TRAIN_UPPER_BID_CIRCULATION_SCREEN", "1"
)
# Let the circulation answer the infeasible side as well, and hand its
# Hoffman inequality to the Benders master instead of an LP dual.  The
# inequality is exact, but it is a different inequality, so the master walks a
# different path to an equal-capacity optimum. Measured at seven and twenty
# stations, on pools of 128, 256 and 512 commands and on a second EV draw: the
# capacity came out identical every time, and the solve took a third to an
# eighth of the time, because the commands that miss stop going to an LP for
# their dual. Set to 0 to take the LP dual.
LOWER_TRAIN_UPPER_BID_CIRCULATION_CUT = _env_flag(
    "EVMA_LOWER_TRAIN_UPPER_BID_CIRCULATION_CUT", "1"
)
LOWER_TRAIN_UPPER_BID_VERBOSE_BID_BUILD = _env_flag("EVMA_LOWER_TRAIN_UPPER_BID_VERBOSE_BID_BUILD", "1")
LOWER_TRAIN_UPPER_BID_SEED = int(os.environ.get("EVMA_LOWER_TRAIN_UPPER_BID_SEED", 73000))
# Every submitted block is assumed to clear. The study optimizes execution of
# the fixed bid and does not model participant-level award uncertainty.
# The production bank designs against this many forecast commands and stores
# the same number of disjoint feedback commands for lower-controller training.
LOWER_TRAIN_UPPER_BID_ACTIVATION_SOURCE_DIR = os.environ.get(
    "EVMA_LOWER_TRAIN_UPPER_BID_ACTIVATION_SOURCE_DIR",
    ACTIVATION_SCENARIO_DIR,
)
