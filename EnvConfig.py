"""
EnvConfig.py - environment and EV-scenario configuration.

Keep simulator, demand-scaling, arrival and station-scenario settings here.
`Config.py` re-exports these names; environment and market code imports from
`EnvConfig` directly.
"""

from __future__ import annotations

import os


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


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
# It is paid on the raw actor residual: the EV-fleet power before the central
# residual allocator and the battery.
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
# Departure reward: LOCAL_R_SOC_HIT + SOC_HIT_BONUS at or above target,
# otherwise -(LOCAL_DEPARTURE_MISS_PENALTY + LINEAR * shortfall in SoC points).
SOC_HIT_BONUS = 0.5
LOCAL_DEPARTURE_MISS_PENALTY = 0.5
LOCAL_DEPARTURE_DEFICIT_PENALTY_LINEAR = 0.05
# SoC shaping: each step, the fall in each EV's shortfall below target, weighted
# from 1 up to 1 + URGENCY_GAIN as the steps left before departure fall below
# URGENCY_STEPS, averaged over the station's present EVs and clipped to +-CLIP.
LOCAL_DEFICIT_SHAPING_COEF = 0.50
LOCAL_DEFICIT_SHAPING_CLIP = float(os.environ.get("EVMA_LOCAL_SHAPING_CLIP", "0.08"))
LOCAL_DEFICIT_SHAPING_URGENCY_GAIN = float(os.environ.get("EVMA_LOCAL_URGENCY_GAIN", "1.0"))
LOCAL_DEFICIT_SHAPING_URGENCY_STEPS = 48

# Tracking tolerance (kW) of an episode reset without a per-step tolerance.
TOL_NARROW_METRICS = 150.0


# --- EV physics and episode geometry ---
EPISODE_STEPS = 288

# Per-EV battery capacity and charger rating. Each arriving EV draws its own
# capacity/power, and the upper bidder picks these up
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
# rather than today's 6-7 kW workplace AC.

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
EV_CAPACITY_OBS_SCALE_KWH = max(EV_BATTERY_CAPACITY_OPTIONS_KWH)
EV_CHARGER_POWER_OBS_SCALE_KW = max(EV_CHARGER_MAX_POWER_OPTIONS_KW)
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


# --- Command sets and command scaling ---
# Empirical load-following command sets sharing one scenario interface. What
# each contains and how it is normalised: docs/指令データ.md.
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
    "aemo_bess_dispatch": os.path.join(
        PROJECT_ROOT,
        "data",
        "aemo",
        "nem",
        "command_libraries",
        "bess_dispatch_calendar_day",
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
ACTIVATION_SCENARIO_DIR = _ACTIVATION_SIGNAL_DIRS[ACTIVATION_SIGNAL_SET]
# Fleet power ceiling used as the bid cap and demand-target clamp. Each EV
# draws its own rating, so the max-rating product does not describe the
# fleet: with the current mix the expected aggregate is ~1756 kW, while
# "every EV draws the 50 kW tier" (3500 kW) has effectively zero probability
# over 70 slots. Scale by the mean rating with a headroom factor so the cap is
# a loose-but-reachable bound; the bidder then shrinks it to the deliverable
# width through its own feasibility check.
_MEAN_PER_EV_POWER_KW = sum(
    w * p for w, p in zip(EV_CHARGER_MAX_POWER_OPTIONS_KW, EV_CHARGER_MAX_POWER_PROBS)
)
_PER_EV_POWER_CAP_KW = min(1.25 * _MEAN_PER_EV_POWER_KW, max(EV_CHARGER_MAX_POWER_OPTIONS_KW))
PHYSICAL_MAX_POWER_KW = float(NUM_STATIONS * MAX_EV_PER_STATION * _PER_EV_POWER_CAP_KW)
DEMAND_TARGET_MIN_KW = float(os.environ.get("EVMA_DEMAND_MIN_KW", -PHYSICAL_MAX_POWER_KW))
DEMAND_TARGET_MAX_KW = float(os.environ.get("EVMA_DEMAND_MAX_KW", PHYSICAL_MAX_POWER_KW))

# --- Grid-side residual BESS ------------------------------------------------
# This BESS is an independent PCC actuator. It never rewrites an EV action and
# never changes the EV-side learning reward. The physical PCC output and a
# separate set of system metrics use the post-BESS result. During free steps a
# deterministic controller recentres stored energy for the next dispatch block.
# EVEnv runs it unless a caller turns it off; training and the interim tests
# turn it off, since its only consumer there is a reported column.
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
# retained only as a central-observation rule baseline (the rule_based_central
# evaluation pipeline): every station obeys a centrally assigned power target.
# The proposed MARL system instead assumes that each station owns its connected
# EV load and makes its own action.
CENTRAL_EV_ALLOCATOR_WATERFILL_ITERS = int(os.environ.get(
    "EVMA_CENTRAL_EV_ALLOCATOR_WATERFILL_ITERS", "16"
))
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


# --- Bid observation ---
# Number of known day-ahead bid blocks exposed from the current 30-minute
# block onward. The values are baseline/up/down schedules fixed before the
# operating day; realized future activation is never included.
LOWER_BID_LOOKAHEAD_BLOCKS = int(os.environ.get(
    "EVMA_LOWER_BID_LOOKAHEAD_BLOCKS", "24"
))
if not 0 <= LOWER_BID_LOOKAHEAD_BLOCKS <= 48:
    raise ValueError("EVMA_LOWER_BID_LOOKAHEAD_BLOCKS must be in [0, 48]")


# Wider submitted bids must let bid magnitude through the observation clamp.
OBS_DEMAND_CLAMP = float(os.environ.get(
    "EVMA_OBS_DEMAND_CLAMP",
    "2.5",
))


# --- Upper bid and bid banks -------------------------------------------------
# Each bank day's submitted bid is robust to a few EV realizations picked from
# the candidates sampled from the day's forecast (see
# LOWER_TRAIN_UPPER_BID_EV_SCENARIO_SELECTION). Lower-controller training runs
# on banks built by tools/build_training_bid_bank.py.
LOWER_TRAIN_UPPER_BID_ACTIVATION_SCENARIOS = int(os.environ.get(
    "EVMA_LOWER_TRAIN_UPPER_BID_ACTIVATION_SCENARIOS", 128
))
LOWER_TRAIN_UPPER_BID_EV_SCENARIO_CANDIDATES = int(os.environ.get(
    "EVMA_LOWER_TRAIN_UPPER_BID_EV_SCENARIO_CANDIDATES", 128
))
# How the realizations are picked, and how many.
#   session_count (3): the minimum, lower median and maximum session count.
#   low_connection_2_max_count (3): two candidates that cover the low side of
#     the connected charging power block by block (30 minutes), then the
#     maximum session count.
#   low_connection_1_median_max_count (3): one such low-side candidate, then
#     the lower median and the maximum session count of the rest.
#   low_connection_3_max_count (4): three low-side candidates, then the
#     maximum session count.
# Independent realizations that break a session-count bid have, in some
# block, fewer EVs connected than every pick.
LOWER_TRAIN_UPPER_BID_EV_SCENARIO_SELECTION = os.environ.get(
    "EVMA_LOWER_TRAIN_UPPER_BID_EV_SCENARIO_SELECTION", "low_connection_3_max_count"
)
_EV_SCENARIO_SELECTIONS = {
    "session_count": ("minmedmax", 3),
    "low_connection_2_max_count": ("low2max", 3),
    "low_connection_1_median_max_count": ("low1medmax", 3),
    "low_connection_3_max_count": ("low3max", 4),
}
if LOWER_TRAIN_UPPER_BID_EV_SCENARIO_SELECTION not in _EV_SCENARIO_SELECTIONS:
    raise ValueError(
        "EVMA_LOWER_TRAIN_UPPER_BID_EV_SCENARIO_SELECTION must be one of "
        f"{sorted(_EV_SCENARIO_SELECTIONS)}, got "
        f"{LOWER_TRAIN_UPPER_BID_EV_SCENARIO_SELECTION!r}"
    )
LOWER_TRAIN_UPPER_BID_EV_SCENARIOS = _EV_SCENARIO_SELECTIONS[
    LOWER_TRAIN_UPPER_BID_EV_SCENARIO_SELECTION
][1]
if LOWER_TRAIN_UPPER_BID_EV_SCENARIO_CANDIDATES < LOWER_TRAIN_UPPER_BID_EV_SCENARIOS:
    raise ValueError(
        f"EV scenario candidates must be at least {LOWER_TRAIN_UPPER_BID_EV_SCENARIOS} "
        f"for {LOWER_TRAIN_UPPER_BID_EV_SCENARIO_SELECTION}"
    )
LOWER_TRAIN_UPPER_BID_EV_SCENARIO_LAYOUT = (
    f"{_EV_SCENARIO_SELECTIONS[LOWER_TRAIN_UPPER_BID_EV_SCENARIO_SELECTION][0]}"
    f"_{LOWER_TRAIN_UPPER_BID_EV_SCENARIOS}of{LOWER_TRAIN_UPPER_BID_EV_SCENARIO_CANDIDATES}"
)
LOWER_TRAIN_UPPER_BID_BANK_DIR = os.environ.get(
    "EVMA_LOWER_TRAIN_BID_BANK_DIR",
    os.path.join(
        PROJECT_ROOT,
        "execute_results",
        "bid_banks",
        f"train_25_{LOWER_TRAIN_UPPER_BID_EV_SCENARIO_LAYOUT}ev_"
        f"{LOWER_TRAIN_UPPER_BID_ACTIVATION_SCENARIOS}cmd_all_commands_"
        f"{ACTIVATION_SIGNAL_SET}",
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
        f"validation_5_{LOWER_TRAIN_UPPER_BID_EV_SCENARIO_LAYOUT}ev_"
        f"{LOWER_TRAIN_UPPER_BID_ACTIVATION_SCENARIOS}cmd_all_commands_"
        f"{ACTIVATION_SIGNAL_SET}",
    ),
)
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
# The initial bid is also the per-block upper limit of the Benders search. It
# comes from an aggregate LP that checks the fleet's cumulative energy over the
# day in every EV realization x design command, not only the 30-minute
# Assessment-I capability of each block.
LOWER_TRAIN_UPPER_BID_ENERGY_INITIAL_TIME_LIMIT_S = float(os.environ.get(
    "EVMA_LOWER_TRAIN_UPPER_BID_ENERGY_INITIAL_TIME_LIMIT_S",
    "1800",
))
# Objective weight on the baseline step between adjacent participating blocks,
# in kW-block per kW of step, in the seed LP and the Benders master. Capacity
# alone leaves the baseline free wherever Assessment I and the recourse do not
# pin it, so the solver returns an arbitrary vertex; and in a block that offers
# only up, raising the baseline widens the up band one for one, so capacity
# alone pushes that block's baseline to the Assessment-I ceiling and back. At
# 0.5 the search gives up at most 0.5 kW-block of capacity per kW of step it
# removes. 20 stations, AEMO, 2024-01-03: summed step 14,063 kW at 1e-3 and
# 3,762 kW at 0.5; capacity 121,722 and 118,464 kW-block (the 0.5 run also has
# the EVSpec SoC-unit fix). 0 leaves the baseline to the solver.
LOWER_TRAIN_UPPER_BID_BASELINE_STEP_WEIGHT = float(os.environ.get(
    "EVMA_LOWER_TRAIN_UPPER_BID_BASELINE_STEP_WEIGHT",
    "0.5",
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
LOWER_TRAIN_UPPER_BID_SEED = int(os.environ.get("EVMA_LOWER_TRAIN_UPPER_BID_SEED", 73000))
# Every submitted block is assumed to clear. The study optimizes execution of
# the fixed bid and does not model participant-level award uncertainty.
# The production bank designs against this many forecast commands and stores
# the same number of disjoint feedback commands for lower-controller training.
LOWER_TRAIN_UPPER_BID_ACTIVATION_SOURCE_DIR = os.environ.get(
    "EVMA_LOWER_TRAIN_UPPER_BID_ACTIVATION_SOURCE_DIR",
    ACTIVATION_SCENARIO_DIR,
)
