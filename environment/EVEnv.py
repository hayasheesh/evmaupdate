"""
EV environment for multi-station EV charging control.

This module manages:
- episode progression and per-step demand requests
- EV arrivals, departures, and per-station slot state
- SoC-constrained action application
- local reward shaping and global demand-tracking reward
- optional snapshot generation for evaluation and plotting

Data inputs:
- Demand series comes from `reset(net_demand_series=...)` or, when omitted, from
  `environment.readcsv` train/test demand episodes.
- EVs come from each station's own measured charging sessions
  (`environment.station_sessions`, tables in `EnvConfig.STATION_SESSION_DIR`).

Core state tensors:
- Shape is `(num_stations, MAX_EV_PER_STATION)` for `soc`, `target`,
  `ev_capacity_kwh`, `ev_max_power_kw`, `depart`, `ev_mask`, `switch_count`,
  and related per-slot tensors.
- `ev_mask` marks active EV slots. Empty slots keep zeroed state and are ignored
  by action application, rewards, and observations.

Arrival model:
- The service day's weekday/holiday class picks each station's arrival rate,
  hourly arrival shape and session pool.
- At reset time, `_pregenerate_arrival_events()` samples all episode arrivals
  as Bernoulli trials on every charger, and `_session_attributes()` gives each
  arrival the dwell, delivered energy and (residential) plug-in SoC of one
  session that started in the same clock hour.
- The EVs present at 00:00 come from running the same process over the
  previous days without control (`_preroll_initial_evs()`).
- An EV staying past the last step is held, at the last step, to the SoC the
  next day assumes for it at 00:00 (charged at its rated power from arrival
  until its target).
"""

from __future__ import annotations

import random
import random as _pyrandom
from dataclasses import replace

import numpy as np
import torch
from environment.ev_info_loader import load_arrival_soc_cdf
from environment.calendars import WEEKDAY
from environment.station_sessions import (
arrival_probabilities ,
draw_session ,
load_station_pool ,
previous_day_classes ,
service_day_class ,
)
from Config import DEVICE
from EnvConfig import (
EV_CAPACITY ,EPISODE_STEPS ,TOL_NARROW_METRICS ,
SOC_WIDE ,
MAX_EV_PER_STATION ,
EV_SOC_ARRIVAL_DISTRIBUTION_PATH ,
STATION_SESSION_DIR ,
PER_STATION_SESSION_IDS ,
FUTURE_EV_ARRIVAL_GROWTH_MULTIPLIER ,
SERVICE_CALENDAR_COUNTRY ,
SESSION_MATCH_MIN_CANDIDATES ,
EV_PREROLL_DAYS ,
EV_TARGET_REACHABLE_POWER_FRACTION ,
NUM_STATIONS ,NUM_EVS ,GLOBAL_BALANCE_REWARD ,GLOBAL_BALANCE_REWARD_SLOPE ,
GLOBAL_BALANCE_REWARD_MODE ,GLOBAL_BALANCE_REWARD_ERROR_SCALE_KW ,GLOBAL_BALANCE_REWARD_LINEAR_TAIL_KW ,LOCAL_USE_FLEET_RESIDUAL ,
GLOBAL_REWARD_ON_SYSTEM_OUTPUT ,
MAX_EV_POWER_KW ,POWER_TO_ENERGY ,TIME_STEP_MINUTES ,
EV_BATTERY_CAPACITY_OPTIONS_KWH ,EV_BATTERY_CAPACITY_PROBS ,
EV_CHARGER_MAX_POWER_OPTIONS_KW ,EV_CHARGER_MAX_POWER_PROBS ,
USE_HETEROGENEOUS_EV_PHYSICS ,
USE_STATION_TOTAL_POWER_LIMIT ,STATION_MAX_TOTAL_POWER_KW ,
LOCAL_DEFICIT_SHAPING_COEF ,
LOCAL_DEFICIT_SHAPING_CLIP ,
LOCAL_DEFICIT_SHAPING_URGENCY_GAIN ,
LOCAL_DEFICIT_SHAPING_URGENCY_STEPS ,
LOCAL_SHAPING_REDUCTION ,
LOCAL_SURPLUS_SHAPING_COEF ,
LOCAL_URGENCY_BASIS ,
LOCAL_DEPARTURE_REWARD_MODE ,
LOCAL_DEPARTURE_SMOOTH_LINEAR ,
LOCAL_DEPARTURE_SMOOTH_QUADRATIC ,
LOCAL_DEPARTURE_MISS_PENALTY ,
LOCAL_DEPARTURE_DEFICIT_PENALTY_LINEAR ,
LOCAL_R_SOC_HIT ,
SOC_HIT_BONUS ,
LOCAL_STATION_LIMIT_PENALTY ,
LOCAL_SWITCH_PENALTY ,
USE_SWITCHING_CONSTRAINTS ,
DAY_CONTEXT_USE_OBS ,
DAY_CONTEXT_INCLUDE_WEATHER ,
USE_RESIDUAL_BESS ,
TRAIN_FORCE_CHARGING ,
TRAIN_FORCE_SLACK_KWH ,
LOCAL_FORCED_PENALTY_PER_POINT ,
LOCAL_CRITIC_PER_EV ,
LOCAL_REWARD_MODE ,
LOCAL_POTENTIAL_COEF ,
LOCAL_POTENTIAL_URGENCY_GAIN ,
LOCAL_POTENTIAL_WINDOW_STEPS ,
LOCAL_POTENTIAL_MISS_PENALTY ,
LOCAL_POTENTIAL_GAMMA ,
BESS_CONTEXT_USE_OBS ,
BESS_POWER_KW ,
BESS_ENERGY_KWH ,
BESS_INITIAL_SOC_PCT ,
BESS_TARGET_SOC_PCT ,
BESS_MIN_SOC_PCT ,
BESS_MAX_SOC_PCT ,
BESS_CHARGE_EFFICIENCY ,
BESS_DISCHARGE_EFFICIENCY ,
USE_CENTRAL_EV_RESIDUAL_ALLOCATOR ,
CENTRAL_EV_ALLOCATOR_WATERFILL_ITERS ,
CENTRAL_EV_ALLOCATOR_DEPARTURE_SLACK_KWH ,
)
from environment.observation_config import (
EV_FEAT_DIM ,
LOCAL_DEMAND_STEPS ,
LOCAL_USE_TRACKING_ENABLED ,
LOCAL_USE_STEP ,
BID_CONTEXT_FEATURES ,
BESS_CONTEXT_FEATURES ,
LOWER_BID_CONTEXT_USE_OBS ,
CALENDAR_CONTEXT_FEATURES ,
WEATHER_CONTEXT_FEATURES ,
)
from environment.central_residual_allocator import allocate_central_ev_residual
from environment.force_floor import force_floor_fraction

device =DEVICE


def soc_progress_shaping (
prev_socs ,new_socs ,target ,remaining_steps ,*,
full_power_soc_per_step =None ,
coef :float =LOCAL_DEFICIT_SHAPING_COEF ,
clip :float =LOCAL_DEFICIT_SHAPING_CLIP ,
urgency_gain :float =LOCAL_DEFICIT_SHAPING_URGENCY_GAIN ,
urgency_steps :float =LOCAL_DEFICIT_SHAPING_URGENCY_STEPS ,
urgency_basis :str =LOCAL_URGENCY_BASIS ,
reduction :str =LOCAL_SHAPING_REDUCTION ,
surplus_coef :float =LOCAL_SURPLUS_SHAPING_COEF ,
return_per_ev :bool =False ,
):
    """One station's SoC shaping reward for one step, from its present EVs.

    Deficit term: the fall in each EV's shortfall below target (fraction of
    capacity), weighted from 1 up to 1 + urgency_gain as the EV's slack falls
    from urgency_steps to zero. The slack is the steps left before departure
    ("time"), or those steps minus the steps full-power charging needs to reach
    the target ("laxity", which requires full_power_soc_per_step, the SoC points
    one step at the EV's rating adds). Surplus term: the fall in each EV's SoC
    above target, unweighted. Both are combined over the station's EVs by
    `reduction` and the total is clipped to +-clip. With return_per_ev it also
    returns each EV's share, which sums to the station value: its term through
    the same reduction, scaled down with the others when the clip binds.
    """
    deficit_prev =torch .clamp (target -prev_socs ,min =0.0 )/100.0
    deficit_curr =torch .clamp (target -new_socs ,min =0.0 )/100.0
    window =max (float (urgency_steps ),1.0 )
    slack =remaining_steps
    if urgency_basis =="laxity":
        if full_power_soc_per_step is None :
            raise ValueError ("laxity urgency needs full_power_soc_per_step")
        steps_needed =(deficit_prev *100.0 )/torch .clamp (full_power_soc_per_step ,min =1e-6 )
        slack =remaining_steps -steps_needed
    urgency =1.0 +float (urgency_gain )*torch .clamp ((window -slack )/window ,0.0 ,1.0 )
    per_ev =(deficit_prev -deficit_curr )*urgency
    if surplus_coef :
        surplus_prev =torch .clamp (prev_socs -target ,min =0.0 )/100.0
        surplus_curr =torch .clamp (new_socs -target ,min =0.0 )/100.0
        per_ev =per_ev +(float (surplus_coef )/float (coef ))*(surplus_prev -surplus_curr )
    progress =per_ev .sum ()if reduction =="sum"else per_ev .mean ()
    station =torch .clamp (float (coef )*progress ,-float (clip ),float (clip ))
    if not return_per_ev :
        return station
    share =float (coef )*(per_ev if reduction =="sum"else per_ev /max (int (per_ev .numel ()),1 ))
    total =share .sum ()
    scale =torch .where (total .abs ()>1e-12 ,station /torch .where (total .abs ()>1e-12 ,total ,torch .ones_like (total )),torch .zeros_like (total ))
    return station ,share *scale


def soc_potential_rewards (
prev_socs ,new_socs ,target ,steps_left ,*,
coef :float =LOCAL_POTENTIAL_COEF ,
urgency_gain :float =LOCAL_POTENTIAL_URGENCY_GAIN ,
window :int =LOCAL_POTENTIAL_WINDOW_STEPS ,
gamma :float =LOCAL_POTENTIAL_GAMMA ,
):
    """Each EV's reward for one step under EVMA_LOCAL_REWARD_MODE=potential.

    steps_left is the number of action steps the EV has from this one on (1 on
    its last), so it has steps_left - 1 after acting. See EnvConfig for the form.
    """
    w =max (float (window ),1.0 )

    def urgency (k ):
        return 1.0 +float (urgency_gain )*torch .clamp ((w -k )/w ,0.0 ,1.0 )

    d_prev =torch .clamp (target -prev_socs ,min =0.0 )/100.0
    d_next =torch .clamp (target -new_socs ,min =0.0 )/100.0
    return float (coef )*(urgency (steps_left )*d_prev -float (gamma )*urgency (steps_left -1.0 )*d_next )


def departure_rewards (
final_socs ,target_socs ,*,
mode :str =LOCAL_DEPARTURE_REWARD_MODE ,
hit_reward :float =float (LOCAL_R_SOC_HIT )+max (float (SOC_HIT_BONUS ),0.0 ),
miss_penalty :float =LOCAL_DEPARTURE_MISS_PENALTY ,
linear_per_point :float =LOCAL_DEPARTURE_DEFICIT_PENALTY_LINEAR ,
smooth_linear :float =LOCAL_DEPARTURE_SMOOTH_LINEAR ,
smooth_quadratic :float =LOCAL_DEPARTURE_SMOOTH_QUADRATIC ,
):
    """Reward each departing EV once for the SoC it leaves with.

    "step": hit_reward at or above target, otherwise
    -(miss_penalty + linear_per_point * shortfall in SoC points). "smooth":
    hit_reward - smooth_linear * f - smooth_quadratic * f**2 with f the
    shortfall as a fraction, which equals hit_reward at the target.
    """
    shortfall_points =torch .clamp (target_socs -final_socs ,min =0.0 )
    if mode =="smooth":
        f =shortfall_points /100.0
        return float (hit_reward )-float (smooth_linear )*f -float (smooth_quadratic )*f *f
    missed =-(float (linear_per_point )*shortfall_points +float (miss_penalty )*(shortfall_points >0 ).float ())
    return torch .where (shortfall_points <=0 ,torch .full_like (final_socs ,float (hit_reward )),missed )

class EVEnv :





    def __init__ (
    self ,
    num_stations :int =NUM_STATIONS ,
    num_evs :int =NUM_EVS ,
    episode_steps :int =EPISODE_STEPS ,
    ):
        self .num_stations =num_stations
        self .num_evs =num_evs
        self .episode_steps =episode_steps
        # Plug-in SoC distribution for sessions that do not carry their own.
        self .soc_values ,self .soc_cdf =load_arrival_soc_cdf (EV_SOC_ARRIVAL_DISTRIBUTION_PATH )
        # Each station draws its EVs from its own measured sessions.
        station_ids =list (PER_STATION_SESSION_IDS )
        if len (station_ids )!=int (self .num_stations ):
            raise ValueError (
            "PER_STATION_SESSION_IDS must name one station per EVEnv station. "
            f"num_stations={self.num_stations}, stations={len(station_ids)}"
            )
        self .station_ids =station_ids
        self .station_pools =[
        load_station_pool (str (STATION_SESSION_DIR ),station_id ,int (SESSION_MATCH_MIN_CANDIDATES ))
        for station_id in station_ids
        ]
        self .arrival_growth =float (FUTURE_EV_ARRIVAL_GROWTH_MULTIPLIER )
        self .service_date =None
        self .day_class =WEEKDAY
        self .arrival_profiles_by_station =self ._class_arrival_probabilities (WEEKDAY )
        self ._arrival_probability_override =None
        self ._arrival_count_override =None
        self ._initial_evs_override =None
        self ._baseline_series =None
        self .current_regulation_kw =0.0
        self .initial_evs_by_station =[0 ]*self .num_stations
        self .day_context =np .zeros (self ._day_context_dim (),dtype =np .float32 )
        self .arrival_soc_log =[]
        self .arrival_needed_log =[]
        self .arrival_dwell_log =[]
        self .arrivals_this_step =0
        self .arrivals_by_station =[0 ]*self .num_stations


        self .ev_capacity =EV_CAPACITY
        self .tol_narrow_metrics =TOL_NARROW_METRICS
        self .balance_reward =GLOBAL_BALANCE_REWARD
        self .global_reward_on_system_output =bool (GLOBAL_REWARD_ON_SYSTEM_OUTPUT )
        self .balance_reward_mode =str (GLOBAL_BALANCE_REWARD_MODE ).strip ().lower ()
        if self .balance_reward_mode not in (
        "bounded_absolute_error","legacy_tolerance_band",
        ):
            raise ValueError (
            "GLOBAL_BALANCE_REWARD_MODE must be 'bounded_absolute_error' or "
            f"'legacy_tolerance_band', got {self.balance_reward_mode!r}"
            )
        self ._balance_reward_error_scale_kw =max (
        float (GLOBAL_BALANCE_REWARD_ERROR_SCALE_KW ),1e-6
        )
        self ._balance_reward_linear_tail_kw =float (GLOBAL_BALANCE_REWARD_LINEAR_TAIL_KW )
        # Kept only for explicit reproduction of the historical flat-top reward.
        self ._balance_reward_slope =(
            float (GLOBAL_BALANCE_REWARD_SLOPE )if float (GLOBAL_BALANCE_REWARD_SLOPE )>0
            else 2.0 *float (self .balance_reward )/max (float (self .tol_narrow_metrics ),1e-6 )
        )
        self .soc_wide =SOC_WIDE

        self .max_ev_per_station =MAX_EV_PER_STATION
        self .use_station_total_power_limit =bool (USE_STATION_TOTAL_POWER_LIMIT )
        self .station_total_power_limit_kw =float (STATION_MAX_TOTAL_POWER_KW )
        self .local_discharge_penalty_coef =0.0
        self .local_station_limit_penalty_coef =float (LOCAL_STATION_LIMIT_PENALTY )
        self .local_use_switch_features =bool (USE_SWITCHING_CONSTRAINTS )
        self .use_residual_bess =bool (USE_RESIDUAL_BESS )
        # The departure force floor applied inside apply_action during training
        # and interim tests. An evaluation pipeline that decides its own force
        # layer turns this off, so the process environment cannot add it.
        self .apply_train_force_floor =bool (TRAIN_FORCE_CHARGING )
        self .bess_power_limit_kw =float (BESS_POWER_KW )
        self .bess_energy_capacity_kwh =float (BESS_ENERGY_KWH )
        self .bess_initial_soc_pct =float (BESS_INITIAL_SOC_PCT )
        self .bess_target_soc_pct =float (BESS_TARGET_SOC_PCT )
        self .bess_min_soc_pct =float (BESS_MIN_SOC_PCT )
        self .bess_max_soc_pct =float (BESS_MAX_SOC_PCT )
        self .bess_charge_efficiency =float (BESS_CHARGE_EFFICIENCY )
        self .bess_discharge_efficiency =float (BESS_DISCHARGE_EFFICIENCY )
        self .use_central_ev_residual_allocator =bool (USE_CENTRAL_EV_RESIDUAL_ALLOCATOR )
        self .central_ev_allocator_waterfill_iters =int (CENTRAL_EV_ALLOCATOR_WATERFILL_ITERS )
        self .central_ev_allocator_departure_slack_kwh =float (CENTRAL_EV_ALLOCATOR_DEPARTURE_SLACK_KWH )
        self .station_power_limit_kw =torch .full (
        (num_stations ,),
        self .station_total_power_limit_kw ,
        dtype =torch .float32 ,
        device =device ,
        )


        self .ev_ids =torch .zeros ((num_stations ,self .max_ev_per_station ),dtype =torch .int32 ,device =device )
        self .soc =torch .zeros ((num_stations ,self .max_ev_per_station ),dtype =torch .float32 ,device =device )
        self .prev_soc =torch .zeros ((num_stations ,self .max_ev_per_station ),dtype =torch .float32 ,device =device )
        self .target =torch .zeros ((num_stations ,self .max_ev_per_station ),dtype =torch .float32 ,device =device )
        self .ev_capacity_kwh =torch .zeros ((num_stations ,self .max_ev_per_station ),dtype =torch .float32 ,device =device )
        self .ev_max_power_kw =torch .zeros ((num_stations ,self .max_ev_per_station ),dtype =torch .float32 ,device =device )
        self .ev_soc_pct_per_kwh =torch .zeros ((num_stations ,self .max_ev_per_station ),dtype =torch .float32 ,device =device )
        self .ev_soc_step_per_kw =torch .zeros ((num_stations ,self .max_ev_per_station ),dtype =torch .float32 ,device =device )
        self .ev_kwh_per_soc_pct =torch .zeros ((num_stations ,self .max_ev_per_station ),dtype =torch .float32 ,device =device )
        self .depart =torch .zeros ((num_stations ,self .max_ev_per_station ),dtype =torch .int32 ,device =device )
        self .ev_mask =torch .zeros ((num_stations ,self .max_ev_per_station ),dtype =torch .bool ,device =device )

        self .initial_remaining =torch .zeros ((num_stations ,self .max_ev_per_station ),dtype =torch .float32 ,device =device )

        self .arrival_step =torch .zeros ((num_stations ,self .max_ev_per_station ),dtype =torch .float32 ,device =device )

        self .switch_count =torch .zeros ((num_stations ,self .max_ev_per_station ),dtype =torch .int32 ,device =device )
        self .last_non_zero_state =torch .zeros ((num_stations ,self .max_ev_per_station ),dtype =torch .int32 ,device =device )


        self .stations_evs ={i :[]for i in range (self .num_stations )}


        self .metrics ={
        'total_steps':0 ,
        'surplus_within_narrow':0 ,
        'shortage_within_narrow':0 ,
        'surplus_steps':0 ,
        'shortage_steps':0 ,
        'station_limit_hits':0 ,
        'station_limit_steps':0 ,
        'station_charge_limit_hits':0 ,
        'station_discharge_limit_hits':0 ,
        'station_limit_penalty_total':0.0 ,
        'departing_evs':0 ,
        'departing_evs_soc_met':0 ,
        'total_switches_departed':0 ,
        'total_switches_current':0
        }




        self .net_demand_series =None
        self .net_demand_series_cpu =None
        self .current_net_demand =0.0
        self .current_observed_net_demand =0.0
        self .current_demand_clip_kw =0.0
        self .current_tracking_enabled =True
        self ._tracking_enabled_series =None
        self ._market_context_series ={}
        self .current_market_context ={name :0.0 for name in BID_CONTEXT_FEATURES }


        self .step_count =0
        self .used_ev_ids =set ()
        self .free_ev_ids =[]
        self .free_ev_id_ptr =0
        self ._free_ev_id_pos ={}
        self .active_evs_total =0
        self .active_ev_power_limit_kw =0.0
        self .last_station_powers =torch .zeros (self .num_stations ,dtype =torch .float32 ,device =device )
        self ._reset_bess_state ()
        self ._active_order_cache =[None ]*self .num_stations
        self .last_ev_local_rewards =None
        self ._departure_slots_by_step ={}





        self .record_snapshots =False


        self .reset_metrics ()





    def reset (
    self ,
    net_demand_series :np .ndarray |None =None ,
    arrival_probabilities_by_station :np .ndarray |None =None ,
    arrival_counts_by_station_step :np .ndarray |None =None ,
    initial_evs_by_station :np .ndarray |None =None ,
    day_context :np .ndarray |None =None ,
    tol_narrow_series :np .ndarray |None =None ,
    tracking_enabled_series :np .ndarray |None =None ,
    market_context_series :dict |None =None ,
    service_date =None ,
    baseline_series :np .ndarray |None =None ,
    ):
        """Start an episode.

        ``service_date`` sets the day class (weekday / holiday) that picks each
        station's arrival rate and session pool, and the previous days run to
        find the EVs present at 00:00; without it the day is a weekday.
        ``baseline_series`` is the submitted baseline per step; with it the
        metrics classify a step by the instruction's direction (request minus
        baseline) rather than by the sign of the request.
        """

        # Per-step tracking tolerance (kW). When set, begin_step() updates
        # self.tol_narrow_metrics each step so reward shaping and narrow-band
        # metrics follow the block's committed width (width augmentation).
        self ._tol_narrow_series =(
        None if tol_narrow_series is None
        else np .asarray (tol_narrow_series ,dtype =np .float32 ).reshape (-1 )
        )
        if self ._tol_narrow_series is None :
            self .tol_narrow_metrics =TOL_NARROW_METRICS
        self ._tracking_enabled_series =(
        None if tracking_enabled_series is None
        else np .asarray (tracking_enabled_series ,dtype =bool ).reshape (-1 )
        )
        raw_market_context =dict (market_context_series or {})
        self ._market_context_series ={
        name :np .asarray (raw_market_context .get (name ,[]),dtype =np .float32 ).reshape (-1 )
        for name in BID_CONTEXT_FEATURES
        }
        self .current_market_context ={
        name :(float (values [0 ])if values .size >0 else 0.0 )
        for name ,values in self ._market_context_series .items ()
        }

        # Carrying is a property of a continuing simulation, so it follows the
        # env by default. A caller that reuses one env for a set of independent
        # scenarios passes False, or scenario 2 would silently start from
        # whatever scenario 1 left plugged in.
        self .step_count =0


        if net_demand_series is None :



            try :
                from environment.readcsv import load_multiple_demand_files ,get_random_demand_episode
                all_demand_data =load_multiple_demand_files (train_split =25 )
                data_pool =all_demand_data .get ('train')or all_demand_data .get ('test')or []
                if not data_pool :
                    raise ValueError ("CSV demand data list is empty.")
                net_demand_series =get_random_demand_episode (data_pool ,self .episode_steps )
            except Exception as e :
                raise ValueError (
                "EVEnv.reset() requires either an explicit net_demand_series "
                "or loadable daily demand CSV files from Config.DEMAND_ADJUSTMENT_DIR."
                )from e


        self .net_demand_series_cpu =np .asarray (net_demand_series ,dtype =np .float32 ).reshape (-1 )
        self .net_demand_series =torch .as_tensor (self .net_demand_series_cpu ,dtype =torch .float32 ,device =device )
        self .current_net_demand =float (self .net_demand_series_cpu [0 ])if self .net_demand_series_cpu .size >0 else 0.0
        self .current_observed_net_demand =self .current_net_demand
        self .current_demand_clip_kw =0.0
        self .current_tracking_enabled =(
        True if self ._tracking_enabled_series is None or self ._tracking_enabled_series .size <=0
        else bool (self ._tracking_enabled_series [0 ])
        )


        self .ev_ids .zero_ ()
        self .soc .zero_ ()
        self .prev_soc .zero_ ()
        self .target .zero_ ()
        self .ev_capacity_kwh .zero_ ()
        self .ev_max_power_kw .zero_ ()
        self .ev_soc_pct_per_kwh .zero_ ()
        self .ev_soc_step_per_kw .zero_ ()
        self .ev_kwh_per_soc_pct .zero_ ()
        self .depart .zero_ ()
        self .ev_mask .fill_ (False )
        self .initial_remaining .zero_ ()
        self .arrival_step .zero_ ()
        self .switch_count .zero_ ()
        self .last_non_zero_state .zero_ ()
        self .last_station_powers .zero_ ()
        self ._reset_bess_state ()
        self .active_evs_total =0
        self .active_ev_power_limit_kw =0.0
        self ._invalidate_active_order_cache ()
        self ._departure_slots_by_step ={}


        self .stations_evs ={i :[]for i in range (self .num_stations )}
        self .arrival_soc_log =[]
        self .arrival_needed_log =[]
        self .arrival_dwell_log =[]
        self .arrivals_this_step =0
        self .arrivals_by_station =[0 ]*self .num_stations

        self .service_date =(
        None if service_date is None or str (service_date ).strip ()==""
        else str (service_date )[:10 ]
        )
        self .day_class =service_day_class (self .service_date ,SERVICE_CALENDAR_COUNTRY )
        self .arrival_profiles_by_station =self ._class_arrival_probabilities (self .day_class )
        self ._baseline_series =(
        None if baseline_series is None
        else np .asarray (baseline_series ,dtype =np .float64 ).reshape (-1 )
        )
        self .current_regulation_kw =0.0
        self ._arrival_probability_override =self ._coerce_arrival_probability_override (
        arrival_probabilities_by_station
        )
        self ._arrival_count_override =self ._coerce_arrival_count_override (
        arrival_counts_by_station_step
        )
        self ._initial_evs_override =self ._coerce_initial_evs_override (
        initial_evs_by_station
        )
        self .day_context =self ._coerce_day_context (day_context )
        self ._arrival_events =self ._pregenerate_arrival_events ()
        self ._reset_free_ev_ids ()
        initial_seed =random .randrange (2 **31 )
        if self ._initial_evs_override is not None :
            initial =self ._fresh_initial_evs (self ._initial_evs_override ,initial_seed )
        else :
            initial =self ._preroll_initial_evs (initial_seed )
        self .initial_evs_by_station =[len (evs )for evs in initial ]
        for st ,evs in enumerate (initial ):
            for slot_idx ,ev in enumerate (evs ):
                ev_id =self ._allocate_ev_id ()
                if ev_id is None :
                    raise RuntimeError ("EV id pool exhausted while placing the EVs present at 00:00")
                dep ,target_soc =self ._horizon_obligation (
                ev ['target_soc'],ev ['dep'],ev ['capacity_kwh'],ev ['max_power_kw'],
                ev ['arrival'],ev ['arrival_soc'],
                )
                self ._record_arrival (ev ['init_soc'],target_soc -ev ['init_soc'],dep )
                self ._set_ev_slot (
                st ,slot_idx ,ev_id ,ev ['init_soc'],target_soc ,dep ,
                capacity_kwh =ev ['capacity_kwh'],
                max_power_kw =ev ['max_power_kw'],
                profile_remaining =max (dep -self .step_count ,0 ),
                )


        self .reset_metrics ()

        return self ._get_obs ()

    def _coerce_arrival_probability_override (self ,values ):
        if values is None :
            return None
        arr =np .asarray (values ,dtype =np .float64 )
        if arr .shape !=(self .num_stations ,self .episode_steps ):
            raise ValueError (
            "arrival_probabilities_by_station must have shape "
            f"({self.num_stations}, {self.episode_steps}), got {arr.shape}"
            )
        return np .clip (arr ,0.0 ,1.0 )

    def _coerce_arrival_count_override (self ,values ):
        if values is None :
            return None
        arr =np .asarray (values ,dtype =np .float64 )
        if arr .shape !=(self .num_stations ,self .episode_steps ):
            raise ValueError (
            "arrival_counts_by_station_step must have shape "
            f"({self.num_stations}, {self.episode_steps}), got {arr.shape}"
            )
        return np .clip (np .rint (arr ),0.0 ,float (self .max_ev_per_station )).astype (np .int64 )

    def _coerce_initial_evs_override (self ,values ):
        if values is None :
            return None
        arr =np .asarray (values ,dtype =np .float64 ).reshape (-1 )
        if arr .shape !=(self .num_stations ,):
            raise ValueError (
            "initial_evs_by_station must have shape "
            f"({self.num_stations},), got {arr.shape}"
            )
        return np .clip (np .rint (arr ),0.0 ,float (self .max_ev_per_station )).astype (np .int64 )

    def _class_arrival_probabilities (self ,cls :str )->np .ndarray :
        """Per-step arrival probability on one charger, for every station, on a ``cls`` day."""
        return np .stack ([
        arrival_probabilities (pool ,cls ,growth =self .arrival_growth ,steps =self .episode_steps )
        for pool in self .station_pools
        ])

    def _session_attributes (self ,station :int ,cls :str ,step :int ,rng )->dict :
        """One EV arriving at ``station`` on step ``step`` (0-based) of a ``cls`` day.

        Dwell, delivered energy and, for residential stations, plug-in SoC come
        from one session of the station that started in the same clock hour;
        other stations draw plug-in SoC from the DESL distribution. Battery
        capacity and charger rating come from the fleet distributions. The
        target is plug-in SoC plus the delivered energy over this EV's battery,
        capped at a full battery and at what EV_TARGET_REACHABLE_POWER_FRACTION
        of its rated power adds during the stay.
        """
        pool =self .station_pools [int (station )]
        idx =draw_session (pool ,cls ,int (step ),self .episode_steps ,rng )
        sessions =pool .pools [cls ]
        capacity_kwh ,max_power_kw =self ._sample_physical_profile (rng )
        soc =float (sessions .arrival_soc_pct [idx ])
        if not np .isfinite (soc ):
            soc_idx =int (np .searchsorted (self .soc_cdf ,rng .random (),side ="right"))
            soc =float (self .soc_values [min (soc_idx ,len (self .soc_values )-1 )])
        init_soc =float (np .clip (soc ,0.0 ,100.0 ))
        dwell_steps =max (1 ,int (round (float (sessions .dwell_minutes [idx ])/float (TIME_STEP_MINUTES ))))
        soc_per_kwh =100.0 /max (float (capacity_kwh ),1e-6 )
        delivered_soc =max (float (sessions .energy_kwh [idx ]),0.0 )*soc_per_kwh
        reachable_soc =(
        init_soc
        +float (dwell_steps )*float (max_power_kw )*float (POWER_TO_ENERGY )*soc_per_kwh
        *float (EV_TARGET_REACHABLE_POWER_FRACTION )
        )
        target_soc =max (min (init_soc +delivered_soc ,reachable_soc ,100.0 ),init_soc )
        return {
        'init_soc':init_soc ,
        'target_soc':float (target_soc ),
        'dwell_steps':int (dwell_steps ),
        'needed_soc':float (target_soc -init_soc ),
        'profile_ev_id':None ,
        'capacity_kwh':float (capacity_kwh ),
        'max_power_kw':float (max_power_kw ),
        }

    def _horizon_obligation (
    self ,target_soc :float ,dep :int ,capacity_kwh :float ,max_power_kw :float ,
    arrival_step :int ,arrival_soc :float ,
    )->tuple [int ,float ]:
        """Departure step and target SoC the episode holds an EV to.

        An EV leaving after the last step is held, at the last step, to the SoC
        the next day assumes for it at 00:00: charged at its rated power from
        arrival until its target (the assumption _preroll_initial_evs makes).
        ``arrival_step`` is its first step on the service day's step count
        (zero or negative for an EV that arrived on a previous day). The two
        days then agree at midnight, and no energy the EV will need is spent
        today and left for tomorrow to restore. The level is never below what
        rated-power charging after the day needs to reach the target, since the
        target is reachable in the stay at a fraction of the rating.
        """
        horizon =int (self .episode_steps )
        if int (dep )<=horizon :
            return int (dep ),float (target_soc )
        charged_steps =max (horizon -int (arrival_step )+1 ,0 )
        charged_soc =(
        float (charged_steps )*float (max_power_kw )*float (POWER_TO_ENERGY )
        *100.0 /max (float (capacity_kwh ),1e-6 )
        )
        level =float (arrival_soc )+charged_soc
        if level <float (target_soc ):
            # Reached only by charging at the rating every step; a hair below
            # keeps float rounding from turning that into a miss.
            level -=1e-3
        else :
            level =float (target_soc )
        return horizon ,max (level ,0.0 )

    def _preroll_initial_evs (self ,seed :int )->list :
        """EVs still plugged in at 00:00, per station.

        The service day's arrival process runs for EV_PREROLL_DAYS previous days
        (each with its own day class) without control: an arrival takes a free
        charger if there is one and charges at its rated power until its
        target. The EVs whose stay reaches into the service day are returned
        with their SoC at 00:00 and their departure step counted on the service
        day (step 1 is its first step).
        """
        steps =int (self .episode_steps )
        classes =previous_day_classes (self .service_date ,int (EV_PREROLL_DAYS ),SERVICE_CALENDAR_COUNTRY )
        trial_rng =_pyrandom .Random (int (seed ))
        attribute_seed =(int (seed )*2654435761 +97 )%(2 **31 )
        max_evs =int (self .max_ev_per_station )
        present =[]
        for st in range (self .num_stations ):
            connected =[]
            for day_offset ,cls in enumerate (classes ):
                probs =arrival_probabilities (
                self .station_pools [st ],cls ,growth =self .arrival_growth ,steps =steps
                )
                first_step =(day_offset -len (classes ))*steps +1
                for t in range (steps ):
                    step_count =first_step +t
                    connected =[ev for ev in connected if ev ['dep']>=step_count ]
                    p =float (probs [t ])
                    if p <=0.0 :
                        continue
                    for slot_pos in range (max_evs ):
                        if trial_rng .random ()>=p or len (connected )>=max_evs :
                            continue
                        key =((day_offset *steps +t )*self .num_stations +st )*max_evs +slot_pos
                        ev =self ._session_attributes (st ,cls ,t ,_pyrandom .Random (attribute_seed +key ))
                        ev ['arrival']=step_count
                        ev ['dep']=step_count +ev ['dwell_steps']-1
                        connected .append (ev )
            midnight =[]
            for ev in connected :
                if ev ['dep']<1 :
                    continue
                charged_steps =1 -ev ['arrival']
                gained_soc =(
                charged_steps *ev ['max_power_kw']*float (POWER_TO_ENERGY )
                *100.0 /max (ev ['capacity_kwh'],1e-6 )
                )
                ev ['arrival_soc']=ev ['init_soc']
                ev ['init_soc']=float (max (min (ev ['init_soc']+gained_soc ,ev ['target_soc']),ev ['init_soc']))
                midnight .append (ev )
            present .append (midnight )
        return present

    def _fresh_initial_evs (self ,counts ,seed :int )->list :
        """A fixed number of EVs per station, each a session arriving at 00:00."""
        rng =_pyrandom .Random (int (seed ))
        out =[]
        for st in range (self .num_stations ):
            evs =[]
            for _ in range (int (counts [st ])):
                ev =self ._session_attributes (st ,self .day_class ,0 ,rng )
                ev ['dep']=int (ev ['dwell_steps'])
                ev ['arrival']=1
                ev ['arrival_soc']=ev ['init_soc']
                evs .append (ev )
            out .append (evs )
        return out

    def _day_context_dim (self )->int :
        if not bool (DAY_CONTEXT_USE_OBS ):
            return 0
        dim =len (CALENDAR_CONTEXT_FEATURES )
        if bool (DAY_CONTEXT_INCLUDE_WEATHER ):
            dim +=len (WEATHER_CONTEXT_FEATURES )
        return int (dim )

    def _coerce_day_context (self ,values ):
        dim =self ._day_context_dim ()
        if dim <=0 :
            return np .zeros (0 ,dtype =np .float32 )
        if values is None :
            return np .zeros (dim ,dtype =np .float32 )
        arr =np .asarray (values ,dtype =np .float32 ).reshape (-1 )
        if arr .size !=dim :
            raise ValueError (f"day_context must have length {dim}, got {arr.size}")
        return np .clip (arr ,-1.0 ,1.0 ).astype (np .float32 )


    def _reset_free_ev_ids (self ):
        """Shuffle EV IDs once per episode and allocate from that list in O(1)."""
        self .used_ev_ids =set ()
        self .free_ev_ids =list (range (int (self .num_evs )))
        random .shuffle (self .free_ev_ids )
        self .free_ev_id_ptr =0
        self ._free_ev_id_pos ={ev_id :idx for idx ,ev_id in enumerate (self .free_ev_ids )}

    def _reserve_ev_id (self ,ev_id :int )->bool :
        ev_id =int (ev_id )
        if ev_id <0 or ev_id >=int (self .num_evs )or ev_id in self .used_ev_ids :
            return False
        pos =self ._free_ev_id_pos .get (ev_id )
        if pos is None or pos <self .free_ev_id_ptr :
            return False

        ptr =self .free_ev_id_ptr
        swap_id =self .free_ev_ids [ptr ]
        self .free_ev_ids [ptr ],self .free_ev_ids [pos ]=self .free_ev_ids [pos ],self .free_ev_ids [ptr ]
        self ._free_ev_id_pos [ev_id ]=ptr
        self ._free_ev_id_pos [swap_id ]=pos
        self .free_ev_id_ptr =ptr +1
        self .used_ev_ids .add (ev_id )
        return True

    def _allocate_ev_id (self )->int |None :
        while self .free_ev_id_ptr <len (self .free_ev_ids ):
            ev_id =int (self .free_ev_ids [self .free_ev_id_ptr ])
            self .free_ev_id_ptr +=1
            if ev_id not in self .used_ev_ids :
                self .used_ev_ids .add (ev_id )
                return ev_id
        return None

    def _release_ev_id (self ,ev_id :int ):
        self .used_ev_ids .discard (int (ev_id ))




    def _remaining_action_steps (self ,depart_steps :torch .Tensor )->torch .Tensor :
        """Return how many control actions remain before the EV leaves."""
        return torch .clamp (depart_steps .float ()-float (self .step_count )+1.0 ,min =0.0 )

    def _reset_bess_state (self ):
        capacity =max (float (self .bess_energy_capacity_kwh ),1e-9 )
        initial_soc =float (np .clip (
        self .bess_initial_soc_pct ,self .bess_min_soc_pct ,self .bess_max_soc_pct
        ))
        self .bess_energy_kwh =capacity *initial_soc /100.0
        self .last_raw_actor_total_power_kw =0.0
        self .last_raw_actor_residual_kw =0.0
        self .last_central_correction_power_kw =0.0
        self .last_central_ev_total_power_kw =0.0
        self .last_pre_bess_residual_kw =0.0
        self .last_bess_requested_power_kw =0.0
        self .last_bess_power_kw =0.0
        self .last_pcc_power_kw =0.0
        self .last_post_bess_residual_kw =0.0

    def _bess_soc_pct (self )->float :
        return float (100.0 *self .bess_energy_kwh /max (
        float (self .bess_energy_capacity_kwh ),1e-9
        ))

    def _bess_context_values (self )->dict :
        return {
        'last_raw_actor_total_power_kw':float (self .last_raw_actor_total_power_kw ),
        'last_raw_actor_residual_kw':float (self .last_raw_actor_residual_kw ),
        'last_central_correction_power_kw':float (self .last_central_correction_power_kw ),
        'last_central_ev_total_power_kw':float (self .last_central_ev_total_power_kw ),
        'last_pre_bess_residual_kw':float (self .last_pre_bess_residual_kw ),
        'last_bess_power_kw':float (self .last_bess_power_kw ),
        'last_pcc_power_kw':float (self .last_pcc_power_kw ),
        'last_post_bess_residual_kw':float (self .last_post_bess_residual_kw ),
        'bess_soc_pct':self ._bess_soc_pct (),
        }

    def bess_feasible_power_bounds_kw (self )->tuple [float ,float ]:
        """Return current PCC-sign BESS bounds ``(min_kw, max_kw)``.

        Negative is discharge/export and positive is charge/import.  The
        bounds include power, energy, SoC, and one-way efficiency limits for
        the next environment interval.
        """
        if not bool (self .use_residual_bess ):
            return 0.0 ,0.0
        dt_hours =max (float (POWER_TO_ENERGY ),1e-9 )
        capacity_kwh =max (float (self .bess_energy_capacity_kwh ),1e-9 )
        min_energy_kwh =capacity_kwh *float (self .bess_min_soc_pct )/100.0
        max_energy_kwh =capacity_kwh *float (self .bess_max_soc_pct )/100.0
        energy_kwh =float (np .clip (
        self .bess_energy_kwh ,min_energy_kwh ,max_energy_kwh
        ))
        power_limit_kw =max (float (self .bess_power_limit_kw ),0.0 )
        max_discharge_kw =min (
        power_limit_kw ,max (
        0.0 ,(energy_kwh -min_energy_kwh )
        *float (self .bess_discharge_efficiency )/dt_hours
        ))
        max_charge_kw =min (
        power_limit_kw ,max (
        0.0 ,(max_energy_kwh -energy_kwh )
        /(max (float (self .bess_charge_efficiency ),1e-9 )*dt_hours )
        ))
        return -float (max_discharge_kw ),float (max_charge_kw )

    def bess_sustained_capacity_kw (
    self ,duration_hours :float =0.5
    )->tuple [float ,float ]:
        """Return sustained ``(charge_kw, discharge_kw)`` capability.

        This is the BESS contribution to Assessment I.  It is deliberately
        separate from the one-step dispatch bounds because a five-minute SoC
        margin cannot certify a thirty-minute reserve award.
        """
        if not bool (self .use_residual_bess ):
            return 0.0 ,0.0
        duration =max (float (duration_hours ),1e-9 )
        capacity_kwh =max (float (self .bess_energy_capacity_kwh ),1e-9 )
        min_energy_kwh =capacity_kwh *float (self .bess_min_soc_pct )/100.0
        max_energy_kwh =capacity_kwh *float (self .bess_max_soc_pct )/100.0
        energy_kwh =float (np .clip (
        self .bess_energy_kwh ,min_energy_kwh ,max_energy_kwh
        ))
        power_limit_kw =max (float (self .bess_power_limit_kw ),0.0 )
        charge_kw =min (
        power_limit_kw ,max (0.0 ,max_energy_kwh -energy_kwh )
        /(max (float (self .bess_charge_efficiency ),1e-9 )*duration )
        )
        discharge_kw =min (
        power_limit_kw ,max (0.0 ,energy_kwh -min_energy_kwh )
        *float (self .bess_discharge_efficiency )/duration
        )
        return float (charge_kw ),float (discharge_kw )

    def _dispatch_residual_bess (
    self ,request_kw :float ,ev_total_power_kw :float ,tracking_enabled :bool
    )->dict :
        """Apply the independent PCC-side residual controller for one step.

        The sign follows the EV/PCC convention used everywhere else in the
        environment: positive power is grid import (BESS charging), negative
        power is grid export (BESS discharging). EV state and EV rewards have
        already been determined before this is called, so this controller
        cannot silently rewrite an actor transition.
        """
        request_kw =float (request_kw )
        ev_total_power_kw =float (ev_total_power_kw )
        pre_residual_kw =request_kw -ev_total_power_kw
        dt_hours =max (float (POWER_TO_ENERGY ),1e-9 )
        capacity_kwh =max (float (self .bess_energy_capacity_kwh ),1e-9 )
        min_energy_kwh =capacity_kwh *float (self .bess_min_soc_pct )/100.0
        max_energy_kwh =capacity_kwh *float (self .bess_max_soc_pct )/100.0
        target_energy_kwh =capacity_kwh *float (self .bess_target_soc_pct )/100.0
        energy_before_kwh =float (np .clip (
        self .bess_energy_kwh ,min_energy_kwh ,max_energy_kwh
        ))

        if bool (tracking_enabled ):
            requested_power_kw =pre_residual_kw
        elif energy_before_kwh <target_energy_kwh :
            requested_power_kw =(target_energy_kwh -energy_before_kwh )/(
            max (float (self .bess_charge_efficiency ),1e-9 )*dt_hours
            )
        elif energy_before_kwh >target_energy_kwh :
            requested_power_kw =-(
            (energy_before_kwh -target_energy_kwh )
            *float (self .bess_discharge_efficiency )/dt_hours
            )
        else :
            requested_power_kw =0.0

        power_limit_hit =False
        energy_limit_hit =False
        actual_power_kw =0.0
        if bool (self .use_residual_bess ):
            power_limit_kw =max (float (self .bess_power_limit_kw ),0.0 )
            power_limited_kw =float (np .clip (
            requested_power_kw ,-power_limit_kw ,power_limit_kw
            ))
            power_limit_hit =abs (power_limited_kw -requested_power_kw )>1e-6

            min_bess_power_kw ,max_bess_power_kw =self .bess_feasible_power_bounds_kw ()
            actual_power_kw =float (np .clip (
            power_limited_kw ,min_bess_power_kw ,max_bess_power_kw
            ))
            energy_limit_hit =abs (actual_power_kw -power_limited_kw )>1e-6

        if actual_power_kw >=0.0 :
            energy_after_kwh =energy_before_kwh +(
            actual_power_kw *float (self .bess_charge_efficiency )
            )*dt_hours
        else :
            energy_after_kwh =energy_before_kwh -(
            -actual_power_kw /max (float (self .bess_discharge_efficiency ),1e-9 )
            )*dt_hours
        energy_after_kwh =float (np .clip (
        energy_after_kwh ,min_energy_kwh ,max_energy_kwh
        ))

        pcc_power_kw =ev_total_power_kw +actual_power_kw
        post_residual_kw =request_kw -pcc_power_kw
        self .bess_energy_kwh =energy_after_kwh
        self .last_pre_bess_residual_kw =pre_residual_kw
        self .last_bess_requested_power_kw =requested_power_kw
        self .last_bess_power_kw =actual_power_kw
        self .last_pcc_power_kw =pcc_power_kw
        self .last_post_bess_residual_kw =post_residual_kw

        self .metrics ['bess_charge_energy_kwh']+=max (actual_power_kw ,0.0 )*dt_hours
        self .metrics ['bess_discharge_energy_kwh']+=max (-actual_power_kw ,0.0 )*dt_hours
        self .metrics ['bess_throughput_kwh']+=abs (actual_power_kw )*dt_hours
        self .metrics ['bess_power_limit_hits']+=int (power_limit_hit )
        self .metrics ['bess_energy_limit_hits']+=int (energy_limit_hit )
        self .metrics ['bess_max_abs_power_kw']=max (
        self .metrics ['bess_max_abs_power_kw'],abs (actual_power_kw )
        )
        if bool (tracking_enabled ):
            self .metrics ['pre_bess_abs_error_sum_kw']+=abs (pre_residual_kw )
            self .metrics ['post_bess_abs_error_sum_kw']+=abs (post_residual_kw )

        return {
        'requested_power_kw':requested_power_kw ,
        'power_kw':actual_power_kw ,
        'energy_before_kwh':energy_before_kwh ,
        'energy_after_kwh':energy_after_kwh ,
        'soc_pct':self ._bess_soc_pct (),
        'pcc_power_kw':pcc_power_kw ,
        'pre_residual_kw':pre_residual_kw ,
        'post_residual_kw':post_residual_kw ,
        'power_limit_hit':power_limit_hit ,
        'energy_limit_hit':energy_limit_hit ,
        }

    def _get_obs (self )->np .ndarray :

        ev_block_dim =self .max_ev_per_station *EV_FEAT_DIM
        tail_features =[]

        if LOCAL_DEMAND_STEPS >0 and self .net_demand_series is not None :
            L_local =int (LOCAL_DEMAND_STEPS )
            start_idx =self .step_count -1
            indices =torch .arange (start_idx ,start_idx +L_local ,dtype =torch .long ,device =device )
            valid_mask =(indices >=0 )&(indices <len (self .net_demand_series ))
            ag_tensor =torch .zeros (L_local ,dtype =torch .float32 ,device =device )
            if valid_mask .any ():
                ag_tensor [valid_mask ]=self ._clip_demand_for_obs (self .net_demand_series [indices [valid_mask ]])
            tail_features .append (ag_tensor )

        if LOCAL_USE_TRACKING_ENABLED :
            tail_features .append (
            torch .as_tensor (
            [1.0 if self .current_tracking_enabled else 0.0 ],
            dtype =torch .float32 ,device =device ,
            )
            )

        if LOCAL_USE_FLEET_RESIDUAL :
            # The fleet's own miss on the previous step, one scalar, identical
            # for every station. Nothing else in the observation reports it.
            tail_features .append (
            torch .as_tensor (
            [float (getattr (self ,'last_raw_actor_residual_kw',0.0 ))],
            dtype =torch .float32 ,device =device ,
            )
            )

        if LOWER_BID_CONTEXT_USE_OBS :
            tail_features .append (
            torch .as_tensor (
            [float (self .current_market_context .get (name ,0.0 ))for name in BID_CONTEXT_FEATURES ],
            dtype =torch .float32 ,device =device ,
            )
            )

        if BESS_CONTEXT_USE_OBS :
            bess_context =self ._bess_context_values ()
            tail_features .append (
            torch .as_tensor (
            [float (bess_context [name ])for name in BESS_CONTEXT_FEATURES ],
            dtype =torch .float32 ,device =device ,
            )
            )

        if LOCAL_USE_STEP :
            tail_features .append (
            torch .as_tensor ([float (self .step_count )],dtype =torch .float32 ,device =device )
            )
        if bool (DAY_CONTEXT_USE_OBS )and len (self .day_context )>0 :
            tail_features .append (
            torch .as_tensor (self .day_context ,dtype =torch .float32 ,device =device )
            )

        tail_vec =torch .cat (tail_features )if tail_features else None
        tail_dim =0 if tail_vec is None else int (tail_vec .numel ())
        obs =torch .zeros (
        (self .num_stations ,ev_block_dim +tail_dim ),
        dtype =torch .float32 ,
        device =device ,
        )
        if tail_vec is not None :
            obs [:,ev_block_dim :]=tail_vec

        for st in range (self .num_stations ):
            sorted_active_evs =self ._get_sorted_active_evs (st )
            k =int (sorted_active_evs .numel ())
            if k <=0 :
                continue

            ev_matrix =obs [st ,:ev_block_dim ].reshape (self .max_ev_per_station ,EV_FEAT_DIM )
            ev_view =ev_matrix [:k ]
            ev_socs =torch .clamp (self .soc [st ,sorted_active_evs ],min =1e-3 )
            ev_view [:,0 ]=1.0
            ev_view [:,1 ]=ev_socs
            ev_view [:,2 ]=self ._remaining_action_steps (self .depart [st ,sorted_active_evs ])
            ev_view [:,3 ]=self .target [st ,sorted_active_evs ]-ev_socs

            col_idx =4
            if USE_HETEROGENEOUS_EV_PHYSICS :
                ev_view [:,col_idx ]=self .ev_capacity_kwh [st ,sorted_active_evs ]
                col_idx +=1
                ev_view [:,col_idx ]=self .ev_max_power_kw [st ,sorted_active_evs ]
                col_idx +=1

            if self .local_use_switch_features :
                ev_view [:,col_idx ]=self .switch_count [st ,sorted_active_evs ].float ()
                col_idx +=1
                ev_view [:,col_idx ]=self .last_non_zero_state [st ,sorted_active_evs ].float ()

        return obs

    def _clip_demand_for_obs (self ,demand_values :torch .Tensor )->torch .Tensor :
        return demand_values .to (dtype =torch .float32 ,device =device )

    def _record_arrival (self ,init_soc :float ,needed_soc :float ,dwell_steps :int ):
        if needed_soc <0 :
            needed_soc =0.0
        self .arrival_soc_log .append (float (init_soc ))
        self .arrival_needed_log .append (float (needed_soc ))
        dwell_hours =float (max (dwell_steps ,0 ))*TIME_STEP_MINUTES /60.0
        self .arrival_dwell_log .append (dwell_hours )

    def _sample_physical_profile (self ,rng =random )->tuple [float ,float ]:
        """Sample one EV battery capacity and one charger charge/discharge limit."""
        if not USE_HETEROGENEOUS_EV_PHYSICS :
            return float (EV_CAPACITY ),float (MAX_EV_POWER_KW )

        capacity_kwh =rng .choices (
        list (EV_BATTERY_CAPACITY_OPTIONS_KWH ),
        weights =list (EV_BATTERY_CAPACITY_PROBS ),
        k =1 ,
        )[0 ]
        max_power_kw =rng .choices (
        list (EV_CHARGER_MAX_POWER_OPTIONS_KW ),
        weights =list (EV_CHARGER_MAX_POWER_PROBS ),
        k =1 ,
        )[0 ]
        return float (capacity_kwh ),float (max_power_kw )

    def _set_ev_slot (
    self ,
    station :int ,
    slot_idx :int ,
    ev_id :int ,
    init_soc :float ,
    target_soc :float ,
    dep :int ,
    capacity_kwh :float ,
    max_power_kw :float ,
    profile_remaining :float |None =None ,
    ):


        if profile_remaining is None :
            profile_remaining =max (dep -self .step_count ,0 )
        dep =max (self .step_count ,int (dep ))

        self .ev_ids [station ,slot_idx ]=ev_id
        self .soc [station ,slot_idx ]=init_soc
        self .prev_soc [station ,slot_idx ]=init_soc
        self .target [station ,slot_idx ]=target_soc
        self .ev_capacity_kwh [station ,slot_idx ]=capacity_kwh
        self .ev_max_power_kw [station ,slot_idx ]=max_power_kw
        safe_capacity_kwh =max (float (capacity_kwh ),1e-6 )
        soc_pct_per_kwh =100.0 /safe_capacity_kwh
        self .ev_soc_pct_per_kwh [station ,slot_idx ]=soc_pct_per_kwh
        self .ev_soc_step_per_kw [station ,slot_idx ]=float (POWER_TO_ENERGY )*soc_pct_per_kwh
        self .ev_kwh_per_soc_pct [station ,slot_idx ]=safe_capacity_kwh /100.0
        self .depart [station ,slot_idx ]=dep
        self .ev_mask [station ,slot_idx ]=True
        self ._departure_slots_by_step .setdefault (int (dep ),[]).append ((int (station ),int (slot_idx )))
        self .active_evs_total +=1
        self .active_ev_power_limit_kw +=float (max_power_kw )
        self ._invalidate_active_order_cache (station )
        self .initial_remaining [station ,slot_idx ]=float (profile_remaining )
        self .arrival_step [station ,slot_idx ]=float (self .step_count )
        self .switch_count [station ,slot_idx ]=0
        self .last_non_zero_state [station ,slot_idx ]=0


        self .stations_evs [station ]=[ev for ev in self .stations_evs [station ]if ev .get ('id')!=ev_id ]
        ev =dict (
        id =ev_id ,station =station ,depart =dep ,soc =init_soc ,
        target =target_soc ,prev_soc =init_soc ,
        battery_capacity_kwh =capacity_kwh ,max_power_kw =max_power_kw ,
        )
        self .stations_evs [station ].append (ev )

    def _apply_station_power_limit (
    self ,
    station :int ,
    effective_power_kw :torch .Tensor ,
    )->tuple [torch .Tensor ,torch .Tensor ,torch .Tensor ]:
        """Apply a symmetric site-level import/export cap to one station."""
        if (not self .use_station_total_power_limit )or effective_power_kw .numel ()==0 :
            limited =torch .zeros ((),dtype =torch .bool ,device =device )
            return effective_power_kw ,limited ,limited

        limit_kw =self .station_power_limit_kw [station ]
        charge_mask =effective_power_kw >0
        discharge_mask =effective_power_kw <0

        total_charge_kw =effective_power_kw [charge_mask ].sum ()
        total_discharge_kw =(-effective_power_kw [discharge_mask ]).sum ()

        charge_limited =total_charge_kw >(limit_kw +1e-6 )
        discharge_limited =total_discharge_kw >(limit_kw +1e-6 )

        charge_scale =torch .where (
        charge_limited ,
        limit_kw /torch .clamp (total_charge_kw ,min =1e-6 ),
        torch .ones_like (total_charge_kw ),
        )
        effective_power_kw =torch .where (
        charge_mask ,
        effective_power_kw *charge_scale ,
        effective_power_kw ,
        )

        discharge_scale =torch .where (
        discharge_limited ,
        limit_kw /torch .clamp (total_discharge_kw ,min =1e-6 ),
        torch .ones_like (total_discharge_kw ),
        )
        effective_power_kw =torch .where (
        discharge_mask ,
        effective_power_kw *discharge_scale ,
        effective_power_kw ,
        )

        return effective_power_kw ,charge_limited ,discharge_limited

    def _pregenerate_arrival_events (self ):
        """Pre-sample arrival events for every step and station in the episode."""
        events =[[[]for _ in range (self .num_stations )]for _ in range (self .episode_steps )]
        max_evs =int (self .max_ev_per_station )
        count_override =getattr (self ,"_arrival_count_override",None )
        prob_override =getattr (self ,"_arrival_probability_override",None )
        # Both streams follow from the episode seed, so a run is reproducible;
        # they are simply independent of one another.
        stream_seed =random .randrange (2 **31 )
        arrival_rng =_pyrandom .Random (stream_seed )
        attribute_seed =(stream_seed *2654435761 )%(2 **31 )
        for step_idx in range (self .episode_steps ):
            for st in range (self .num_stations ):
                slot_base =(step_idx *self .num_stations +st )*max_evs
                if count_override is not None :
                    n_arrivals =int (count_override [st ,step_idx ])
                    events [step_idx ][st ]=[
                    self ._session_attributes (
                    st ,self .day_class ,step_idx ,_pyrandom .Random (attribute_seed +slot_base +k )
                    )
                    for k in range (n_arrivals )
                    ]
                    continue
                # Each active slot performs an independent Bernoulli arrival
                # trial using that station's time-varying probability curve.
                #
                # The arrival trial and the EV's attributes are drawn from
                # separate streams, each keyed by (step, station, slot). Sharing
                # one stream made the arrival rate impossible to perturb: one
                # extra arrival consumed the draws that every later trial and
                # every later EV's attributes would have used, so raising the
                # rate by a few per cent returned an unrelated day rather than
                # the same day with a few more cars. Sensitivity to the arrival
                # forecast could not be measured at all -- what looked like
                # "5 fewer EVs breaks the bid" was a different fleet entirely.
                # Keyed streams make the arrival sets nested in the rate, which
                # is what a sensitivity test on the arrival rate needs.
                prof =prob_override [st ]if prob_override is not None else self .arrival_profiles_by_station [st ]
                arrival_threshold =float (prof [min (step_idx ,len (prof )-1 )])
                station_events =[None ]*max_evs
                if arrival_threshold <=0.0 :
                    events [step_idx ][st ]=station_events
                    continue
                for slot_pos in range (max_evs ):
                    if arrival_rng .random ()>=arrival_threshold :
                        continue
                    station_events [slot_pos ]=self ._session_attributes (
                    st ,self .day_class ,step_idx ,_pyrandom .Random (attribute_seed +slot_base +slot_pos )
                    )
                events [step_idx ][st ]=station_events
        return events

    def _handle_arrivals (self ):
        """Spawn all EVs scheduled to arrive at the current step."""
        total_arrivals =0
        arrivals_by_station =[0 ]*self .num_stations
        step_idx =self .step_count -1
        if step_idx <0 or step_idx >=len (self ._arrival_events ):
            self .arrivals_this_step =0
            self .arrivals_by_station =arrivals_by_station
            return
        for st in range (self .num_stations ):
            station_events =self ._arrival_events [step_idx ][st ]
            if not station_events :
                continue
            if not any (ev_event is not None for ev_event in station_events ):
                continue
            empty_slots =torch .nonzero (~self .ev_mask [st ],as_tuple =False ).squeeze (-1 )
            empty_slot_list =[int (x )for x in empty_slots .detach ().cpu ().tolist ()]
            avail_slots =len (empty_slot_list )
            if avail_slots <=0 :
                continue
            # ``station_events`` is a Bernoulli candidate list, so ``None``
            # entries do not consume an available charger.  Limiting the loop
            # to the first ``avail_slots`` candidates silently discarded real
            # arrivals behind an early ``None`` whenever a station was partly
            # occupied.  Scan the frozen candidates and stop only when the
            # physical slots are exhausted.
            for ev_event in station_events :
                if ev_event is None :
                    continue
                if not empty_slot_list :
                    break
                chosen_pos =random .randint (0 ,len (empty_slot_list )-1 )
                slot_idx =empty_slot_list [chosen_pos ]
                if self ._spawn_ev_from_event_at_slot (st ,ev_event ,slot_idx ):
                    empty_slot_list .pop (chosen_pos )
                    total_arrivals +=1
                    arrivals_by_station [st ]+=1
        self .arrivals_this_step =total_arrivals
        self .arrivals_by_station =arrivals_by_station

    def step (self ,actions ):
        """Advance the environment by one step using the provided actions."""
        self .begin_step ()
        observation ,local_rewards ,global_reward ,done ,info =self .apply_action (actions )
        return observation ,local_rewards ,global_reward ,done ,info

    def begin_step (self ):
        """
        Advance internal time, update the current demand request, and process arrivals.

        This method prepares all per-step state needed before `apply_action()`
        is called, and optionally records pre-action snapshots.
        """
        self .step_count +=1



        demand_idx =self .step_count -1
        if getattr (self ,'_tol_narrow_series',None ) is not None and 0 <=demand_idx <len (self ._tol_narrow_series ):
            self .tol_narrow_metrics =max (float (self ._tol_narrow_series [demand_idx ]),1e-6 )
        self .current_tracking_enabled =(
        True
        if getattr (self ,'_tracking_enabled_series',None ) is None
        else bool (
        0 <=demand_idx <len (self ._tracking_enabled_series )
        and self ._tracking_enabled_series [demand_idx ]
        )
        )
        for name ,values in getattr (self ,'_market_context_series',{}).items ():
            self .current_market_context [name ]=(
            float (values [demand_idx ])if 0 <=demand_idx <len (values )else 0.0
            )

        if (
        self .net_demand_series_cpu is not None
        and 0 <=demand_idx <len (self .net_demand_series_cpu )
        ):
            raw_net_demand =float (self .net_demand_series_cpu [demand_idx ])
        else :
            raw_net_demand =0.0

        self .arrivals_this_step =0
        self .arrivals_by_station =[0 ]*self .num_stations

        self ._handle_arrivals ()


        observed_net_demand =raw_net_demand

        # The actor, reward, metrics and market settlement all use the complete
        # committed request.
        self .current_net_demand =raw_net_demand
        self .current_observed_net_demand =observed_net_demand
        # The instruction relative to the submitted baseline: positive is a down
        # instruction (consume more, absorbing surplus), negative an up
        # instruction (consume less, supplying). Without a baseline the sign of
        # the request itself stands in.
        baseline =getattr (self ,'_baseline_series',None )
        if baseline is not None and 0 <=demand_idx <len (baseline ):
            self .current_regulation_kw =raw_net_demand -float (baseline [demand_idx ])
        else :
            self .current_regulation_kw =raw_net_demand
        self .current_demand_clip_kw =raw_net_demand -observed_net_demand


        self ._snapshot_pre ={}
        if self .record_snapshots :
            for st in range (self .num_stations ):
                details =[]
                sorted_evs =self ._get_sorted_active_evs (st )
                if sorted_evs .numel ()>0 :
                    ids =self .ev_ids [st ,sorted_evs ]
                    socs =self .soc [st ,sorted_evs ]

                    remains =self ._remaining_action_steps (self .depart [st ,sorted_evs ])

                    needs =self .target [st ,sorted_evs ]-socs
                    for i in range (int (sorted_evs .numel ())):
                        details .append ({
                        'id':int (ids [i ].item ()),
                        'soc':float (socs [i ].item ()),
                        'remaining_time':float (remains [i ].item ()),
                        'needed_soc':float (needs [i ].item ()),
                        'target_soc':float (self .target [st ,sorted_evs ][i ].item ()),
                        'battery_capacity_kwh':float (self .ev_capacity_kwh [st ,sorted_evs ][i ].item ()),
                        'max_power_kw':float (self .ev_max_power_kw [st ,sorted_evs ][i ].item ()),
                        'switch_count':int (self .switch_count [st ,sorted_evs ][i ].item ()),
                        })
                self ._snapshot_pre [st ]=details
        return self ._get_obs ()

    def apply_action (
    self ,actions ,build_info :bool =True ,return_observation :bool =True ,
    include_active_evs :bool =True ,include_ev_physics :bool =True ,
    include_reward_breakdown :bool =True ,
    ):
        """
        Apply one step of charging/discharging actions.

        Args:
            actions: Tensor-like action array shaped per station and EV slot.
            build_info: When False, return a lightweight `info` payload for
                faster training-time rollouts. When True, include detailed
                bookkeeping used by evaluation and plotting.
            include_active_evs: Include the per-station active-EV count in the
                detailed info payload. Runtime execution needs it; training/test
                plots do not.
            include_ev_physics: Include cloned EV capacity/power tensors in the
                detailed info payload. This is diagnostic-only and can be
                omitted by faster evaluation paths.
            include_reward_breakdown: Include per-station reward component
                lists in the detailed info payload. Detailed reward plots need
                it, but station cooperation plots do not.

        Returns:
            observation, local_rewards, global_reward, done, info
        """
        if not torch .is_tensor (actions ):
            actions =torch .as_tensor (actions ,dtype =torch .float32 ,device =device )
        else :
            actions =actions .to (device =device ,dtype =torch .float32 )
        if actions .dim ()!=2 :
            raise ValueError (
            f"actions must have shape [num_stations, max_ev_per_station], got {tuple(actions.shape)}"
            )
        forced_points_by_station =None
        if getattr (self ,'apply_train_force_floor',TRAIN_FORCE_CHARGING ):
            actions ,forced_points_by_station =self .apply_force_floor (actions )
        actor_actions =actions .clone ()
        actions ,central_allocator_info =allocate_central_ev_residual (
        self ,actor_actions ,
        request_kw =self .current_net_demand ,
        tracking_enabled =self .current_tracking_enabled ,
        enabled =self .use_central_ev_residual_allocator ,
        departure_slack_kwh =self .central_ev_allocator_departure_slack_kwh ,
        waterfill_iterations =self .central_ev_allocator_waterfill_iters ,
        )
        raw_actor_ev_power_kw_tensor =central_allocator_info ['raw_actor_ev_power_kw']
        raw_actor_station_powers_tensor =central_allocator_info ['raw_actor_station_powers']
        # Rewards are evaluated on the actor proposal.  The central allocator
        # is allowed to change the physical transition, but must not make the
        # actor look locally better than the action it actually emitted.
        raw_reward_soc_tensor =self .soc .clone ()

        local_rewards_tensor =torch .zeros (self .num_stations ,dtype =torch .float32 ,device =device )
        if forced_points_by_station is not None and LOCAL_FORCED_PENALTY_PER_POINT :
            local_rewards_tensor -=float (LOCAL_FORCED_PENALTY_PER_POINT )*forced_points_by_station
        progress_shaping_rewards_tensor =torch .zeros (self .num_stations ,dtype =torch .float32 ,device =device )
        departing_ev_stats =[0 ,0 ]
        current_request =self .current_net_demand



        station_powers_tensor =torch .zeros (self .num_stations ,dtype =torch .float32 ,device =device )
        discharge_penalties_tensor =torch .zeros (self .num_stations ,dtype =torch .float32 ,device =device )
        switch_penalties_tensor =torch .zeros (self .num_stations ,dtype =torch .float32 ,device =device )
        station_limit_penalties_tensor =torch .zeros (self .num_stations ,dtype =torch .float32 ,device =device )
        station_charge_limit_hits_tensor =torch .zeros (self .num_stations ,dtype =torch .bool ,device =device )
        station_discharge_limit_hits_tensor =torch .zeros (self .num_stations ,dtype =torch .bool ,device =device )


        actual_ev_power_kw_tensor =torch .zeros ((self .num_stations ,self .max_ev_per_station ),device =device )
        # Per-EV local rewards, indexed like the observation's EV slots, and the
        # slot each physical EV index occupied in that observation.
        ev_local_rewards =None
        physical_to_slot =None
        if LOCAL_CRITIC_PER_EV :
            ev_local_rewards =torch .zeros ((self .num_stations ,self .max_ev_per_station ),dtype =torch .float32 ,device =device )
            physical_to_slot =torch .full ((self .num_stations ,self .max_ev_per_station ),-1 ,dtype =torch .long ,device =device )
        snapshot_after ={}
        for st in range (self .num_stations ):
            # Process one station independently, using only currently active EV slots.
            sorted_active_evs =self ._get_sorted_active_evs (st )
            if physical_to_slot is not None and sorted_active_evs .numel ()>0 :
                physical_to_slot [st ,sorted_active_evs ]=torch .arange (
                int (sorted_active_evs .numel ()),dtype =torch .long ,device =device
                )

            if sorted_active_evs .numel ()>0 :
                active_count =int (sorted_active_evs .numel ())
                actions_for_station =actions [st ,:active_count ]
                ev_capacity_kwh =self .ev_capacity_kwh [st ,sorted_active_evs ]
                ev_max_power_kw =self .ev_max_power_kw [st ,sorted_active_evs ]
                ev_soc_step_per_kw =self .ev_soc_step_per_kw [st ,sorted_active_evs ]
                ev_soc_pct_per_kwh =self .ev_soc_pct_per_kwh [st ,sorted_active_evs ]
                ev_kwh_per_soc_pct =self .ev_kwh_per_soc_pct [st ,sorted_active_evs ]
                scaled_power_kw =torch .clamp (
                actions_for_station ,-1.0 ,1.0
                )*ev_max_power_kw

                prev_socs =self .soc [st ,sorted_active_evs ].clone ()
                raw_power_kw =raw_actor_ev_power_kw_tensor [st ,:active_count ]
                raw_delta_kwh =raw_power_kw *POWER_TO_ENERGY
                raw_new_socs =torch .clamp (
                prev_socs +raw_delta_kwh *ev_soc_pct_per_kwh ,0.0 ,100.0
                )
                raw_reward_soc_tensor [st ,sorted_active_evs ]=raw_new_socs

                proposed_delta_soc =scaled_power_kw *ev_soc_step_per_kw
                proposed_delta_soc [torch .abs (proposed_delta_soc )<1e-7 ]=0.0

                if self .local_use_switch_features :
                    current_state =torch .zeros_like (raw_power_kw ,dtype =torch .int32 )
                    current_state [raw_power_kw >0 ]=1
                    current_state [raw_power_kw <0 ]=-1


                    last_states =self .last_non_zero_state [st ,sorted_active_evs ]
                    switched =(current_state !=0 )&(last_states !=0 )&(current_state !=last_states )

                    self .switch_count [st ,sorted_active_evs ]+=switched .int ()

                    non_zero_mask =current_state !=0
                    if non_zero_mask .any ():
                        updated_last_states =last_states .clone ()
                        updated_last_states [non_zero_mask ]=current_state [non_zero_mask ]
                        self .last_non_zero_state [st ,sorted_active_evs ]=updated_last_states

                    if switched .any ():
                        self .metrics ['total_switches_current']+=int (switched .sum ().item ())

                    st_switch_penalty =switched .float ().sum ()*LOCAL_SWITCH_PENALTY
                    local_rewards_tensor [st ]-=st_switch_penalty
                    switch_penalties_tensor [st ]=st_switch_penalty


                new_socs =prev_socs +proposed_delta_soc
                new_socs =torch .clamp (new_socs ,0.0 ,100.0 )


                actual_delta_soc =new_socs -prev_socs
                actual_delta_kwh =actual_delta_soc *ev_kwh_per_soc_pct
                effective_power_kw =actual_delta_kwh /POWER_TO_ENERGY
                if self .use_station_total_power_limit or self .local_station_limit_penalty_coef !=0.0 :
                    raw_unprojected_power_kw =torch .clamp (
                    actor_actions [st ,:active_count ],-1.0 ,1.0
                    )*ev_max_power_kw
                    raw_unprojected_soc =torch .clamp (
                    prev_socs +raw_unprojected_power_kw *ev_soc_step_per_kw ,0.0 ,100.0
                    )
                    raw_unprojected_power_kw =(
                    (raw_unprojected_soc -prev_socs )*ev_kwh_per_soc_pct /POWER_TO_ENERGY
                    )
                    charge_excess_kw =torch .clamp (
                    raw_unprojected_power_kw [raw_unprojected_power_kw >0 ].sum ()-self .station_power_limit_kw [st ],
                    min =0.0 ,
                    )
                    discharge_excess_kw =torch .clamp (
                    (-raw_unprojected_power_kw [raw_unprojected_power_kw <0 ]).sum ()-self .station_power_limit_kw [st ],
                    min =0.0 ,
                    )
                    station_limit_penalty =self .local_station_limit_penalty_coef *(
                    charge_excess_kw +discharge_excess_kw
                    )
                    local_rewards_tensor [st ]-=station_limit_penalty
                    station_limit_penalties_tensor [st ]=station_limit_penalty
                    effective_power_kw ,charge_limited ,discharge_limited =self ._apply_station_power_limit (
                    st ,effective_power_kw
                    )
                else :
                    limit_not_hit =torch .zeros ((),dtype =torch .bool ,device =device )
                    charge_limited =limit_not_hit
                    discharge_limited =limit_not_hit
                actual_delta_kwh =effective_power_kw *POWER_TO_ENERGY
                actual_delta_soc =actual_delta_kwh *ev_soc_pct_per_kwh
                new_socs =prev_socs +actual_delta_soc
                self .prev_soc [st ,sorted_active_evs ]=prev_socs
                self .soc [st ,sorted_active_evs ]=new_socs

                discharge_energy_kwh =torch .clamp (-raw_delta_kwh ,min =0.0 ).sum ()
                discharge_penalty =self .local_discharge_penalty_coef *discharge_energy_kwh
                local_rewards_tensor [st ]-=discharge_penalty
                discharge_penalties_tensor [st ]=discharge_penalty

                station_charge_limit_hits_tensor [st ]=charge_limited
                station_discharge_limit_hits_tensor [st ]=discharge_limited

                if build_info and self .record_snapshots :
                    for i ,ev_idx in enumerate (sorted_active_evs ):
                        ev_id =int (self .ev_ids [st ,ev_idx ].item ())
                        for ev_dict in self .stations_evs [st ]:
                            if ev_dict ['id']==ev_id :
                                ev_dict ['soc']=float (new_socs [i ].item ())
                                break

                target =self .target [st ,sorted_active_evs ]

                remaining_steps =torch .clamp (
                self ._remaining_action_steps (self .depart [st ,sorted_active_evs ]),
                min =1.0 ,
                )
                if LOCAL_REWARD_MODE =="potential":
                    shaping_per_ev =soc_potential_rewards (prev_socs ,raw_new_socs ,target ,remaining_steps )
                    shaping_sum =shaping_per_ev .sum ()
                    if ev_local_rewards is not None :
                        ev_local_rewards [st ,:active_count ]+=shaping_per_ev
                elif ev_local_rewards is not None :
                    shaping_sum ,shaping_per_ev =soc_progress_shaping (
                    prev_socs ,raw_new_socs ,target ,remaining_steps ,
                    full_power_soc_per_step =ev_max_power_kw *ev_soc_step_per_kw ,
                    return_per_ev =True ,
                    )
                    ev_local_rewards [st ,:active_count ]+=shaping_per_ev
                else :
                    shaping_sum =soc_progress_shaping (
                    prev_socs ,raw_new_socs ,target ,remaining_steps ,
                    full_power_soc_per_step =ev_max_power_kw *ev_soc_step_per_kw ,
                    )
                local_rewards_tensor [st ]+=shaping_sum
                progress_shaping_rewards_tensor [st ]=shaping_sum

                effective_power_kw =actual_delta_kwh /POWER_TO_ENERGY

                station_powers_tensor [st ]=effective_power_kw .sum ()




                critic_power =effective_power_kw .clone ()

                actual_ev_power_kw_tensor [st ,:active_count ]=critic_power

                if self .record_snapshots :
                    remains =self ._remaining_action_steps (self .depart [st ,sorted_active_evs ])
                    needs_before =self .target [st ,sorted_active_evs ]-prev_socs
                    ids =self .ev_ids [st ,sorted_active_evs ]
                    details_after =[]
                    for i in range (active_count ):
                        details_after .append ({
                        'id':int (ids [i ].item ()),
                        'remaining_time':float (remains [i ].item ()),
                        'needed_soc':float (needs_before [i ].item ()),
                        'prev_soc':float (prev_socs [i ].item ()),
                        'action_scaled':float (scaled_power_kw [i ].item ()),
                        'new_soc':float (new_socs [i ].item ()),
                        'delta_soc':float (actual_delta_soc [i ].item ()),
                        'critic_input':float (critic_power [i ].item ()),
                        'target_soc':float (self .target [st ,sorted_active_evs ][i ].item ()),
                        'battery_capacity_kwh':float (ev_capacity_kwh [i ].item ()),
                        'max_power_kw':float (ev_max_power_kw [i ].item ()),
                        'switch_count':int (self .switch_count [st ,sorted_active_evs ][i ].item ()),
                        })
                    snapshot_after [st ]=details_after
                else :
                    snapshot_after [st ]=[]
            else :
                snapshot_after [st ]=[]

        raw_actor_total_power_kw =float (central_allocator_info ['raw_actor_total_power_kw'])
        central_correction_tensor =actual_ev_power_kw_tensor -raw_actor_ev_power_kw_tensor
        central_station_targets_tensor =central_allocator_info ['central_station_target_powers']
        station_safe_min_power_tensor =central_allocator_info ['station_safe_min_power_kw']
        station_safe_max_power_tensor =central_allocator_info ['station_safe_max_power_kw']
        central_station_correction_tensor =station_powers_tensor -raw_actor_station_powers_tensor
        # One transfer for the five scalars this block needs.  Each `.item()`
        # is its own device synchronisation, and this runs on every step of
        # every episode.  Counts go through the float stack and back: they are
        # at most a few hundred, far inside what float32 holds exactly.
        _dtype =station_powers_tensor .dtype
        _station_target_error =(
        torch .abs (station_powers_tensor -central_station_targets_tensor ).max ()
        if central_station_targets_tensor .numel ()>0
        else torch .zeros ((),dtype =_dtype ,device =station_powers_tensor .device )
        )
        _scalars =torch .stack ([
        station_powers_tensor .sum (),
        (torch .abs (central_correction_tensor )>1e-5 ).sum ().to (_dtype ),
        (torch .abs (central_station_correction_tensor )>1e-5 ).sum ().to (_dtype ),
        _station_target_error ,
        torch .abs (central_correction_tensor ).sum (),
        ]).tolist ()
        total_ev_transport =_scalars [0 ]
        corrected_ev_count =int (round (_scalars [1 ]))
        corrected_station_count =int (round (_scalars [2 ]))
        central_station_target_max_abs_error_kw =float (_scalars [3 ])
        absolute_correction_kw =float (_scalars [4 ])
        central_correction_power_kw =total_ev_transport -raw_actor_total_power_kw
        raw_actor_deviation =abs (current_request -raw_actor_total_power_kw )
        central_deviation =abs (current_request -total_ev_transport )

        self .last_station_powers =station_powers_tensor .clone ()
        self .last_raw_actor_total_power_kw =raw_actor_total_power_kw
        self .last_raw_actor_residual_kw =current_request -raw_actor_total_power_kw
        self .last_central_correction_power_kw =central_correction_power_kw
        self .last_central_ev_total_power_kw =total_ev_transport
        bess_result =self ._dispatch_residual_bess (
        current_request ,total_ev_transport ,self .current_tracking_enabled
        )
        self .metrics ['central_corrected_ev_steps']+=corrected_ev_count
        self .metrics ['central_corrected_station_steps']+=corrected_station_count
        self .metrics ['central_correction_active_steps']+=int (corrected_ev_count >0 )
        self .metrics ['central_absolute_correction_kwh']+=absolute_correction_kw *float (POWER_TO_ENERGY )
        self .metrics ['central_max_abs_aggregate_correction_kw']=max (
        self .metrics ['central_max_abs_aggregate_correction_kw'],abs (central_correction_power_kw )
        )
        self .metrics ['central_max_corrected_evs_per_step']=max (
        self .metrics ['central_max_corrected_evs_per_step'],corrected_ev_count
        )
        self .metrics ['central_max_corrected_stations_per_step']=max (
        self .metrics ['central_max_corrected_stations_per_step'],corrected_station_count
        )
        self .metrics ['central_max_station_target_error_kw']=max (
        self .metrics ['central_max_station_target_error_kw'],central_station_target_max_abs_error_kw
        )
        # How much the central layer was asked to move versus how much it could.
        # Both numbers were computed every step and only the first reached the
        # per-step info dict, so a run could not say whether a missed band was
        # the fleet running out of headroom or the allocator leaving some
        # unused.
        _requested_kw =float (central_allocator_info ['requested_correction_kw'])
        _planned_kw =float (central_allocator_info ['planned_correction_kw'])
        _unmet_kw =_requested_kw -_planned_kw
        self .metrics ['central_requested_correction_kwh']+=abs (_requested_kw )*float (POWER_TO_ENERGY )
        self .metrics ['central_planned_correction_kwh']+=abs (_planned_kw )*float (POWER_TO_ENERGY )
        self .metrics ['central_unmet_correction_kwh']+=abs (_unmet_kw )*float (POWER_TO_ENERGY )
        self .metrics ['central_unmet_steps']+=int (abs (_unmet_kw )>1e-5 )


        deviation =raw_actor_deviation
        system_deviation =abs (float (bess_result ['post_residual_kw']))
        if self .use_station_total_power_limit or self .local_station_limit_penalty_coef !=0.0 :
            station_limit_mask =station_charge_limit_hits_tensor |station_discharge_limit_hits_tensor
            self .metrics ['station_charge_limit_hits']+=int (station_charge_limit_hits_tensor .sum ().item ())
            self .metrics ['station_discharge_limit_hits']+=int (station_discharge_limit_hits_tensor .sum ().item ())
            self .metrics ['station_limit_hits']+=int (station_limit_mask .sum ().item ())
            self .metrics ['station_limit_steps']+=int (station_limit_mask .any ().item ())
            self .metrics ['station_limit_penalty_total']+=float (station_limit_penalties_tensor .sum ().item ())
        self .metrics ['total_steps']+=1
        if self .current_tracking_enabled :
            self .metrics ['tracking_steps']+=1
            self .metrics ['raw_actor_abs_error_sum_kw']+=raw_actor_deviation
        else :
            self .metrics ['free_steps']+=1
        regulation =float (getattr (self ,'current_regulation_kw',current_request ))
        down_instruction =regulation >1e-6
        up_instruction =regulation <-1e-6
        if self .current_tracking_enabled and down_instruction :
            self .metrics ['surplus_steps']+=1
            self .metrics ['central_surplus_steps']+=1
            self .metrics ['system_surplus_steps']+=1
        elif self .current_tracking_enabled and up_instruction :
            self .metrics ['shortage_steps']+=1
            self .metrics ['central_shortage_steps']+=1
            self .metrics ['system_shortage_steps']+=1


        if self .current_tracking_enabled and down_instruction :

            if deviation <=self .tol_narrow_metrics :
                self .metrics ['surplus_within_narrow']+=1
            if central_deviation <=self .tol_narrow_metrics :
                self .metrics ['central_surplus_within_narrow']+=1
            if system_deviation <=self .tol_narrow_metrics :
                self .metrics ['system_surplus_within_narrow']+=1
        elif self .current_tracking_enabled and up_instruction :
            if deviation <=self .tol_narrow_metrics :
                self .metrics ['shortage_within_narrow']+=1
            if central_deviation <=self .tol_narrow_metrics :
                self .metrics ['central_shortage_within_narrow']+=1
            if system_deviation <=self .tol_narrow_metrics :
                self .metrics ['system_shortage_within_narrow']+=1
        elif self .current_tracking_enabled :
            if central_deviation <=self .tol_narrow_metrics :
                self .metrics ['central_zero_request_within_narrow']+=1
            if deviation <=self .tol_narrow_metrics :
                self .metrics ['zero_request_within_narrow']+=1
            if system_deviation <=self .tol_narrow_metrics :
                self .metrics ['system_zero_request_within_narrow']+=1
            self .metrics ['zero_request_steps']+=1
            self .metrics ['central_zero_request_steps']+=1
            self .metrics ['system_zero_request_steps']+=1
        else :
            pass



        global_reward =(
        self ._calculate_balance_reward (deviation )
        if self .current_tracking_enabled
        else 0.0
        )
        balance_reward_only =float (global_reward )
        system_global_reward =(
        self ._calculate_balance_reward (system_deviation )
        if self .current_tracking_enabled
        else 0.0
        )
        # Pay the learner for the residual the market judges, when asked to.
        # `balance_reward_only` keeps reporting the pre-correction figure either
        # way, so the two stay comparable in the diagnostics.
        if self .global_reward_on_system_output :
            global_reward =system_global_reward



        all_departing_data ={
        'station_ids':[],
        'slot_indices':[],
        'ev_ids':[],
        'final_socs':[],
        'central_final_socs':[],
        'target_socs':[],
        'capacity_kwh':[],
        }


        departed_stations =set ()
        scheduled_departures =self ._departure_slots_by_step .pop (int (self .step_count ),[])
        if scheduled_departures :
            departures_by_station =[[]for _ in range (self .num_stations )]
            for st ,slot_idx in scheduled_departures :
                if 0 <=st <self .num_stations and 0 <=slot_idx <self .max_ev_per_station :
                    departures_by_station [st ].append (slot_idx )
        else :
            departures_by_station =[]

        for st ,slot_indices in enumerate (departures_by_station ):
            if not slot_indices :
                continue

            candidate_indices =torch .as_tensor (
            sorted (slot_indices ),dtype =torch .long ,device =device
            )
            valid_departures =(
            self .ev_mask [st ,candidate_indices ]
            &(self .depart [st ,candidate_indices ]==self .step_count )
            )
            if valid_departures .any ():
                departing_indices =candidate_indices [valid_departures ]
                departed_stations .add (st )

                all_departing_data ['station_ids'].extend ([st ]*int (departing_indices .numel ()))
                all_departing_data ['slot_indices'].extend (departing_indices .tolist ())
                all_departing_data ['ev_ids'].append (self .ev_ids [st ,departing_indices ])
                all_departing_data ['final_socs'].append (raw_reward_soc_tensor [st ,departing_indices ])
                all_departing_data ['central_final_socs'].append (self .soc [st ,departing_indices ])
                all_departing_data ['target_socs'].append (self .target [st ,departing_indices ])
                all_departing_data ['capacity_kwh'].append (self .ev_capacity_kwh [st ,departing_indices ])



                self .metrics ['total_switches_departed']+=int (self .switch_count [st ,departing_indices ].sum ().item ())
                departing_count =int (departing_indices .numel ())
                self .active_evs_total =max (0 ,self .active_evs_total -departing_count )
                if USE_HETEROGENEOUS_EV_PHYSICS :
                    departed_power_kw =float (self .ev_max_power_kw [st ,departing_indices ].sum ().item ())
                else :
                    departed_power_kw =float (departing_count )*float (MAX_EV_POWER_KW )
                self .active_ev_power_limit_kw =max (
                0.0 ,self .active_ev_power_limit_kw -departed_power_kw
                )


                self .ev_mask [st ,departing_indices ]=False
                self .ev_ids [st ,departing_indices ]=0
                self .soc [st ,departing_indices ]=0.0
                self .prev_soc [st ,departing_indices ]=0.0
                self .target [st ,departing_indices ]=0.0
                self .ev_capacity_kwh [st ,departing_indices ]=0.0
                self .ev_max_power_kw [st ,departing_indices ]=0.0
                self .ev_soc_pct_per_kwh [st ,departing_indices ]=0.0
                self .ev_soc_step_per_kw [st ,departing_indices ]=0.0
                self .ev_kwh_per_soc_pct [st ,departing_indices ]=0.0
                self .depart [st ,departing_indices ]=0
                self .initial_remaining [st ,departing_indices ]=0.0
                self .arrival_step [st ,departing_indices ]=0.0
                self .switch_count [st ,departing_indices ]=0
                self .last_non_zero_state [st ,departing_indices ]=0

        for st in departed_stations :
            self ._invalidate_active_order_cache (st )


        departure_reward_sum_tensor =torch .zeros (self .num_stations ,dtype =torch .float32 ,device =device )
        if all_departing_data ['ev_ids']:

            departing_ev_ids =torch .cat (all_departing_data ['ev_ids'])
            departing_final_socs =torch .cat (all_departing_data ['final_socs'])
            central_departing_final_socs =torch .cat (all_departing_data ['central_final_socs'])
            departing_target_socs =torch .cat (all_departing_data ['target_socs'])
            departing_capacity_kwh =torch .cat (all_departing_data ['capacity_kwh'])


            soc_achieved_metric =departing_final_socs >=departing_target_socs
            soc_diff =departing_target_socs -departing_final_socs


            soc_deficit =torch .clamp (soc_diff ,min =0.0 )
            soc_deficit_kwh =soc_deficit *departing_capacity_kwh /100.0
            self .metrics ['total_soc_deficit']+=float (soc_deficit_kwh .sum ().item ())
            self .metrics ['total_soc_unmet']+=int ((soc_deficit >0 ).sum ().item ())

            central_soc_diff =departing_target_socs -central_departing_final_socs
            central_soc_deficit =torch .clamp (central_soc_diff ,min =0.0 )
            central_soc_deficit_kwh =central_soc_deficit *departing_capacity_kwh /100.0
            self .metrics ['central_departing_evs_soc_met']+=int ((central_soc_diff <=0 ).sum ().item ())
            self .metrics ['central_total_soc_deficit']+=float (central_soc_deficit_kwh .sum ().item ())
            self .metrics ['central_total_soc_unmet']+=int ((central_soc_deficit >0 ).sum ().item ())

            if LOCAL_REWARD_MODE =="potential":
                # The shortfall is already charged by the last step's potential
                # term; only leaving below target at all is charged here.
                per_ev_departure_rewards =-float (LOCAL_POTENTIAL_MISS_PENALTY )*(
                departing_final_socs <departing_target_socs
                ).float ()
            else :
                per_ev_departure_rewards =departure_rewards (
                departing_final_socs ,departing_target_socs
                )


            station_ids_cpu =all_departing_data ['station_ids']

            departing_ev_stats [0 ]=int (departing_ev_ids .numel ())
            departing_ev_stats [1 ]=int (soc_achieved_metric .sum ().item ())

            departure_reward_sum_tensor =torch .zeros (self .num_stations ,dtype =torch .float32 ,device =device )
            station_ids_tensor =torch .as_tensor (station_ids_cpu ,dtype =torch .long ,device =device )
            departure_reward_sum_tensor .index_add_ (0 ,station_ids_tensor ,per_ev_departure_rewards )
            local_rewards_tensor +=departure_reward_sum_tensor
            if ev_local_rewards is not None :
                departing_slots =physical_to_slot [
                station_ids_tensor ,
                torch .as_tensor (all_departing_data ['slot_indices'],dtype =torch .long ,device =device ),
                ]
                ev_local_rewards .index_put_ (
                (station_ids_tensor ,departing_slots ),per_ev_departure_rewards ,accumulate =True
                )

            ev_ids_cpu =departing_ev_ids .detach ().cpu ().tolist ()
            for ev_id in ev_ids_cpu :
                self ._release_ev_id (int (ev_id ))

            if build_info :
                for st ,ev_id in zip (station_ids_cpu ,ev_ids_cpu ):
                    ev_dict_idx =next ((j for j ,ev in enumerate (self .stations_evs [st ])if ev ['id']==int (ev_id )),-1 )
                    if ev_dict_idx >=0 :
                        self .stations_evs [st ].pop (ev_dict_idx )


        self .metrics ['departing_evs']+=departing_ev_stats [0 ]
        self .metrics ['departing_evs_soc_met']+=departing_ev_stats [1 ]


        snapshot_end ={}
        if build_info and self .record_snapshots :
            for st in range (self .num_stations ):
                details_end =[]
                sorted_evs =self ._get_sorted_active_evs (st )
                if sorted_evs .numel ()>0 :
                    ids =self .ev_ids [st ,sorted_evs ]
                    socs =self .soc [st ,sorted_evs ]

                    remains =self ._remaining_action_steps (self .depart [st ,sorted_evs ])

                    needs =self .target [st ,sorted_evs ]-socs
                    for i in range (int (sorted_evs .numel ())):
                        details_end .append ({
                        'id':int (ids [i ].item ()),
                        'soc':float (socs [i ].item ()),
                        'needed_soc':float (needs [i ].item ()),
                        'remaining_time':float (remains [i ].item ()),
                        'target_soc':float (self .target [st ,sorted_evs ][i ].item ()),
                        'battery_capacity_kwh':float (self .ev_capacity_kwh [st ,sorted_evs ][i ].item ()),
                        'max_power_kw':float (self .ev_max_power_kw [st ,sorted_evs ][i ].item ()),
                        'switch_count':int (self .switch_count [st ,sorted_evs ][i ].item ()),
                        })
                snapshot_end [st ]=details_end



        if ev_local_rewards is not None :
            # Station-level terms (penalties) go equally to the station's EVs,
            # so the per-EV rewards always sum to the station's local reward.
            present =physical_to_slot .max (dim =1 ).values +1
            slot_present =torch .arange (self .max_ev_per_station ,device =device ).unsqueeze (0 )<present .unsqueeze (1 )
            residual =local_rewards_tensor -ev_local_rewards .sum (dim =1 )
            ev_local_rewards +=slot_present .float ()*(residual /torch .clamp (present ,min =1 ).float ()).unsqueeze (1 )
        self .last_ev_local_rewards =ev_local_rewards

        done =[self .step_count >=self .episode_steps ]*self .num_stations

        observation =self ._get_obs ()if return_observation else None

        if not build_info :
            info ={
            'net_demand':float (current_request ),
            'raw_net_demand':float (current_request ),
            'observed_net_demand':float (self .current_observed_net_demand ),
            'demand_clip_kw':float (self .current_demand_clip_kw ),
            'tracking_enabled':bool (self .current_tracking_enabled ),
            'station_powers':station_powers_tensor ,
            'total_ev_transport':station_powers_tensor .sum (),
            'raw_actor_station_powers':raw_actor_station_powers_tensor ,
            'raw_actor_total_power_kw':float (raw_actor_total_power_kw ),
            'raw_actor_residual_kw':float (current_request -raw_actor_total_power_kw ),
            'raw_actor_ev_power_kw':raw_actor_ev_power_kw_tensor ,
            'station_safe_min_power_kw':station_safe_min_power_tensor ,
            'station_safe_max_power_kw':station_safe_max_power_tensor ,
            'central_station_target_powers':central_station_targets_tensor ,
            'central_correction_power_kw':float (central_correction_power_kw ),
            'central_absolute_correction_kw':float (absolute_correction_kw ),
            'central_corrected_ev_count':int (corrected_ev_count ),
            'central_corrected_station_count':int (corrected_station_count ),
            'central_station_target_max_abs_error_kw':float (central_station_target_max_abs_error_kw ),
            'central_allocator_architecture':str (central_allocator_info ['architecture']),
            'pre_bess_residual_kw':float (bess_result ['pre_residual_kw']),
            'bess_requested_power_kw':float (bess_result ['requested_power_kw']),
            'bess_power_kw':float (bess_result ['power_kw']),
            'bess_soc_pct':float (bess_result ['soc_pct']),
            'pcc_power_kw':float (bess_result ['pcc_power_kw']),
            'post_bess_residual_kw':float (bess_result ['post_residual_kw']),
            'system_global_reward':float (system_global_reward ),
            'actual_ev_power_kw':actual_ev_power_kw_tensor ,
            'executed_ev_power_kw':actual_ev_power_kw_tensor ,
            'step_count':int (self .step_count ),
            }
            return observation ,local_rewards_tensor ,float (global_reward ),done ,info

        _station_powers_list =station_powers_tensor .cpu ().tolist ()

        info ={
        'net_demand':current_request ,
        'raw_net_demand':float (current_request ),
        'observed_net_demand':float (self .current_observed_net_demand ),
        'demand_clip_kw':float (self .current_demand_clip_kw ),
        'demand_was_clipped':bool (abs (self .current_demand_clip_kw )>1e-6 ),
        'tracking_enabled':bool (self .current_tracking_enabled ),
        'station_powers':_station_powers_list ,
        'total_ev_transport':total_ev_transport ,
        'raw_actor_station_powers':raw_actor_station_powers_tensor .cpu ().tolist (),
        'raw_actor_total_power_kw':float (raw_actor_total_power_kw ),
        'raw_actor_residual_kw':float (current_request -raw_actor_total_power_kw ),
        'raw_actor_ev_power_kw':raw_actor_ev_power_kw_tensor ,
        'station_safe_min_power_kw':station_safe_min_power_tensor .cpu ().tolist (),
        'station_safe_max_power_kw':station_safe_max_power_tensor .cpu ().tolist (),
        'central_station_target_powers':central_station_targets_tensor .cpu ().tolist (),
        'central_correction_power_kw':float (central_correction_power_kw ),
        'central_absolute_correction_kw':float (absolute_correction_kw ),
        'central_corrected_ev_count':int (corrected_ev_count ),
        'central_corrected_station_count':int (corrected_station_count ),
        'central_station_target_max_abs_error_kw':float (central_station_target_max_abs_error_kw ),
        'central_allocator_architecture':str (central_allocator_info ['architecture']),
        'central_requested_correction_kw':float (central_allocator_info ['requested_correction_kw']),
        'central_safety_floor_correction_kw':float (central_allocator_info ['safety_floor_correction_kw']),
        'pre_bess_residual_kw':float (bess_result ['pre_residual_kw']),
        'bess_requested_power_kw':float (bess_result ['requested_power_kw']),
        'bess_power_kw':float (bess_result ['power_kw']),
        'bess_energy_before_kwh':float (bess_result ['energy_before_kwh']),
        'bess_energy_after_kwh':float (bess_result ['energy_after_kwh']),
        'bess_soc_pct':float (bess_result ['soc_pct']),
        'bess_power_limit_hit':bool (bess_result ['power_limit_hit']),
        'bess_energy_limit_hit':bool (bess_result ['energy_limit_hit']),
        'pcc_power_kw':float (bess_result ['pcc_power_kw']),
        'post_bess_residual_kw':float (bess_result ['post_residual_kw']),
        'system_global_reward':float (system_global_reward ),
        'actual_ev_power_kw':actual_ev_power_kw_tensor ,
        'executed_ev_power_kw':actual_ev_power_kw_tensor ,
        'snapshot_after':snapshot_after ,
        'arrivals_by_station':[int (x )for x in self .arrivals_by_station ],
        'step_count':int (self .step_count ),
        'global_reward':float (global_reward ),
        }

        if include_reward_breakdown :
            signed_deviation =float (raw_actor_total_power_kw -current_request )
            _local_rewards_list =local_rewards_tensor .cpu ().tolist ()
            _progress_shaping_list =progress_shaping_rewards_tensor .cpu ().tolist ()
            _discharge_penalties_list =discharge_penalties_tensor .cpu ().tolist ()
            _switch_penalties_list =switch_penalties_tensor .cpu ().tolist ()
            _departure_reward_list =departure_reward_sum_tensor .cpu ().tolist ()
            _station_limit_penalties_list =station_limit_penalties_tensor .cpu ().tolist ()
            info ['reward_breakdown']={
            'global':{
            'balance_reward':float (balance_reward_only ),
            'global_total':float (global_reward ),
            'deviation':float (signed_deviation ),
            'abs_deviation':float (deviation ),
            'net_demand':float (current_request ),
            'raw_actor_total_power_kw':float (raw_actor_total_power_kw ),
            'central_ev_total_power_kw':float (total_ev_transport ),
            'central_abs_deviation':float (central_deviation ),
            'system_balance_reward':float (system_global_reward ),
            'pcc_power_kw':float (bess_result ['pcc_power_kw']),
            'post_bess_deviation':float (bess_result ['post_residual_kw']),
            'post_bess_abs_deviation':float (system_deviation ),
            'bess_power_kw':float (bess_result ['power_kw']),
            'bess_soc_pct':float (bess_result ['soc_pct']),
            },
            'per_station':[
            {
            'station_power':_station_powers_list [st ],
            'raw_actor_station_power':float (raw_actor_station_powers_tensor [st ].item ()),
            'progress_shaping':_progress_shaping_list [st ],
            'discharge_penalty':_discharge_penalties_list [st ],
            'switch_penalty':_switch_penalties_list [st ],
            'station_limit_penalty':_station_limit_penalties_list [st ],
            'departure_reward':_departure_reward_list [st ],
            'local_total':_local_rewards_list [st ],
            }
            for st in range (self .num_stations )
            ]
            }

        if include_ev_physics :
            info ['ev_capacity_kwh']=self .ev_capacity_kwh .clone ()
            info ['ev_max_power_kw']=self .ev_max_power_kw .clone ()
        if include_active_evs :
            info ['active_evs']={i :int (self .ev_mask [i ].sum ().item ())for i in range (self .num_stations )}

        if all_departing_data ['ev_ids']:
            station_ids_cpu =all_departing_data ['station_ids']
            slot_indices_cpu =all_departing_data ['slot_indices']
            info ['departed_evs']=[
            {'id':int (eid ),'station':int (st ),'slot':int (si )}
            for st ,si ,eid in zip (station_ids_cpu ,slot_indices_cpu ,ev_ids_cpu )
            ]
        else :
            info ['departed_evs']=[]

        local_rewards_out =_local_rewards_list if include_reward_breakdown else local_rewards_tensor
        return observation ,local_rewards_out ,global_reward ,done ,info





    def _spawn_ev_from_event (self ,station :int ,ev_event :dict ,empty_slots :torch .Tensor )->bool :
        slot_idx =int (empty_slots [random .randint (0 ,int (empty_slots .numel ())-1 )])
        return self ._spawn_ev_from_event_at_slot (station ,ev_event ,slot_idx )

    def _spawn_ev_from_event_at_slot (self ,station :int ,ev_event :dict ,slot_idx :int )->bool :
        init_soc =float (ev_event ['init_soc'])
        target_soc =float (ev_event ['target_soc'])
        dwell_steps =int (ev_event ['dwell_steps'])
        needed_soc =float (ev_event ['needed_soc'])
        profile_ev_id =ev_event .get ('profile_ev_id')
        capacity_kwh =float (ev_event .get ('capacity_kwh',EV_CAPACITY ))
        max_power_kw =float (ev_event .get ('max_power_kw',MAX_EV_POWER_KW ))

        if profile_ev_id is not None and self ._reserve_ev_id (int (profile_ev_id )):
            ev_id =int (profile_ev_id )
        else :
            ev_id =self ._allocate_ev_id ()
            if ev_id is None :
                return False

        dep ,target_soc =self ._horizon_obligation (
        target_soc ,self .step_count +dwell_steps -1 ,capacity_kwh ,max_power_kw ,
        self .step_count ,init_soc ,
        )
        needed_soc =max (target_soc -init_soc ,0.0 )
        profile_remaining =max (dep -self .step_count ,0 )

        self ._record_arrival (init_soc ,needed_soc ,dwell_steps )
        self ._set_ev_slot (
        station ,slot_idx ,ev_id ,init_soc ,target_soc ,dep ,
        capacity_kwh =capacity_kwh ,
        max_power_kw =max_power_kw ,
        profile_remaining =profile_remaining ,
        )
        return True

    def _calculate_balance_reward (self ,deviation :float )->float :
        balance_reward_value =getattr (self ,'balance_reward',GLOBAL_BALANCE_REWARD )
        d =max (float (deviation ),0.0 )
        mode =getattr (self ,'balance_reward_mode',GLOBAL_BALANCE_REWARD_MODE )
        if mode =="bounded_absolute_error":
            scale =max (
            float (getattr (
            self ,'_balance_reward_error_scale_kw',
            GLOBAL_BALANCE_REWARD_ERROR_SCALE_KW ,
            )),1e-6 ,
            )
            # The formal pass/fail tolerance remains an assessment metric. The
            # learner instead sees a smooth, bounded incentive to reduce every
            # residual kW, including errors inside that tolerance band.
            # Past ``d0`` the curve stops bending and keeps the slope it had
            # there, so a deep miss is paid for in proportion to its depth.
            # Below d0 nothing changes, so the origin slope and the zero
            # crossing are exactly where they were.
            d0 =float (getattr (
            self ,'_balance_reward_linear_tail_kw',
            GLOBAL_BALANCE_REWARD_LINEAR_TAIL_KW ,
            ))
            if d0 >0.0 and d >d0 :
                t0 =float (np .tanh (d0 /scale ))
                return float (balance_reward_value )*(
                (1.0 -2.0 *t0 )-(2.0 /scale )*(1.0 -t0 *t0 )*(d -d0 )
                )
            return float (balance_reward_value )*(
            1.0 -2.0 *float (np .tanh (d /scale ))
            )

        D =max (float (self .tol_narrow_metrics ),1e-6 )
        if d <=D :
            return float (balance_reward_value )
        slope =self ._balance_reward_slope
        return float (balance_reward_value )-slope *(d -D )


    def get_metrics (self ):
        metrics ={}
        forced =getattr (self ,'_train_forced_kwh',None )
        metrics ['train_forced_kwh']=float (forced .item ())if forced is not None else 0.0


        if self .metrics ['departing_evs']>0 :
            metrics ['soc_miss_rate']=100.0 -(self .metrics ['departing_evs_soc_met']/self .metrics ['departing_evs']*100.0 )

            metrics ['avg_switches']=self .metrics ['total_switches_departed']/self .metrics ['departing_evs']

            unmet =int (self .metrics .get ('total_soc_unmet',0 ))
            if unmet >0 :
                metrics ['avg_soc_deficit']=self .metrics ['total_soc_deficit']/unmet
            else :
                metrics ['avg_soc_deficit']=0.0
        else :
            metrics ['soc_miss_rate']=0.0
            metrics ['avg_switches']=0.0
            metrics ['avg_soc_deficit']=0.0

        if self .metrics ['departing_evs']>0 :
            metrics ['central_soc_miss_rate']=100.0 -(
            self .metrics ['central_departing_evs_soc_met']
            /self .metrics ['departing_evs']*100.0
            )
            central_unmet =int (self .metrics .get ('central_total_soc_unmet',0 ))
            metrics ['central_avg_soc_deficit']=(
            self .metrics ['central_total_soc_deficit']/central_unmet
            if central_unmet >0 else 0.0
            )
        else :
            metrics ['central_soc_miss_rate']=0.0
            metrics ['central_avg_soc_deficit']=0.0


        if self .metrics ['surplus_steps']>0 :
            metrics ['surplus_absorption_rate']=self .metrics ['surplus_within_narrow']/self .metrics ['surplus_steps']*100.0
        else :
            metrics ['surplus_absorption_rate']=0.0


        if self .metrics ['shortage_steps']>0 :
            metrics ['supply_cooperation_rate']=self .metrics ['shortage_within_narrow']/self .metrics ['shortage_steps']*100.0
        else :
            metrics ['supply_cooperation_rate']=0.0


        if self .metrics ['zero_request_steps']>0 :
            metrics ['zero_request_maintenance_rate']=self .metrics ['zero_request_within_narrow']/self .metrics ['zero_request_steps']*100.0
        else :
            metrics ['zero_request_maintenance_rate']=0.0

        tracking_within =(
        self .metrics ['surplus_within_narrow']
        +self .metrics ['shortage_within_narrow']
        +self .metrics ['zero_request_within_narrow']
        )
        tracking_steps =self .metrics ['tracking_steps']
        metrics ['tracking_within_narrow']=tracking_within
        metrics ['tracking_success_rate']=(
        tracking_within /tracking_steps *100.0 if tracking_steps >0 else 0.0
        )
        metrics ['raw_actor_mae_kw']=(
        self .metrics ['raw_actor_abs_error_sum_kw']/tracking_steps
        if tracking_steps >0 else 0.0
        )

        central_tracking_within =(
        self .metrics ['central_surplus_within_narrow']
        +self .metrics ['central_shortage_within_narrow']
        +self .metrics ['central_zero_request_within_narrow']
        )
        metrics ['central_tracking_within_narrow']=central_tracking_within
        metrics ['central_tracking_success_rate']=(
        central_tracking_within /tracking_steps *100.0 if tracking_steps >0 else 0.0
        )

        system_tracking_within =(
        self .metrics ['system_surplus_within_narrow']
        +self .metrics ['system_shortage_within_narrow']
        +self .metrics ['system_zero_request_within_narrow']
        )
        metrics ['system_tracking_within_narrow']=system_tracking_within
        metrics ['system_tracking_success_rate']=(
        system_tracking_within /tracking_steps *100.0 if tracking_steps >0 else 0.0
        )
        metrics ['pre_bess_mae_kw']=(
        self .metrics ['pre_bess_abs_error_sum_kw']/tracking_steps
        if tracking_steps >0 else 0.0
        )
        metrics ['post_bess_mae_kw']=(
        self .metrics ['post_bess_abs_error_sum_kw']/tracking_steps
        if tracking_steps >0 else 0.0
        )


        metrics ['total_switches']=self .metrics ['total_switches_current']
        metrics ['surplus_steps']=self .metrics ['surplus_steps']
        metrics ['surplus_within_narrow']=self .metrics ['surplus_within_narrow']
        metrics ['shortage_steps']=self .metrics ['shortage_steps']
        metrics ['shortage_within_narrow']=self .metrics ['shortage_within_narrow']
        metrics ['zero_request_steps']=self .metrics ['zero_request_steps']
        metrics ['zero_request_within_narrow']=self .metrics ['zero_request_within_narrow']
        metrics ['central_surplus_steps']=self .metrics ['central_surplus_steps']
        metrics ['central_surplus_within_narrow']=self .metrics ['central_surplus_within_narrow']
        metrics ['central_shortage_steps']=self .metrics ['central_shortage_steps']
        metrics ['central_shortage_within_narrow']=self .metrics ['central_shortage_within_narrow']
        metrics ['central_zero_request_steps']=self .metrics ['central_zero_request_steps']
        metrics ['central_zero_request_within_narrow']=self .metrics ['central_zero_request_within_narrow']
        metrics ['central_allocator_enabled']=bool (self .use_central_ev_residual_allocator )
        metrics ['central_corrected_ev_steps']=self .metrics ['central_corrected_ev_steps']
        metrics ['central_corrected_station_steps']=self .metrics ['central_corrected_station_steps']
        metrics ['central_correction_active_steps']=self .metrics ['central_correction_active_steps']
        metrics ['central_absolute_correction_kwh']=self .metrics ['central_absolute_correction_kwh']
        metrics ['central_max_abs_aggregate_correction_kw']=self .metrics ['central_max_abs_aggregate_correction_kw']
        metrics ['central_max_corrected_evs_per_step']=self .metrics ['central_max_corrected_evs_per_step']
        metrics ['central_max_corrected_stations_per_step']=self .metrics ['central_max_corrected_stations_per_step']
        metrics ['central_max_station_target_error_kw']=self .metrics ['central_max_station_target_error_kw']
        metrics ['central_requested_correction_kwh']=self .metrics ['central_requested_correction_kwh']
        metrics ['central_planned_correction_kwh']=self .metrics ['central_planned_correction_kwh']
        metrics ['central_unmet_correction_kwh']=self .metrics ['central_unmet_correction_kwh']
        metrics ['central_unmet_steps']=self .metrics ['central_unmet_steps']
        metrics ['system_surplus_steps']=self .metrics ['system_surplus_steps']
        metrics ['system_surplus_within_narrow']=self .metrics ['system_surplus_within_narrow']
        metrics ['system_shortage_steps']=self .metrics ['system_shortage_steps']
        metrics ['system_shortage_within_narrow']=self .metrics ['system_shortage_within_narrow']
        metrics ['system_zero_request_steps']=self .metrics ['system_zero_request_steps']
        metrics ['system_zero_request_within_narrow']=self .metrics ['system_zero_request_within_narrow']
        metrics ['bess_enabled']=bool (self .use_residual_bess )
        metrics ['bess_power_kw_rating']=float (self .bess_power_limit_kw )
        metrics ['bess_energy_kwh_rating']=float (self .bess_energy_capacity_kwh )
        metrics ['bess_initial_soc_pct']=float (self .bess_initial_soc_pct )
        metrics ['bess_final_soc_pct']=self ._bess_soc_pct ()
        metrics ['bess_charge_energy_kwh']=self .metrics ['bess_charge_energy_kwh']
        metrics ['bess_discharge_energy_kwh']=self .metrics ['bess_discharge_energy_kwh']
        metrics ['bess_throughput_kwh']=self .metrics ['bess_throughput_kwh']
        metrics ['bess_power_limit_hits']=self .metrics ['bess_power_limit_hits']
        metrics ['bess_energy_limit_hits']=self .metrics ['bess_energy_limit_hits']
        metrics ['bess_max_abs_power_kw']=self .metrics ['bess_max_abs_power_kw']
        metrics ['total_steps']=self .metrics ['total_steps']
        metrics ['tracking_steps']=self .metrics ['tracking_steps']
        metrics ['free_steps']=self .metrics ['free_steps']
        metrics ['station_limit_hits']=self .metrics ['station_limit_hits']
        metrics ['station_limit_steps']=self .metrics ['station_limit_steps']
        metrics ['station_charge_limit_hits']=self .metrics ['station_charge_limit_hits']
        metrics ['station_discharge_limit_hits']=self .metrics ['station_discharge_limit_hits']
        metrics ['station_limit_penalty_total']=self .metrics ['station_limit_penalty_total']
        metrics ['departing_evs']=self .metrics ['departing_evs']
        metrics ['departing_evs_soc_met']=self .metrics ['departing_evs_soc_met']
        metrics ['central_departing_evs_soc_met']=self .metrics ['central_departing_evs_soc_met']
        metrics ['initial_evs_total']=int (sum (getattr (self ,"initial_evs_by_station",[])))
        metrics ['initial_evs_by_station']=",".join (
        str (int (x ))for x in getattr (self ,"initial_evs_by_station",[])
        )

        return metrics

    def reset_metrics (self ):
        """Initialize per-episode counters used by training and evaluation."""
        # Energy the departure force floor added this episode, kept on the
        # device so applying the floor does not synchronize every step.
        self ._train_forced_kwh =None
        self .metrics ={
        'total_steps':0 ,
        'tracking_steps':0 ,
        'free_steps':0 ,
        'surplus_within_narrow':0 ,
        'shortage_within_narrow':0 ,
        'surplus_steps':0 ,
        'shortage_steps':0 ,
        'zero_request_within_narrow':0 ,
        'zero_request_steps':0 ,
        'raw_actor_abs_error_sum_kw':0.0 ,
        'central_surplus_within_narrow':0 ,
        'central_shortage_within_narrow':0 ,
        'central_surplus_steps':0 ,
        'central_shortage_steps':0 ,
        'central_zero_request_within_narrow':0 ,
        'central_zero_request_steps':0 ,
        'central_corrected_ev_steps':0 ,
        'central_corrected_station_steps':0 ,
        'central_correction_active_steps':0 ,
        'central_absolute_correction_kwh':0.0 ,
        'central_max_abs_aggregate_correction_kw':0.0 ,
        'central_max_corrected_evs_per_step':0 ,
        'central_max_corrected_stations_per_step':0 ,
        'central_max_station_target_error_kw':0.0 ,
        'central_requested_correction_kwh':0.0 ,
        'central_planned_correction_kwh':0.0 ,
        'central_unmet_correction_kwh':0.0 ,
        'central_unmet_steps':0 ,
        'system_surplus_within_narrow':0 ,
        'system_shortage_within_narrow':0 ,
        'system_surplus_steps':0 ,
        'system_shortage_steps':0 ,
        'system_zero_request_within_narrow':0 ,
        'system_zero_request_steps':0 ,
        'pre_bess_abs_error_sum_kw':0.0 ,
        'post_bess_abs_error_sum_kw':0.0 ,
        'bess_charge_energy_kwh':0.0 ,
        'bess_discharge_energy_kwh':0.0 ,
        'bess_throughput_kwh':0.0 ,
        'bess_power_limit_hits':0 ,
        'bess_energy_limit_hits':0 ,
        'bess_max_abs_power_kw':0.0 ,
        'station_limit_hits':0 ,
        'station_limit_steps':0 ,
        'station_charge_limit_hits':0 ,
        'station_discharge_limit_hits':0 ,
        'station_limit_penalty_total':0.0 ,
        'departing_evs':0 ,
        'departing_evs_soc_met':0 ,
        'central_departing_evs_soc_met':0 ,
        'total_switches_departed':0 ,
        'total_switches_current':0 ,
        'total_soc_deficit':0.0 ,
        'total_soc_unmet':0 ,
        'central_total_soc_deficit':0.0 ,
        'central_total_soc_unmet':0 ,
        }


    def _invalidate_active_order_cache (self ,station :int |None =None ):
        if station is None :
            self ._active_order_cache =[None ]*self .num_stations
            return
        if 0 <=int (station )<self .num_stations :
            self ._active_order_cache [int (station )]=None

    def apply_force_floor (self ,actions :torch .Tensor ):
        """Raise each present EV's action to the departure force-charging floor.

        Returns the raised actions and, per station, the SoC points the floor
        added this step on top of what the actor asked for.
        """
        actions =actions .clone ()
        forced_points =torch .zeros (self .num_stations ,dtype =torch .float32 ,device =actions .device )
        for st in range (self .num_stations ):
            idx =self ._get_sorted_active_evs (st )
            k =int (idx .numel ())
            if k ==0 :
                continue
            floor =force_floor_fraction (
            self .target [st ,idx ]-self .soc [st ,idx ],
            self ._remaining_action_steps (self .depart [st ,idx ]),
            self .ev_capacity_kwh [st ,idx ],
            self .ev_max_power_kw [st ,idx ],
            float (POWER_TO_ENERGY ),
            slack_kwh =float (TRAIN_FORCE_SLACK_KWH ),
            ).to (actions .dtype )
            proposed =torch .clamp (actions [st ,:k ],-1.0 ,1.0 )
            raised =torch .maximum (proposed ,floor )
            added =raised -proposed
            forced_points [st ]=(added *self .ev_max_power_kw [st ,idx ]*self .ev_soc_step_per_kw [st ,idx ]).sum ()
            actions [st ,:k ]=torch .where (added >0 ,raised ,actions [st ,:k ])
            step_kwh =(added *self .ev_max_power_kw [st ,idx ]).sum ()*float (POWER_TO_ENERGY )
            acc =getattr (self ,'_train_forced_kwh',None )
            self ._train_forced_kwh =step_kwh if acc is None else acc +step_kwh
        return actions ,forced_points

    def slot_ev_ids (self )->torch .Tensor :
        """EV id in each observation slot, [stations, max_ev_per_station], -1 where empty.

        Ids are unique within an episode, so the same EV can be found in the
        next observation even after the slot order or physical slots changed.
        """
        ids =torch .full ((self .num_stations ,self .max_ev_per_station ),-1 ,dtype =torch .long ,device =device )
        for st in range (self .num_stations ):
            order =self ._get_sorted_active_evs (st )
            if order .numel ()>0 :
                ids [st ,:order .numel ()]=self .ev_ids [st ,order ].long ()
        return ids

    def _get_sorted_active_evs (self ,station :int )->torch .Tensor :
        cached =self ._active_order_cache [station ]
        if cached is not None :
            return cached

        active_evs =torch .nonzero (self .ev_mask [station ],as_tuple =False ).squeeze (-1 )
        sorted_active_evs =self ._sort_active_evs (station ,active_evs )
        self ._active_order_cache [station ]=sorted_active_evs
        return sorted_active_evs

    def _sort_active_evs (self ,station :int ,active_evs :torch .Tensor )->torch .Tensor :
        """Sort active EV slots by departure time so urgent EVs come first."""
        if active_evs .numel ()<=1 :
            return active_evs


        depart_times =self .depart [station ,active_evs ]
        sorted_indices =torch .argsort (depart_times )
        return active_evs [sorted_indices ]
