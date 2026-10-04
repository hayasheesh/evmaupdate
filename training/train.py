"""
Training entry point for EV multi-agent charging control.

This module coordinates the full training experiment:
- create an archive directory and snapshot the source code;
- take the bank days and their episode preparers from the caller;
- construct EVEnv and the selected agent class;
- fill replay memory during warmup;
- run online environment interaction and gradient updates;
- record TensorBoard diagnostics, CSV/PNG summaries, and checkpoint evaluations.

Training data flow:
1. A bank day is chosen and its episode preparer resets EVEnv with the day's
   submitted bid and one command.
2. EVEnv generates station-specific arrivals and EV profiles for the episode.
3. The agent observes normalized station states and outputs per-station EV-slot
   actions.
4. EVEnv applies actions, returns local/global rewards, and exposes physical
   power traces.
5. Agents with replay buffers cache the transition and update their networks.
6. Periodic deterministic tests run through `tools.evaluator.test()` on the held
   out test bank.

Output structure:
- `archive/{model_name}_{timestamp}/code_snapshot`: source snapshot.
- `.../performance`: TensorBoard event files for training diagnostics.
- `.../results`: train/test CSVs, plots, and checkpoint folders.
"""
import os
import numpy as np
import torch
import random
from datetime import datetime
import traceback
import time
import warnings
import logging
import itertools


warnings .filterwarnings ("ignore",category =UserWarning ,module ="matplotlib")
warnings .filterwarnings ("ignore",message ="Glyph .* missing from font")
logging .getLogger ('matplotlib.font_manager').setLevel (logging .ERROR )
logging .getLogger ('matplotlib.ticker').setLevel (logging .ERROR )

from torch .utils .tensorboard import SummaryWriter
from environment.normalize import (
configure_observation_normalization ,
normalize_observation ,
OBSERVATION_NORMALIZATION_FILENAME ,
save_observation_normalization_profile ,
)
from tools.Utils import (
create_tensorboard_writer ,
GradientLossVisualizer ,
snapshot_code_to_archive ,
InterruptHandler ,
write_train_episode_tb_scalars ,
)
import matplotlib
matplotlib .use ('Agg')
matplotlib .rcParams ['font.family']='sans-serif'
matplotlib .rcParams ['font.sans-serif']=['Arial','Helvetica','Liberation Sans','FreeSans','sans-serif']
from tools.Utils import plot_daily_rewards ,plot_performance_metrics
from tools.evaluator import set_env_seed ,test
from Config import (
NUM_EPISODES ,EPISODE_STEPS ,NUM_EVS ,NUM_STATIONS ,
LR_ACTOR ,LR_CRITIC_LOCAL ,LR_GLOBAL_CRITIC ,
ENV_SEED ,GAMMA ,TAU ,TAU_GLOBAL ,BATCH_SIZE ,SMOOTHL1_BETA ,
TD3_SIGMA_GLOBAL ,TD3_CLIP_GLOBAL ,
MEMORY_SIZE ,WARMUP_STEPS ,
TRAIN_UPDATES_PER_ENV_STEP ,
TRAIN_INTERIM_CSV_INTERVAL_EPISODES ,TRAIN_INTERIM_GRAPH_INTERVAL_EPISODES ,
INTERIM_TEST_EPISODES ,
INTERIM_TEST_SEED ,
)
from environment.EVEnv import EVEnv
from training.Agent.standard_maddpg import build_marl_agent
from training.Agent .maddpg import device
from training.training_resume import (
clear_stop_request ,
load_training_resume ,
read_resume_manifest ,
restore_rng_state ,
save_training_resume ,
stop_requested ,
)

INTERIM_TEST_INTERVAL =max (1 ,int (TRAIN_INTERIM_CSV_INTERVAL_EPISODES ))
INTERIM_PLOT_INTERVAL =max (1 ,int (TRAIN_INTERIM_GRAPH_INTERVAL_EPISODES ))
VISUALIZER_UPDATE_INTERVAL_STEPS =10


_interrupt_handler =InterruptHandler ()


def format_flops(flops):
    """Render a FLOP count in the largest unit that keeps it above one."""
    for threshold, suffix in ((1e12, "TFLOP"), (1e9, "GFLOP"),
                              (1e6, "MFLOP"), (1e3, "KFLOP")):
        if flops >= threshold:
            return f"{flops / threshold:.2f} {suffix}"
    return f"{flops:.0f} FLOP"


def _episode_limit_or_none (value ):
    """Return a positive episode limit, or None for manually stopped training."""
    if value is None :
        return None
    try :
        limit =int (value )
    except (TypeError ,ValueError ):
        return None
    return limit if limit >0 else None


def sample_episode_payload_strict (demand_data ,episode_steps ,index =None ):
    if int (episode_steps )<=0 :
        raise ValueError (f"episode_steps must be > 0, got {episode_steps}")
    if len (demand_data )==0 :
        raise ValueError ("demand_data is empty")
    _idx =(
    random .randrange (len (demand_data ))
    if index is None else int (index )%len (demand_data )
    )
    _payload =demand_data [_idx ]
    _date =None
    if isinstance (_payload ,dict ):
        _date =_payload .get ("date")
        _payload =_payload .get ("series")
    _data =np .asarray (_payload ,dtype =float ).reshape (-1 )
    if _data .size ==0 :
        raise ValueError ("Sampled demand episode is empty.")

    if _data .size >=int (episode_steps ):
        return _data [:int (episode_steps )],_date
    return np .pad (_data ,(0 ,int (episode_steps )-int (_data .size ))),_date


def create_model_directory (model_name ):
    """
    Create the archive folder for one training run.

    The timestamped directory stores results, TensorBoard files, optional input
    copies, and a source-code snapshot so later analysis can be tied to the exact
    implementation used for the run.
    """
    base_dir =os .getcwd ()

    archive_dir =os .path .join (base_dir ,"archive")
    os .makedirs (archive_dir ,exist_ok =True )

    timestamp =datetime .now ().strftime ("%Y%m%d_%H%M%S")
    model_dir_name =f"{model_name}_{timestamp}"
    model_dir =os .path .join (archive_dir ,model_dir_name )


    os .makedirs (model_dir ,exist_ok =True )

    subdirs =["results","runs","performance","input"]
    for subdir in subdirs :
        os .makedirs (os .path .join (model_dir ,subdir ),exist_ok =True )


    os .makedirs (os .path .join (model_dir ,"results","test"),exist_ok =True )


    try :
        snapshot_code_to_archive (model_dir )
    except Exception as exc :
        warnings .warn (f"Failed to snapshot source code: {exc}")

    return model_dir

def train (
num_episodes =NUM_EPISODES ,
random_window =True ,
agent =None ,
start_episode =0 ,
model_name ="model",
working_dir =None ,
all_rewards =None ,
performance_metrics =None ,
all_episode_data =None ,
profile_episode =None ,
demand_data_override =None ,
arrival_sampler_override =None ,
episode_preparer =None ,
test_demand_data_override =None ,
test_episode_preparer =None ,
interim_eval_fn =None ,
balanced_demand_sampling =False ,
observation_normalization_profile =None ,
resume_run_dir =None ,
resume_context =None ,
resume_checkpoint_interval =100 ,
):
    """
    Run training and return the trained agent plus collected metrics.

    When `agent` is None, Config.MARL_ALGORITHM selects the learner. The
    bank days (`demand_data_override`, `test_demand_data_override`) and their
    episode preparers are required. `all_rewards`, `performance_metrics`, and
    `all_episode_data` allow externally managed runs to continue appending to
    existing containers.
    """
    del random_window, profile_episode
    if demand_data_override is None or episode_preparer is None :
        raise ValueError ("train() needs the bank days and their episode preparer")

    resume_manifest =None
    if resume_run_dir is not None :
        resume_run_dir =os .path .abspath (os .path .expanduser (str (resume_run_dir )))
        resume_manifest =read_resume_manifest (resume_run_dir )
        if working_dir is not None and os .path .abspath (str (working_dir ))!=resume_run_dir :
            raise ValueError (
            f"working_dir={working_dir!r} does not match resume_run_dir={resume_run_dir!r}"
            )
        if resume_context is None :
            raise ValueError ("resume_context is required for exact resume validation")
        working_dir =resume_run_dir
    elif working_dir is None :
        working_dir =create_model_directory (model_name )

    if observation_normalization_profile is not None :
        configure_observation_normalization (observation_normalization_profile )
    normalization_path =os .path .join (
    working_dir ,"input",OBSERVATION_NORMALIZATION_FILENAME
    )
    save_observation_normalization_profile (normalization_path )
    print (f"[normalization] profile={normalization_path}",flush =True )

    set_env_seed (ENV_SEED )


    performance_dir =os .path .join (working_dir ,"performance")
    os .makedirs (performance_dir ,exist_ok =True )
    purge_step =(
        int (resume_manifest ["completed_training_episode"])+1
        if resume_manifest is not None else None
        )
    tb_writer =create_tensorboard_writer (
    log_dir =performance_dir ,purge_step =purge_step
    )

    _interrupt_handler .setup ()


    env =EVEnv (num_stations =NUM_STATIONS ,num_evs =NUM_EVS ,episode_steps =EPISODE_STEPS )
    # The battery never rewrites an EV action and never enters the EV-side
    # reward, so during training its only consumer would be a reported column.
    env .use_residual_bess =False
    episode_limit =_episode_limit_or_none (num_episodes )
    episode_limit_label =str (episode_limit )if episode_limit is not None else "manual-stop"
    agent_num_episodes =episode_limit if episode_limit is not None else 10 **12


    demand_data_train =list (demand_data_override )
    if not demand_data_train :
        raise RuntimeError ("Training bank day pool is empty.")
    arrival_sampler =arrival_sampler_override
    episode_demand ,episode_date =sample_episode_payload_strict (
    demand_data_train ,env .episode_steps ,0 if balanced_demand_sampling else None
    )
    episode_preparer (env ,episode_demand ,episode_date ,arrival_sampler ,0 )
    initial_obs =env ._get_obs ()

    state_dim =initial_obs .shape [1 ]

    if agent is None :
        agent =build_marl_agent (
        s_dim =state_dim ,
        max_evs_per_station =env .max_ev_per_station ,
        n_agent =env .num_stations ,
        num_episodes =agent_num_episodes ,
        batch =BATCH_SIZE ,
        gamma =GAMMA ,
        tau =TAU ,
        lr_a =LR_ACTOR ,
        lr_c =LR_CRITIC_LOCAL ,
        lr_global_c =LR_GLOBAL_CRITIC ,
        tau_global =TAU_GLOBAL ,
        td3_sigma =TD3_SIGMA_GLOBAL ,
        td3_clip =TD3_CLIP_GLOBAL ,
        smoothl1_beta =SMOOTHL1_BETA ,
        )
        if hasattr (agent ,'buf')and hasattr (agent .buf ,'buf_size'):
            agent .buf .buf_size =int (MEMORY_SIZE )
    resume_state =None
    resume_completed_training_episode =0
    resume_completed_environment_episodes =0
    if resume_run_dir is not None :
        resume_state ,_ =load_training_resume (
        resume_run_dir ,expected_context =resume_context ,map_location ="cpu"
        )
        if not hasattr (agent ,"load_training_resume_state_dict"):
            raise TypeError (
            f"agent {type(agent).__name__} does not support exact training resume"
            )
        resume_info =agent .load_training_resume_state_dict (resume_state ["agent"])
        resume_completed_training_episode =int (resume_state ["completed_training_episode"])
        resume_completed_environment_episodes =int (resume_state ["completed_environment_episodes"])
        print (
        "[resume] exact learner state restored: "
        f"training_ep={resume_completed_training_episode} "
        f"environment_episodes={resume_completed_environment_episodes} "
        f"replay={resume_info.get('replay_size')} ptr={resume_info.get('replay_ptr')} "
        f"update_step={resume_info.get('update_step')}",
        flush =True ,
        )

    try :
        runs_dir =os .path .join (working_dir ,"runs")
        os .makedirs (runs_dir ,exist_ok =True )
        if hasattr (agent ,'writer'):
            if agent .writer is not None :
                try :
                    agent .writer .close ()
                except Exception as exc :
                    warnings .warn (f"Failed to close previous TensorBoard writer: {exc}")
        agent .writer =SummaryWriter (log_dir =runs_dir )
        agent .use_tensorboard =True
    except Exception as exc :
        warnings .warn (f"TensorBoard writer setup failed: {exc}")

    active_warmup_steps =int (WARMUP_STEPS )
    active_updates_per_step =max (1 ,int (TRAIN_UPDATES_PER_ENV_STEP ))
    print (
    "[training profile] "
    f"episodes={episode_limit_label} warmup_steps={active_warmup_steps} "
    f"batch={BATCH_SIZE} updates_per_step={active_updates_per_step} "
    f"lr_actor/local/global={LR_ACTOR:g}/{LR_CRITIC_LOCAL:g}/{LR_GLOBAL_CRITIC:g}",
    flush =True ,
    )

    if resume_state is not None :
        # Model/buffer construction and SummaryWriter setup may consume random
        # numbers. Restore last, immediately before the episode loop.
        restore_rng_state (resume_state ["rng"])

    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------
    resume_histories =(resume_state or {}).get ("histories",{})
    if resume_state is not None :
        if all_rewards is not None or performance_metrics is not None or all_episode_data is not None :
            raise ValueError ("exact resume histories cannot be overridden by caller state")
        all_rewards =list (resume_histories .get ("all_rewards")or [])
        all_local_rewards =list (resume_histories .get ("all_local_rewards")or [])
        all_global_rewards =list (resume_histories .get ("all_global_rewards")or [])
        performance_metrics =resume_histories .get ("performance_metrics")
        all_episode_data =dict (resume_histories .get ("all_episode_data")or {})
    else :
        if all_rewards is None :
            all_rewards =[]
        all_local_rewards =[]
        all_global_rewards =[]
        if all_episode_data is None :
            all_episode_data ={}

    if performance_metrics is None :
        performance_metrics ={

        'soc_miss_count':[],
        'surplus_absorption_rate':[],
        'supply_cooperation_rate':[],

        'departing_evs':[],
        'departing_evs_soc_met':[],
        'central_soc_miss_count':[],
        'central_avg_soc_deficit':[],
        'central_departing_evs_soc_met':[],
        'surplus_steps':[],
        'surplus_within_narrow':[],
        'shortage_steps':[],
        'shortage_within_narrow':[],
        'zero_request_steps':[],
        'zero_request_within_narrow':[],
        'system_surplus_steps':[],
        'system_surplus_within_narrow':[],
        'system_shortage_steps':[],
        'system_shortage_within_narrow':[],
        'system_zero_request_steps':[],
        'system_zero_request_within_narrow':[],
        'central_surplus_steps':[],
        'central_surplus_within_narrow':[],
        'central_shortage_steps':[],
        'central_shortage_within_narrow':[],
        'central_zero_request_steps':[],
        'central_zero_request_within_narrow':[],
        'central_tracking_success_rate':[],
        'raw_actor_mae_kw':[],
        'central_corrected_ev_steps':[],
        'central_corrected_station_steps':[],
        'central_correction_active_steps':[],
        'central_absolute_correction_kwh':[],
        'central_max_abs_aggregate_correction_kw':[],
        'central_max_corrected_evs_per_step':[],
        'central_max_corrected_stations_per_step':[],
        'central_max_station_target_error_kw':[],
        'system_tracking_success_rate':[],
        'pre_bess_mae_kw':[],
        'post_bess_mae_kw':[],
        'bess_final_soc_pct':[],
        'bess_throughput_kwh':[],
        'bess_power_limit_hits':[],
        'bess_energy_limit_hits':[],
        'bess_max_abs_power_kw':[],
        'avg_soc_deficit':[],
        }

    if resume_state is not None :
        expected_history =int (resume_completed_training_episode )
        history_lengths ={
        "all_rewards":len (all_rewards ),
        "all_local_rewards":len (all_local_rewards ),
        "all_global_rewards":len (all_global_rewards ),
        "performance_metrics":len (performance_metrics .get ("soc_miss_count",[])),
        }
        bad ={key:value for key ,value in history_lengths .items ()if value !=expected_history }
        if bad :
            raise ValueError (
            f"resume history length mismatch for episode {expected_history}: {bad}"
            )

    def _save_exact_resume (completed_training_ep ,completed_environment_ep ,reason ):
        if resume_context is None :
            return None
        if tb_writer is not None :
            tb_writer .flush ()
        if getattr (agent ,"writer",None )is not None :
            agent .writer .flush ()
        save_start =time .time ()
        print (
        f"[resume] saving exact state at training_ep={int(completed_training_ep)} "
        f"environment_episodes={int(completed_environment_ep)} reason={reason}",
        flush =True ,
        )
        manifest =save_training_resume (
        run_dir =working_dir ,
        agent =agent ,
        completed_training_episode =int (completed_training_ep ),
        completed_environment_episodes =int (completed_environment_ep ),
        all_rewards =all_rewards ,
        all_local_rewards =all_local_rewards ,
        all_global_rewards =all_global_rewards ,
        performance_metrics =performance_metrics ,
        all_episode_data =all_episode_data ,
        context =resume_context ,
        reason =reason ,
        )
        print (
        f"[resume] exact state saved: {manifest['state_file']} "
        f"({manifest['state_bytes'] / (1024 ** 2):.1f} MiB, "
        f"{time.time() - save_start:.1f}s)",
        flush =True ,
        )
        return manifest

    try :
        # A finite limit counts learned episodes only. Warmup episodes are
        # intentionally extra and therefore cannot be represented by a fixed
        # range known before the replay buffer is filled.
        if episode_limit is not None and resume_completed_training_episode >=episode_limit :
            print (
            f"[resume] target episode {episode_limit} is already complete; no training run",
            flush =True ,
            )
            episode_iter =()
        else :
            episode_iter =itertools .count (start_episode +1 )
        learned_this_invocation =0
        for ep in episode_iter :

            local_environment_episode =max (int (ep )-int (start_episode ),1 )
            environment_episode =(
            int (resume_completed_environment_episodes )+local_environment_episode
            )


            # Warmup collects replay transitions using exploratory actions
            # before network updates are allowed to dominate the buffer.
            _is_warmup =agent .buf .size <active_warmup_steps

            if _is_warmup :
                env .record_snapshots =False
                _episode_ordinal =max (int (environment_episode )-1 ,0 )
                _wu_demand ,_wu_date =sample_episode_payload_strict (
                demand_data_train ,env .episode_steps ,
                _episode_ordinal if balanced_demand_sampling else None
                )
                episode_preparer (env ,_wu_demand ,_wu_date ,arrival_sampler ,
                int (environment_episode ))
                agent .episode_start ()
                _wu_prefetch =None
                while True :
                    if _wu_prefetch is not None :
                        _wu_obs =normalize_observation (_wu_prefetch )
                        _wu_prefetch =None
                    else :
                        _wu_obs =normalize_observation (env .begin_step ())
                    _wu_act =agent .act (_wu_obs ,env =env ,noise =True )
                    _ ,_wu_rl ,_wu_rg ,_wu_done ,_wu_info =env .apply_action (
                    _wu_act ,build_info =False ,return_observation =False
                    )
                    if all (_wu_done ):
                        _wu_next =normalize_observation (env ._get_obs ())
                    else :
                        _wu_next_raw =env .begin_step ()
                        _wu_next =normalize_observation (_wu_next_raw )
                        _wu_prefetch =_wu_next_raw
                    _wu_sp =_wu_info .get ('raw_actor_station_powers',_wu_info ['station_powers'])
                    _wu_rl_t =_wu_rl if torch .is_tensor (_wu_rl )else torch .as_tensor (_wu_rl ,dtype =torch .float32 ,device =device )
                    _wu_sp_t =_wu_sp if torch .is_tensor (_wu_sp )else torch .as_tensor (_wu_sp ,dtype =torch .float32 ,device =device )
                    agent .cache_experience (
                    torch .as_tensor (_wu_obs ,dtype =torch .float32 ,device =device ),
                    torch .as_tensor (_wu_next ,dtype =torch .float32 ,device =device ),
                    _wu_act ,
                    _wu_rl_t ,
                    torch .tensor (_wu_rg ,dtype =torch .float32 ,device =device ),
                    torch .as_tensor (_wu_done ,dtype =torch .float32 ,device =device ),
                    actual_station_powers =_wu_sp_t ,
                    actual_ev_power_kw =_wu_info .get ('raw_actor_ev_power_kw',_wu_info .get ('actual_ev_power_kw')),
                    )
                    if not all (_wu_done ):
                        agent .update ()
                    if all (_wu_done ):
                        agent .episode_end ()
                        break
                _wu_buf =agent .buf .size
                if ep %5 ==1 or _wu_buf >=active_warmup_steps :
                    print (f"[WARMUP] ep={ep:4d}  buffer={_wu_buf}/{active_warmup_steps}",flush =True )
                if _interrupt_handler .is_interrupted ()or stop_requested (working_dir ):
                    _save_exact_resume (
                    resume_completed_training_episode ,environment_episode ,"stop-request"
                    )
                    clear_stop_request (working_dir )
                    print ("[resume] stopped safely after warmup episode boundary",flush =True )
                    break
                continue


            # From this point onward, `training_ep` counts learned episodes only;
            # warmup episodes are excluded from reward/performance curves.
            learned_this_invocation +=1
            if learned_this_invocation ==1 :
                agent ._learning_started_episode =environment_episode
                # Warmup is pure replay collection. Start the configured
                # epsilon/noise schedule at learned episode 1, not partway
                # through it after the warmup-day count.
                if resume_state is None and hasattr (agent ,'current_episode'):
                    agent .current_episode =0
                if resume_state is None :
                    print (
                    f"[Info] Warmup complete at ep={environment_episode}. "
                    f"Learning starts (training_ep=1/{episode_limit_label}).",
                    flush =True ,
                    )
                else :
                    print (
                    f"[resume] learning continues at training_ep="
                    f"{resume_completed_training_episode + 1}/{episode_limit_label}",
                    flush =True ,
                    )
            training_ep =resume_completed_training_episode +learned_this_invocation


            env .record_snapshots =bool (getattr (agent ,'test_mode',False ))


            visualizer =(
            GradientLossVisualizer (env .num_stations ,tb_writer )
            if tb_writer is not None else None
            )


            _episode_ordinal =max (int (environment_episode )-1 ,0 )
            episode_demand ,episode_date =sample_episode_payload_strict (
            demand_data_train ,env .episode_steps ,
            _episode_ordinal if balanced_demand_sampling else None
            )
            episode_preparer (
            env ,episode_demand ,episode_date ,arrival_sampler ,
            int (environment_episode )
            )

            agent .episode_start ()
            ep_r =0.0

            ep_start_time =time .time ()

            ep_r =0
            ep_local_r =0.0
            ep_global_r =0.0
            ep_local_sum_tensor =torch .zeros ((),dtype =torch .float32 ,device =device )
            ep_local_mean_tensor =torch .zeros ((),dtype =torch .float32 ,device =device )
            ep_global_tensor =torch .zeros ((),dtype =torch .float32 ,device =device )

            ep_local_departure_r =0.0
            ep_local_progress_shaping_r =0.0

            station_local_reward_sums =[
            {
            "total":0.0 ,
            "departure":0.0 ,
            "progress_shaping":0.0 ,
            }
            for _ in range (env .num_stations )
            ]

            _prefetch_obs =None

            # Step loop: observe, act, apply physics/rewards, cache transition,
            # update the agent, and accumulate diagnostics for this episode.
            while True :

                def all_safe (*values ):
                    safe_tensor =None
                    safe_python =True
                    for x in values :
                        if isinstance (x ,torch .Tensor ):
                            current_safe =torch .isfinite (x ).all ()
                            safe_tensor =current_safe if safe_tensor is None else (safe_tensor &current_safe )
                        else :
                            safe_python =safe_python and bool (np .isfinite (x ).all ())
                    if safe_tensor is None :
                        return safe_python
                    return safe_python and bool (safe_tensor .item ())

                _step_obs_raw =_prefetch_obs
                _prefetch_obs =None
                if _step_obs_raw is not None :
                    obs1 =normalize_observation (_step_obs_raw )
                else :
                    obs1 =normalize_observation (env .begin_step ())

                act_tensor =agent .act (obs1 ,env =env ,noise =True )

                # Training uses the lightweight info payload for speed.
                # Evaluation/timing requests detailed snapshots and reward
                # decomposition for plotting and diagnostics.
                _build_info_full =bool (getattr (agent ,'test_mode',False ))
                _ ,r_local ,r_global ,done ,info =env .apply_action (
                act_tensor ,
                build_info =_build_info_full ,
                return_observation =_build_info_full ,
                )

                if all (done ):
                    next_state =normalize_observation (env ._get_obs ())
                else :
                    _next_raw =env .begin_step ()
                    next_state =normalize_observation (_next_raw )
                    _prefetch_obs =_next_raw

                # Checked on the last step only: a periodic check would force a
                # device synchronization during training.
                finite_check_due =(
                bool (getattr (agent ,'test_mode',False ))
                or all (done )
                )
                if finite_check_due and not all_safe (obs1 ,next_state ,act_tensor ,r_local ,r_global ):
                    # The environment has already moved past this step and
                    # cannot be stepped back, so the step is not retried. Its
                    # transition is neither stored nor learned from, and its
                    # rewards stay out of the episode sums.
                    print (
                    f"Warning: NaN/Inf step data at step {int(env.step_count)}; transition skipped",
                    flush =True ,
                    )
                    if all (done ):
                        agent .episode_end ()
                        break
                    continue

                if torch .is_tensor (r_local ):
                    ep_local_sum_tensor =ep_local_sum_tensor +r_local .detach ().sum ()
                    ep_local_mean_tensor =ep_local_mean_tensor +r_local .detach ().mean ()
                else :
                    ep_local_sum_tensor =ep_local_sum_tensor +torch .as_tensor (sum (r_local ),dtype =torch .float32 ,device =device )
                    ep_local_mean_tensor =ep_local_mean_tensor +torch .as_tensor (np .mean (r_local ),dtype =torch .float32 ,device =device )
                ep_global_tensor =ep_global_tensor +torch .as_tensor (r_global ,dtype =torch .float32 ,device =device )

                if 'reward_breakdown'in info and 'per_station'in info ['reward_breakdown']:
                    for st_idx ,station_data in enumerate (info ['reward_breakdown']['per_station']):

                        dep_r =station_data .get ('departure_reward',0.0 )
                        shaping_r =station_data .get ('progress_shaping',0.0 )
                        total_r =station_data .get ('local_total',0.0 )

                        ep_local_departure_r +=dep_r
                        ep_local_progress_shaping_r +=shaping_r

                        if 0 <=st_idx <len (station_local_reward_sums ):
                            sums =station_local_reward_sums [st_idx ]
                            sums ["total"]+=total_r
                            sums ["departure"]+=dep_r
                            sums ["progress_shaping"]+=shaping_r


                # The policy action is the actor proposal. The central EV
                # allocator is part of environment dynamics, so critics are
                # keyed by the actor's physically clipped pre-allocation
                # powers; the next state still reflects executed corrected
                # EV powers. Keeping executed powers here would train Q on
                # actions the actor never emitted.
                state_tensor_for_buffer =torch .as_tensor (obs1 ,dtype =torch .float32 ,device =device )
                next_state_tensor =torch .as_tensor (next_state ,dtype =torch .float32 ,device =device )

                actual_ev_power_kw_tensor =info .get ('raw_actor_ev_power_kw',info .get ('actual_ev_power_kw'))
                _r_local_t =r_local if torch .is_tensor (r_local )else torch .as_tensor (r_local ,dtype =torch .float32 ,device =device )
                _sp =info .get ('raw_actor_station_powers',info ['station_powers'])
                _sp_t =_sp if torch .is_tensor (_sp )else torch .as_tensor (_sp ,dtype =torch .float32 ,device =device )

                agent .cache_experience (
                state_tensor_for_buffer ,
                next_state_tensor ,
                act_tensor ,
                _r_local_t ,
                torch .tensor (r_global ,dtype =torch .float32 ,device =device ),
                torch .as_tensor (done ,dtype =torch .float32 ,device =device ),
                actual_station_powers =_sp_t ,
                actual_ev_power_kw =actual_ev_power_kw_tensor ,
                )

                for _update_idx in range (active_updates_per_step ):
                    agent .update ()


                # Agents expose diagnostics for TensorBoard. Sampling them
                # every few env steps avoids forcing CPU/GPU synchronization on
                # every transition while preserving episode-level trends.
                visualizer_update_due =(
                visualizer is not None
                and (
                all (done )
                or (int (env .step_count )%VISUALIZER_UPDATE_INTERVAL_STEPS ==0 )
                )
                )
                if visualizer_update_due :
                    if hasattr (agent ,'last_central_q_value'):
                        visualizer .update_central_q_value (agent .last_central_q_value )
                    elif hasattr (agent ,'last_local_q_values_per_agent')and hasattr (agent ,'last_global_q_value'):
                        visualizer .update_q_values (
                        agent .last_local_q_values_per_agent ,
                        np .mean (agent .last_local_q_values_per_agent )if agent .last_local_q_values_per_agent else 0.0 ,
                        agent .last_global_q_value
                        )
                    else :
                        raise AttributeError (
                        "Agent does not expose expected Q diagnostics "
                        "(last_central_q_value or last_local_q_values_per_agent + last_global_q_value)."
                        )

                    visualizer .update_gradients (agent )
                    visualizer .update_losses (agent )
                    visualizer .update_clipping (agent )

                if all (done ):
                    agent .episode_end ()
                    break


            ep_end_time =time .time ()
            ep_duration =ep_end_time -ep_start_time


            steps_in_ep =env .step_count if hasattr (env ,'step_count')and env .step_count >0 else 1

            ep_reward_totals =torch .stack ((
            ep_local_sum_tensor +ep_global_tensor ,
            ep_local_mean_tensor ,
            ep_global_tensor ,
            )).detach ().cpu ().tolist ()
            ep_r =float (ep_reward_totals [0 ])
            ep_local_r =float (ep_reward_totals [1 ])
            ep_global_r =float (ep_reward_totals [2 ])


            _is_warmup =False

            metrics =env .get_metrics ()
            soc_miss_rate =metrics ['soc_miss_rate']
            surplus_absorption_rate =metrics ['surplus_absorption_rate']
            supply_cooperation_rate =metrics ['supply_cooperation_rate']
            surplus_steps =metrics ['surplus_steps']
            surplus_success =metrics ['surplus_within_narrow']
            shortage_steps =metrics ['shortage_steps']
            shortage_success =metrics ['shortage_within_narrow']
            departing_evs_total =env .metrics .get ('departing_evs',0 )
            departing_evs_soc_met =env .metrics .get ('departing_evs_soc_met',0 )
            avg_soc_deficit =metrics .get ('avg_soc_deficit',0.0 )

            if not _is_warmup :
                all_rewards .append (ep_r /steps_in_ep )
                all_local_rewards .append (ep_local_r /steps_in_ep )
                all_global_rewards .append (ep_global_r /steps_in_ep )

                if not hasattr (train ,'all_local_rewards'):
                    train .all_local_rewards =[]
                    train .all_global_rewards =[]
                    train .charge_rates =[]
                    train .discharge_rates =[]
                    train .soc_hit_rates =[]
                train .all_local_rewards .append (ep_local_r /steps_in_ep )
                train .all_global_rewards .append (ep_global_r /steps_in_ep )

                performance_metrics ['soc_miss_count'].append (soc_miss_rate )
                performance_metrics ['avg_soc_deficit'].append (avg_soc_deficit )
                performance_metrics ['surplus_absorption_rate'].append (surplus_absorption_rate )
                performance_metrics ['supply_cooperation_rate'].append (supply_cooperation_rate )
                performance_metrics ['departing_evs'].append (departing_evs_total )
                performance_metrics ['departing_evs_soc_met'].append (departing_evs_soc_met )
                performance_metrics .setdefault ('central_soc_miss_count',[]).append (
                metrics .get ('central_soc_miss_rate',0.0 )
                )
                performance_metrics .setdefault ('central_avg_soc_deficit',[]).append (
                metrics .get ('central_avg_soc_deficit',0.0 )
                )
                performance_metrics .setdefault ('central_departing_evs_soc_met',[]).append (
                metrics .get ('central_departing_evs_soc_met',0 )
                )
                performance_metrics ['surplus_steps'].append (surplus_steps )
                performance_metrics ['surplus_within_narrow'].append (surplus_success )
                performance_metrics ['shortage_steps'].append (shortage_steps )
                performance_metrics ['shortage_within_narrow'].append (shortage_success )
                performance_metrics ['zero_request_steps'].append (metrics .get ('zero_request_steps',0 ))
                performance_metrics ['zero_request_within_narrow'].append (metrics .get ('zero_request_within_narrow',0 ))
                for key in (
                'system_surplus_steps','system_surplus_within_narrow',
                'system_shortage_steps','system_shortage_within_narrow',
                'system_zero_request_steps','system_zero_request_within_narrow',
                'central_surplus_steps','central_surplus_within_narrow',
                'central_shortage_steps','central_shortage_within_narrow',
                'central_zero_request_steps','central_zero_request_within_narrow',
                'central_tracking_success_rate','raw_actor_mae_kw',
                'central_corrected_ev_steps','central_corrected_station_steps',
                'central_correction_active_steps',
                'central_absolute_correction_kwh','central_max_abs_aggregate_correction_kw',
                'central_max_corrected_evs_per_step','central_max_corrected_stations_per_step',
                'central_max_station_target_error_kw',
                'system_tracking_success_rate','pre_bess_mae_kw','post_bess_mae_kw',
                'bess_final_soc_pct','bess_throughput_kwh','bess_power_limit_hits',
                'bess_energy_limit_hits','bess_max_abs_power_kw',
                ):
                    performance_metrics .setdefault (key ,[]).append (metrics .get (key ,0.0 ))

                train .charge_rates .append (surplus_absorption_rate )
                train .discharge_rates .append (supply_cooperation_rate )
                train .soc_hit_rates .append (100 -soc_miss_rate )

            train_parts =[
            f"train{training_ep} ",
            f"SoC actor/physical: {100-soc_miss_rate:.1f}/"
            f"{100-metrics.get('central_soc_miss_rate', 0.0):.1f}%",
            ]
            train_parts .append (f"Surplus: {surplus_success}/{surplus_steps} ({surplus_absorption_rate:.1f}%)")
            train_parts .append (f"Supply: {shortage_success}/{shortage_steps} ({supply_cooperation_rate:.1f}%)")
            # The battery is off in training, so only the actor and central
            # columns carry information.
            train_parts .append (
            f"MARL/Central: {metrics.get('tracking_success_rate', 0.0):.1f}/"
            f"{metrics.get('central_tracking_success_rate', 0.0):.1f}% "
            f"(MAE {metrics.get('raw_actor_mae_kw', 0.0):.1f}->"
            f"{metrics.get('pre_bess_mae_kw', 0.0):.1f} kW)"
            )
            train_parts .append (
            f"Central: corrected={metrics.get('central_corrected_ev_steps', 0)} EV-steps, "
            f"max={metrics.get('central_max_abs_aggregate_correction_kw', 0.0):.1f}kW"
            )
            train_parts .append (f"Duration={ep_duration:.1f}s")
            print (" | ".join (train_parts ),flush =True )

            # ---------------------------------------------------------------
            # ---------------------------------------------------------------
            if tb_writer and not _is_warmup :


                write_train_episode_tb_scalars (
                tb_writer ,training_ep ,steps_in_ep ,
                ep_local_r =ep_local_r ,
                ep_global_r =ep_global_r ,
                soc_miss_rate =soc_miss_rate ,
                central_soc_miss_rate =metrics .get ('central_soc_miss_rate',0.0 ),
                surplus_absorption_rate =surplus_absorption_rate ,
                supply_cooperation_rate =supply_cooperation_rate ,
                raw_actor_tracking_success_rate =metrics .get ('tracking_success_rate',0.0 ),
                central_tracking_success_rate =metrics .get ('central_tracking_success_rate',0.0 ),
                raw_actor_mae_kw =metrics .get ('raw_actor_mae_kw',0.0 ),
                central_corrected_ev_steps =metrics .get ('central_corrected_ev_steps',0 ),
                central_corrected_station_steps =metrics .get ('central_corrected_station_steps',0 ),
                central_absolute_correction_kwh =metrics .get ('central_absolute_correction_kwh',0.0 ),
                central_max_abs_aggregate_correction_kw =metrics .get ('central_max_abs_aggregate_correction_kw',0.0 ),
                central_max_corrected_stations_per_step =metrics .get ('central_max_corrected_stations_per_step',0 ),
                central_max_station_target_error_kw =metrics .get ('central_max_station_target_error_kw',0.0 ),
                system_tracking_success_rate =metrics .get ('system_tracking_success_rate',0.0 ),
                pre_bess_mae_kw =metrics .get ('pre_bess_mae_kw',0.0 ),
                post_bess_mae_kw =metrics .get ('post_bess_mae_kw',0.0 ),
                bess_final_soc_pct =metrics .get ('bess_final_soc_pct',0.0 ),
                bess_throughput_kwh =metrics .get ('bess_throughput_kwh',0.0 ),
                bess_max_abs_power_kw =metrics .get ('bess_max_abs_power_kw',0.0 ),
                bess_power_limit_hits =metrics .get ('bess_power_limit_hits',0 ),
                bess_energy_limit_hits =metrics .get ('bess_energy_limit_hits',0 ),
                ep_local_departure_r =ep_local_departure_r ,
                ep_local_progress_shaping_r =ep_local_progress_shaping_r ,
                station_local_reward_sums =station_local_reward_sums ,
                )
                grad_health_tags =(
                ("last_global_critic_grad_norm_before_clip","GradHealth/global_critic_raw"),
                ("last_global_critic_grad_norm","GradHealth/global_critic_after_clip"),
                ("last_local_critic_grad_norm","GradHealth/local_critic_after_clip"),
                ("last_actor_grad_norm","GradHealth/actor_after_clip"),
                ("last_actor_source_local_grad_norm_before_clip","GradHealth/actor_source_local_raw"),
                ("last_actor_source_global_grad_norm_before_clip","GradHealth/actor_source_global_raw"),
                ("last_actor_source_global_ratio","GradHealth/actor_source_global_ratio"),
                ("last_actor_source_cos","GradHealth/actor_source_local_global_cos"),
                ("last_actor_source_cos_valid_fraction","GradHealth/actor_source_cos_valid_fraction"),
                ("last_global_critic_clip_count","GradHealth/global_critic_clip"),
                ("last_local_critic_clip_count","GradHealth/local_critic_clip_count"),
                ("last_actor_clip_count","GradHealth/actor_clip_count"),
                ("last_global_critic_loss","GradHealth/global_critic_loss"),
                ("last_critic_loss","GradHealth/local_critic_loss"),
                ("last_actor_loss","GradHealth/actor_loss"),
                )
                for attr ,tag in grad_health_tags :
                    if hasattr (agent ,attr ):
                        try :
                            value =float (getattr (agent ,attr ))
                        except (TypeError ,ValueError ):
                            continue
                        if np .isfinite (value ):
                            tb_writer .add_scalar (tag ,value ,training_ep )


            _stop_after_episode =bool (
            _interrupt_handler .is_interrupted ()or stop_requested (working_dir )
            )

            if tb_writer and visualizer is not None and not _is_warmup :
                visualizer .record_to_tensorboard (training_ep )
                visualizer .record_agent_state (agent ,training_ep )

            if visualizer is not None :
                visualizer .reset_episode_data ()


            # Periodic deterministic evaluation on held-out demand data. Actor
            # snapshots are saved next to the TEST* artifact folder so each row
            # in test history is traceable to a loadable checkpoint.

            should_run_interim_csv =(not _is_warmup )and (training_ep %INTERIM_TEST_INTERVAL ==0 )
            should_save_interim_graph =(not _is_warmup )and (training_ep %INTERIM_PLOT_INTERVAL ==0 )
            should_run_interim_test =should_run_interim_csv or should_save_interim_graph

            if should_run_interim_test :
                print (f"------------test{training_ep}")
                test_start =time .time ()
                _ =test (agent ,random_window =False ,working_dir =working_dir ,
                test_results =None ,test_episode_num =training_ep ,
                demand_data_override =test_demand_data_override ,
                arrival_sampler_override =arrival_sampler_override ,
                episode_preparer =test_episode_preparer ,
                num_episodes =max (1 ,int (INTERIM_TEST_EPISODES )),
                eval_seed =int (INTERIM_TEST_SEED ),
                enable_png =bool (should_save_interim_graph ),
                enable_history_png =bool (should_save_interim_graph ),
                save_test_detail_files =True ,
                verbose =True ,
                print_summary =True )

                try :
                    save_dir =os .path .join (working_dir ,"results",f"TEST{training_ep}")
                    os .makedirs (save_dir ,exist_ok =True )
                    agent .save_actors (save_dir ,episode =training_ep )
                except Exception as exc :
                    warnings .warn (f"Failed to save checkpoint actors at episode {training_ep}: {exc}")
                test_duration =time .time ()-test_start
                print (f"test_done{training_ep} ({test_duration:.1f}s)")
                print ("======================================")

                if should_save_interim_graph :
                    try :
                        if tb_writer is not None :
                            tb_writer .flush ()
                        from tools.plot_training_diagnostics import plot_training_diagnostics
                        plot_training_diagnostics (
                        working_dir ,through_episode =training_ep ,
                        output_dir =os .path .join (working_dir ,"results",f"TEST{training_ep}")
                        )
                    except Exception as exc :
                        warnings .warn (f"Failed to write training diagnostics at episode {training_ep}: {exc}")

                # Optional periodic controller-precision evaluation hook.
                # Runs with the current agent so unlimited runs still report it.
                if interim_eval_fn is not None :
                    try :
                        interim_eval_fn (agent ,int (training_ep ),working_dir )
                    except Exception as exc :
                        warnings .warn (f"interim_eval_fn failed at episode {training_ep}: {exc}")

                try :
                    interim_results_dir =os .path .join (working_dir ,"results")
                    os .makedirs (interim_results_dir ,exist_ok =True )
                    # Root-level train summaries must stay aligned with the CSV
                    # written by the same helper. Detailed/interim plots are
                    # still gated by should_save_interim_graph in test().
                    skip_png_flag =False

                    if len (all_local_rewards )>0 and len (all_global_rewards )>0 :
                        plot_daily_rewards (all_local_rewards ,all_global_rewards ,
                        interim_results_dir ,episode_num =len (all_local_rewards ),
                        performance_metrics =performance_metrics ,title_prefix ="Train Results",
                        skip_png =skip_png_flag )

                    if performance_metrics and len (performance_metrics .get ('soc_miss_count',[]))>0 :
                        plot_performance_metrics (performance_metrics ,interim_results_dir ,title_prefix ="Train Results",
                        skip_png =skip_png_flag )
                except Exception as exc :
                    warnings .warn (f"Failed to write interim training plots at episode {training_ep}: {exc}")

            _stop_after_episode =bool (
            _stop_after_episode or _interrupt_handler .is_interrupted ()
            or stop_requested (working_dir )
            )
            _completion_due =bool (
            episode_limit is not None and int (training_ep )>=int (episode_limit )
            )
            _resume_interval =max (0 ,int (resume_checkpoint_interval or 0 ))
            _periodic_resume_due =bool (
            resume_context is not None and _resume_interval >0
            and int (training_ep )%_resume_interval ==0
            )
            if _stop_after_episode or _completion_due or _periodic_resume_due :
                if _stop_after_episode :
                    _resume_reason ="stop-request"
                elif _completion_due :
                    _resume_reason ="complete"
                else :
                    _resume_reason ="periodic"
                _save_exact_resume (training_ep ,environment_episode ,_resume_reason )

            if _stop_after_episode :
                clear_stop_request (working_dir )
                print (
                f"[resume] stopped safely after training_ep={training_ep}; "
                "restart with pre_train.py --resume-run <run_dir>",
                flush =True ,
                )
                break

            if _completion_due :
                break

    except Exception as e :
        print (f"Training interrupted at episode {len(all_rewards)}: {e}")
        traceback .print_exc ()

    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------
    results_dir =os .path .join (working_dir ,"results")
    os .makedirs (results_dir ,exist_ok =True )

    skip_png_flag =False
    if len (all_local_rewards )>0 and len (all_global_rewards )>0 :
        plot_daily_rewards (all_local_rewards ,all_global_rewards ,
        results_dir ,episode_num =len (all_local_rewards ),
        performance_metrics =performance_metrics ,title_prefix ="Train Results",
        skip_png =skip_png_flag )

    if performance_metrics and len (performance_metrics .get ('soc_miss_count',[]))>0 :
        plot_performance_metrics (performance_metrics ,results_dir ,title_prefix ="Train Results",
        skip_png =skip_png_flag )

    return agent ,all_rewards ,performance_metrics ,all_episode_data ,working_dir


if __name__ =="__main__":


     ag ,all_rewards ,perf ,ep_data ,work_dir =train ()
