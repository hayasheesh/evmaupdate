
"""
MADDPG agent for multi-station EV charging.

This module defines the training-time agent used by `training/train.py`.
Each station owns an actor and a twin local critic pair. A separate global critic
evaluates system-wide dispatch tracking from all stations at once. The actor
update mixes the station-local objective, which is mainly driven by EV SoC
completion, with the global objective, which is mainly driven by aggregate
power tracking.

Inputs are normalized observations from `environment.normalize`, normalized
actions in [-1, 1], local rewards with shape [batch, stations], and a scalar
global reward per transition. Physical charging limits are applied before the
critics see an action, so Q-values are trained on executable kW commands rather
than on unclipped neural-network outputs.

The update sequence is:
1. sample replay transitions after warmup,
2. update station-local critics with the long-horizon local discount,
3. update the global critic with `GAMMA_GLOBAL`,
4. update actors every `POLICY_DELAY` steps by explicitly mixing local and
   global policy gradients,
5. Polyak-average target actors and critics.
"""

import copy
import math
import os

import torch
import torch .nn as nn
import torch .nn .functional as F
import torch .optim as optim

from environment.normalize import (
denormalize_ev_capacity_kwh ,
denormalize_ev_max_power_kw ,
denormalize_soc ,
)
from Config import (
NUM_EPISODES ,BATCH_SIZE ,GAMMA ,GAMMA_GLOBAL ,TAU ,TAU_GLOBAL ,
LOCAL_CRITIC_HIDDEN_SIZE ,GLOBAL_CRITIC_HIDDEN_SIZE ,
LR_ACTOR ,LR_CRITIC_LOCAL ,LR_GLOBAL_CRITIC ,LOCAL_REWARD_SCALE ,
RANDOM_ACTION_RANGE ,SMOOTHL1_BETA ,
EPSILON_START_EPISODE ,EPSILON_END_EPISODE ,EPSILON_INITIAL ,EPSILON_FINAL ,
OU_NOISE_START_EPISODE ,OU_NOISE_END_EPISODE ,
OU_NOISE_SCALE_INITIAL ,OU_NOISE_SCALE_FINAL ,OU_NOISE_GAIN ,
OU_SIGMA ,OU_CLIP ,
TD3_SIGMA_GLOBAL ,TD3_CLIP_GLOBAL ,
POLICY_DELAY ,
MEMORY_SIZE ,WARMUP_STEPS ,
Q_MIX_GLOBAL_WEIGHT ,
POWER_TO_ENERGY ,
MAX_EV_PER_STATION ,EV_CHARGER_POWER_OBS_SCALE_KW ,
BIAS_GRAD_CLIP_MAX ,GRAD_CLIP_MAX ,GRAD_CLIP_MAX_GLOBAL ,
)

from environment.observation_config import (
EV_FEAT_DIM ,
EV_FEATURE_NAMES ,
LOCAL_TAIL_DIM ,
LOCAL_TAIL_FEATURE_NAMES ,GLOBAL_TAIL_FEATURE_NAMES ,
)

from .actor import Actor
from .critic import LocalEvMLPCritic ,GlobalMLPCritic
from .replay_buffer import ReplayBuffer
from .noise import (
GaussianNoise ,
linear_epsilon_decay ,
)

device =torch .device ("cuda"if torch .cuda .is_available ()else "cpu")


SOC_FEATURE_IDX =EV_FEATURE_NAMES .index ("soc")
CAPACITY_FEATURE_IDX =EV_FEATURE_NAMES .index ("battery_capacity_kwh")
MAX_POWER_FEATURE_IDX =EV_FEATURE_NAMES .index ("max_power_kw")


def _clip_bias_gradients (model ,max_norm =1.0 ):
    """Clip only bias gradients before the full-parameter gradient clip.

    Every bias is still clipped on its own norm, exactly as the per-tensor
    loop did; the tensors are only visited together.  That matters because
    this runs 23 times per update -- fourteen local critics, seven actors, two
    global critics -- and each tensor cost its own handful of kernel launches
    in a workload already spending most of its wall clock waiting on launches
    rather than arithmetic.  The results are bit-identical: `_foreach_norm`
    returns the same per-tensor two-norm, and the scale factor and the
    multiply are unchanged.
    """
    grads =[
    param .grad for name ,param in model .named_parameters ()
    if param .grad is not None and 'bias'in name
    ]
    if not grads :
        return
    norms =torch ._foreach_norm (grads )
    coefs =[torch .clamp (max_norm /(norm +1e-6 ),max =1.0 )for norm in norms ]
    torch ._foreach_mul_ (grads ,coefs )


def _all_grads_finite (model ,loss ):
    """Is the loss finite, and every gradient element with it?

    The per-tensor loop this replaces launched `isfinite(...).all()` once per
    parameter tensor and combined the answers one at a time, which is a handful
    of launches per model and 23 models per update.  The infinity-norm is the
    largest absolute element, so it is non-finite exactly when some element is,
    and `_foreach_norm` takes all the tensors in one fused launch.

    The two-norm would not do: a sum of squares can overflow to infinity while
    every element is finite, and that would report a gradient as unusable when
    the old check called it usable.  A maximum cannot overflow, so the verdict
    is the one the loop gave.
    """
    finite =torch .isfinite (loss .detach ())
    grads =[p .grad for p in model .parameters ()if p .grad is not None ]
    if grads :
        peaks =torch .stack (torch ._foreach_norm (grads ,float ('inf')))
        finite =finite &torch .isfinite (peaks ).all ()
    return finite .detach ()


class MADDPG :
    def __init__ (self ,s_dim ,max_evs_per_station ,n_agent ,
    gamma =GAMMA ,
    tau =TAU ,
    batch =BATCH_SIZE ,
    lr_a =LR_ACTOR ,lr_c =LR_CRITIC_LOCAL ,lr_global_c =LR_GLOBAL_CRITIC ,
    num_episodes =NUM_EPISODES ,
    tau_global :float =TAU_GLOBAL ,
    td3_sigma :float =TD3_SIGMA_GLOBAL ,
    td3_clip :float =TD3_CLIP_GLOBAL ,
    smoothl1_beta =1.0 ,
    **kwargs ):
        """Create actors, critics, target networks, replay memory, and exploration state."""
        del num_episodes
        del kwargs
        if max_evs_per_station !=MAX_EV_PER_STATION :
            raise AssertionError (
            f"max_evs_per_station={max_evs_per_station} != Config.MAX_EV_PER_STATION={MAX_EV_PER_STATION}. "
            "GlobalMLPCritic and normalize.py use Config.MAX_EV_PER_STATION directly; "
            "passing a different value causes silent shape mismatches."
            )
        self .s_dim ,self .a_dim ,self .n =s_dim ,max_evs_per_station ,n_agent
        self .max_ev_per_station =max_evs_per_station
        self .gamma ,self .tau ,self .batch =gamma ,tau ,batch
        self .lr_global_c =lr_global_c
        self .current_episode =0

        self .actor_norms =[0 ]*n_agent
        self .critic_norms =[0 ]*n_agent
        self .actor_losses =[]
        self .critic_losses =[]
        self .last_global_critic_loss =0.0
        self .last_actor_loss =0.0
        self .last_critic_loss =0.0
        self .last_actor_grad_norm =0.0
        self .last_local_critic_grad_norm =0.0
        self .last_global_critic_grad_norm =0.0
        self .last_actor_clip_count =0
        self .last_local_critic_clip_count =0
        self .last_global_critic_clip_count =0
        self .local_critic_clip_counts =[0 ]*n_agent
        self .actor_clip_counts =[0 ]*n_agent
        self .critic_norms_before_clip =[0 ]*n_agent
        self .actor_norms_before_clip =[0 ]*n_agent
        self .actor_source_local_norms_before_clip =[0.0 ]*n_agent
        self .actor_source_global_norms_before_clip =[0.0 ]*n_agent
        self .actor_source_global_ratio =[0.0 ]*n_agent
        self .actor_source_cos =[0.0 ]*n_agent
        self .actor_source_cos_valid =[0 ]*n_agent
        self .last_global_critic_grad_norm_before_clip =0.0
        self .last_actor_source_local_grad_norm_before_clip =0.0
        self .last_actor_source_global_grad_norm_before_clip =0.0
        self .last_actor_source_global_ratio =0.0
        self .last_actor_source_cos =0.0
        self .last_actor_source_cos_valid_fraction =0.0
        self .last_local_q_values_per_agent =[0.0 ]*n_agent
        self .local_q_twin_gap_values_per_agent =[0.0 ]*n_agent
        self .last_local_q_twin_gap_values_per_agent =[0.0 ]*n_agent
        self .last_local_q_twin_gap_mean =0.0
        self .last_global_q_value =0.0
        self .global_reward_scale =1.0
        self .global_reward_baseline =0.0
        self .last_global_reward_scale =1.0
        self .last_global_reward_baseline =0.0
        self .last_global_reward_raw_abs_mean =0.0
        self .last_global_reward_centered_abs_mean =0.0
        self .last_global_reward_term_abs_mean =0.0
        self .last_global_td_target_abs_mean =0.0
        self .last_global_td_error_abs_mean =0.0
        self .last_global_current_q_abs_mean =0.0
        self .last_global_target_q_abs_mean =0.0

        self .env =None
        self .use_tensorboard =False
        self .writer =None

        self .random_action_range =RANDOM_ACTION_RANGE

        self .epsilon_start_episode =EPSILON_START_EPISODE
        self .epsilon_end_episode =EPSILON_END_EPISODE
        self .epsilon_initial =EPSILON_INITIAL
        self .epsilon_final =EPSILON_FINAL
        self .epsilon =self .epsilon_initial

        self .ou_noise_start_episode =OU_NOISE_START_EPISODE
        self .ou_noise_end_episode =OU_NOISE_END_EPISODE
        self .ou_noise_scale_initial =OU_NOISE_SCALE_INITIAL
        self .ou_noise_scale_final =OU_NOISE_SCALE_FINAL
        self .ou_noise_scale =self .ou_noise_scale_initial

        self .ou_noise =GaussianNoise (
        n_agent ,max_evs_per_station ,
        sigma =float (OU_SIGMA ),
        clip =float (OU_CLIP )if OU_CLIP is not None and OU_CLIP >0 else None ,
        )

        self .test_mode =False

        self .buf =ReplayBuffer (cap =int (MEMORY_SIZE ))
        self .buf .maddpg_ref =self
        self .max_evs =max_evs_per_station

        self .active_evs =torch .zeros (n_agent ,dtype =torch .long ,device =device )
        self .active_evs_tensor =self .active_evs
        self .active_slot_mask =torch .zeros (
        n_agent ,max_evs_per_station ,dtype =torch .bool ,device =device
        )

        self .ev_state_dim =EV_FEAT_DIM
        self .local_tail_dim =LOCAL_TAIL_DIM
        self .station_state_dim =self .ev_state_dim *self .max_evs +self .local_tail_dim

        self .actors =[
        Actor (s_dim ,max_evs_per_station ,station_state_dim =self .station_state_dim ).to (device )
        for _ in range (n_agent )
        ]
        self .t_actors =[copy .deepcopy (ac )for ac in self .actors ]

        def local_critic ():
            return LocalEvMLPCritic (
            ev_feat_dim =EV_FEAT_DIM ,
            a_dim =max_evs_per_station ,
            max_evs =max_evs_per_station ,
            hid =LOCAL_CRITIC_HIDDEN_SIZE ,
            station_state_dim =self .station_state_dim ,
            ).to (device )

        self .critics =[local_critic ()for _ in range (n_agent )]
        self .critics2 =[local_critic ()for _ in range (n_agent )]
        self .t_critics =[copy .deepcopy (cr )for cr in self .critics ]
        self .t_critics2 =[copy .deepcopy (cr )for cr in self .critics2 ]


        # Global critic targets use their own Polyak rate and target smoothing
        # scale because the dispatch-tracking value has a shorter horizon than
        # the station-local SoC value.
        self .tau_global =float (tau_global )
        self .td3_sigma =float (td3_sigma )
        self .td3_clip =float (td3_clip )

        # Keep the constructor contract identical to the tensor assembled by
        # _convert_to_global_critic_obs().  In particular, this includes the
        # instruction scale and tracking-enabled flag in GLOBAL_TAIL_DIM.
        additional_features =int (len (GLOBAL_TAIL_FEATURE_NAMES ))
        self ._global_station_dim =EV_FEAT_DIM *self .max_evs
        global_obs_dim =(self .n *self ._global_station_dim )+additional_features

        global_critic_cls =GlobalMLPCritic
        self .global_reward_scale =1.0
        self .global_reward_baseline =0.0
        self .last_global_reward_scale =self .global_reward_scale
        self .last_global_reward_baseline =self .global_reward_baseline

        self .global_critic1 =global_critic_cls (
        global_obs_dim ,max_evs_per_station ,n_agent ,
        hid =GLOBAL_CRITIC_HIDDEN_SIZE ,
        station_state_dim =self ._global_station_dim ,
        init_gain =0.3 ,
        ).to (device )
        self .global_critic2 =global_critic_cls (
        global_obs_dim ,max_evs_per_station ,n_agent ,
        hid =GLOBAL_CRITIC_HIDDEN_SIZE ,
        station_state_dim =self ._global_station_dim ,
        init_gain =0.3 ,
        ).to (device )
        self .t_global_critic1 =copy .deepcopy (self .global_critic1 )
        self .t_global_critic2 =copy .deepcopy (self .global_critic2 )

        self .opt_a =[optim .Adam (self .actors [i ].parameters (),lr =lr_a )for i in range (n_agent )]
        self .opt_c =[optim .Adam (self .critics [i ].parameters (),lr =lr_c )for i in range (n_agent )]
        self .opt_c2 =[optim .Adam (self .critics2 [i ].parameters (),lr =lr_c )for i in range (n_agent )]
        self .opt_global_c1 =optim .Adam (self .global_critic1 .parameters (),lr =self .lr_global_c )
        self .opt_global_c2 =optim .Adam (self .global_critic2 .parameters (),lr =self .lr_global_c )

        self .loss_fn =nn .SmoothL1Loss (beta =smoothl1_beta )

        self .clip_bias_gradients =_clip_bias_gradients

        self ._ep_q_raw_global =[]
        self ._ep_q_raw_local =[[]for _ in range (n_agent )]

        self .policy_delay =max (1 ,int (POLICY_DELAY ))

        # Gradient updates start once the buffer holds this many transitions.
        self .warmup_steps =int (WARMUP_STEPS )

        self .update_step =0


    def update_active_evs (self ,env ):
        """Synchronize the agent-side active-slot counts with the environment mask."""
        num_stations =min (self .n ,env .num_stations )
        counts =torch .zeros (self .n ,dtype =torch .long ,device =device )
        counts [:num_stations ]=env .ev_mask [:num_stations ].sum (dim =1 ).to (
        device =device ,dtype =torch .long
        )
        slot_idx =torch .arange (self .max_evs ,device =device )
        self .active_evs =counts
        self .active_evs_tensor =counts
        self .active_slot_mask =slot_idx .unsqueeze (0 )<counts .unsqueeze (1 )
        self .env =env

    def _convert_to_global_critic_obs (self ,s ,actual_station_powers ):
        """
        Build the flat global-critic state from per-station observations.

        `s` has shape [batch, stations, station_state_dim]. The global critic
        receives all EV feature blocks concatenated across stations, followed
        by optional aggregate power, optional normalized time step, and a
        global demand look-ahead vector derived from the local observation tail.
        `actual_station_powers` is the physically clipped station power in kW
        for the action being evaluated.
        """
        B =s .size (0 )

        ev_features_per_station =self .max_evs *EV_FEAT_DIM
        ev_features_all =s [:,:,:ev_features_per_station ]
        station_features_flat =ev_features_all .reshape (B ,-1 )

        # The largest charger rating, so a fleet of high-rated chargers does
        # not saturate the input.
        max_possible_power =EV_CHARGER_POWER_OBS_SCALE_KW *self .n *self .max_evs
        total_ev_power_raw =actual_station_powers .sum (dim =1 ,keepdim =True )
        total_ev_power =torch .clamp (total_ev_power_raw /max_possible_power ,-1.0 ,1.0 )
        local_tail =s [:,0 ,ev_features_per_station :ev_features_per_station +int (LOCAL_TAIL_DIM )]
        local_index ={name :idx for idx ,name in enumerate (LOCAL_TAIL_FEATURE_NAMES )}

        parts =[station_features_flat ]
        for name in GLOBAL_TAIL_FEATURE_NAMES :
            if name =="total_power":
                parts .append (total_ev_power )
            elif name in local_index and local_index [name ]<local_tail .size (1 ):
                idx =local_index [name ]
                parts .append (local_tail [:,idx :idx +1 ])
            else :
                parts .append (s .new_zeros (B ,1 ))
        global_obs =torch .cat (parts ,dim =1 )
        return global_obs


    def _extract_ev_physics (self ,ev_block ):
        """Return SoC percent, capacity kWh, and per-slot max power from EV features."""
        current_socs =denormalize_soc (ev_block [...,SOC_FEATURE_IDX ])
        capacity_kwh =denormalize_ev_capacity_kwh (ev_block [...,CAPACITY_FEATURE_IDX ])
        capacity_kwh =torch .clamp (capacity_kwh ,min =1e-6 )
        max_power_kw =denormalize_ev_max_power_kw (ev_block [...,MAX_POWER_FEATURE_IDX ])
        max_power_kw =torch .clamp (max_power_kw ,min =0.0 )
        return current_socs ,capacity_kwh ,max_power_kw

    def _ev_physics_factors (self ,capacity_kwh ,max_power_kw ):
        safe_capacity_kwh =torch .clamp (capacity_kwh ,min =1e-6 )
        safe_max_power_kw =torch .clamp (max_power_kw ,min =1e-6 )
        soc_step_per_kw =float (POWER_TO_ENERGY )*100.0 /safe_capacity_kwh
        kw_per_soc_step =safe_capacity_kwh /(100.0 *float (POWER_TO_ENERGY ))
        inv_max_power_kw =torch .reciprocal (safe_max_power_kw )
        return soc_step_per_kw ,kw_per_soc_step ,inv_max_power_kw

    def _normalize_power_by_limit (self ,power_kw ,max_power_kw ,inv_max_power_kw =None ):
        if inv_max_power_kw is None :
            inv_max_power_kw =torch .reciprocal (torch .clamp (max_power_kw ,min =1e-6 ))
        return torch .clamp (power_kw *inv_max_power_kw ,-1.0 ,1.0 )

    def _apply_soc_constraint (
    self ,actions_kw ,current_socs ,ev_padding_mask =None ,use_ste =False ,
    max_power_kw =None ,soc_step_per_kw =None ,kw_per_soc_step =None ,
    ):
        """
        Clip per-EV kW actions to each charger and to one step of SoC.

        The final one-step energy change is bounded by EV SoC. With `use_ste`,
        both bounds keep a straight-through gradient.
        """
        bounded_actions_kw =actions_kw
        if max_power_kw is not None :
            max_power_kw =torch .clamp (max_power_kw ,min =0.0 )
            bounded =torch .clamp (actions_kw ,-max_power_kw ,max_power_kw )
            if use_ste :
                bounded_actions_kw =actions_kw +(bounded -actions_kw ).detach ()
            else :
                bounded_actions_kw =bounded

        proposed_delta_soc =bounded_actions_kw *soc_step_per_kw
        max_charge =100.0 -current_socs
        max_discharge =current_socs
        clamped_soc =torch .clamp (proposed_delta_soc ,-max_discharge ,max_charge )
        if use_ste :
            clamped_soc =proposed_delta_soc +(clamped_soc -proposed_delta_soc ).detach ()
        clamped_actions_kw =clamped_soc *kw_per_soc_step
        if ev_padding_mask is not None :
            clamped_actions_kw =clamped_actions_kw .masked_fill (ev_padding_mask ,0.0 )
        station_powers_kw =clamped_actions_kw .sum (dim =2 )
        return clamped_actions_kw ,station_powers_kw


    def act (self ,state ,env =None ,noise =True ):
        """Return normalized station-by-slot actions for the current observation."""
        if self .test_mode :
            noise =False

        state =torch .as_tensor (state ,dtype =torch .float32 ,device =device )

        ev_block =state [:,:self .max_evs *EV_FEAT_DIM ].reshape (
        self .n ,self .max_evs ,EV_FEAT_DIM
        )
        active_slot_mask =ev_block [...,0 ]>0.5
        self .active_slot_mask =active_slot_mask
        self .active_evs =active_slot_mask .sum (dim =1 ).to (dtype =torch .long )
        self .active_evs_tensor =self .active_evs
        if env is not None :
            self .env =env

        tensor_actions =torch .zeros ((self .n ,self .max_evs ),dtype =torch .float32 ,device =device )
        action_mask =active_slot_mask

        # Sample one noise tensor for the whole step. active_slot_mask gates which
        # slots actually receive it.
        is_training =noise and not self .test_mode
        step_noise =None
        if is_training and self .ou_noise_scale >0.0 :
            step_noise =self .ou_noise .sample (action_mask )

        with torch .no_grad ():
            for agent_idx in range (self .n ):
                agent_state =state [agent_idx :agent_idx +1 ]
                a =self .actors [agent_idx ](agent_state ).squeeze (0 )
                slot_mask =action_mask [agent_idx ]

                if is_training :
                    # 1) Continuous Gaussian perturbation on the policy output.
                    if step_noise is not None :
                        a =a +(OU_NOISE_GAIN *self .ou_noise_scale )*step_noise [agent_idx ]

                    # 2) Per-slot epsilon-greedy replacement.
                    eps_now =float (self .epsilon )
                    if eps_now >0.0 :
                        mask =(torch .rand_like (a )<eps_now )&slot_mask
                        low ,high =float (self .random_action_range [0 ]),float (self .random_action_range [1 ])
                        rand_a =torch .empty_like (a ).uniform_ (low ,high )
                        a =torch .where (mask ,rand_a ,a )

                a =torch .clamp (a ,-1.0 ,1.0 )
                tensor_actions [agent_idx ]=a .masked_fill (~slot_mask ,0.0 )

        return tensor_actions


    def _zero_update_logs (self ):
        """Reset scalar diagnostics when no gradient update is performed."""
        self .last_actor_loss =0.0
        self .last_critic_loss =0.0
        self .last_actor_grad_norm =0.0
        self .last_local_critic_grad_norm =0.0
        self .last_global_critic_grad_norm =0.0
        self .last_actor_source_local_grad_norm_before_clip =0.0
        self .last_actor_source_global_grad_norm_before_clip =0.0
        self .last_actor_source_global_ratio =0.0
        self .last_actor_source_cos =0.0
        self .last_actor_source_cos_valid_fraction =0.0
        self .last_local_q_twin_gap_values_per_agent =[0.0 ]*self .n
        self .last_local_q_twin_gap_mean =0.0
        self .last_global_reward_scale =self .global_reward_scale
        self .last_global_reward_baseline =self .global_reward_baseline
        self .last_global_reward_raw_abs_mean =0.0
        self .last_global_reward_centered_abs_mean =0.0
        self .last_global_reward_term_abs_mean =0.0
        self .last_global_td_target_abs_mean =0.0
        self .last_global_td_error_abs_mean =0.0
        self .last_global_current_q_abs_mean =0.0
        self .last_global_target_q_abs_mean =0.0


    def _build_update_ctx (self ,s ,s2 ,a ,r_local ,d ,
    actual_station_powers ,actual_ev_power_kw ):
        """Precompute masks, action tensors, and objective switches for one update."""
        batch_size =s .size (0 )
        max_evs =self .a_dim

        ev_block =s [:,:,:max_evs *EV_FEAT_DIM ].reshape (
        batch_size ,self .n ,self .max_evs ,EV_FEAT_DIM )
        current_socs ,capacity_kwh ,max_power_kw =self ._extract_ev_physics (ev_block )
        soc_step_per_kw ,kw_per_soc_step ,inv_max_power_kw =self ._ev_physics_factors (
        capacity_kwh ,max_power_kw
        )
        presence_mask =(ev_block [...,0 ]<=0.5 )
        ev_padding_mask =presence_mask
        key_padding_mask =presence_mask .all (dim =2 )

        a_actual =(
        self ._normalize_power_by_limit (actual_ev_power_kw ,max_power_kw ,inv_max_power_kw )
        if actual_ev_power_kw is not None else a
        )

        ev_block_s2 =s2 [:,:,:max_evs *EV_FEAT_DIM ].reshape (
        batch_size ,self .n ,self .max_evs ,EV_FEAT_DIM )
        current_socs_s2 ,capacity_kwh_s2 ,max_power_kw_s2 =self ._extract_ev_physics (ev_block_s2 )
        soc_step_per_kw_s2 ,kw_per_soc_step_s2 ,inv_max_power_kw_s2 =self ._ev_physics_factors (
        capacity_kwh_s2 ,max_power_kw_s2
        )
        presence_mask_s2 =(ev_block_s2 [...,0 ]<=0.5 )
        ev_padding_mask_s2 =presence_mask_s2

        # w = 1 trains the actors from the global critic alone (ABG); the
        # local critics are then not updated.
        skip_local =(Q_MIX_GLOBAL_WEIGHT ==1.0 )
        # Station power enters the critics divided by what ten chargers of the
        # largest rating can draw.
        max_station_power =EV_CHARGER_POWER_OBS_SCALE_KW *MAX_EV_PER_STATION
        w_eff =Q_MIX_GLOBAL_WEIGHT

        return {
        's':s ,'s2':s2 ,
        'r_local':r_local ,'d':d ,
        'actual_station_powers':actual_station_powers ,
        'a_actual':a_actual ,
        'batch_size':batch_size ,'max_evs':max_evs ,
        'current_socs':current_socs ,
        'capacity_kwh':capacity_kwh ,
        'max_power_kw':max_power_kw ,
        'soc_step_per_kw':soc_step_per_kw ,
        'kw_per_soc_step':kw_per_soc_step ,
        'inv_max_power_kw':inv_max_power_kw ,
        'current_socs_s2':current_socs_s2 ,
        'capacity_kwh_s2':capacity_kwh_s2 ,
        'max_power_kw_s2':max_power_kw_s2 ,
        'soc_step_per_kw_s2':soc_step_per_kw_s2 ,
        'kw_per_soc_step_s2':kw_per_soc_step_s2 ,
        'inv_max_power_kw_s2':inv_max_power_kw_s2 ,
        'ev_block':ev_block ,
        'ev_block_s2':ev_block_s2 ,
        'ev_padding_mask':ev_padding_mask ,
        'ev_padding_mask_s2':ev_padding_mask_s2 ,
        'key_padding_mask':key_padding_mask ,
        'skip_local':skip_local ,
        'max_station_power':max_station_power ,
        'w_eff':w_eff ,
        }


    def _update_local_critics (self ,ctx ):
        """
        Update one station-local critic per station.

        Local critics use the long-horizon discount `self.gamma` because their
        reward is tied to SoC progress over the dwell time of each EV. Target
        actions are generated by target actors, smoothed with clipped TD3 noise,
        clipped to SoC feasibility, and evaluated by the twin target local critics.
        """
        if ctx ['skip_local']:
            self .critic_losses =[]
            self .local_q_twin_gap_values_per_agent =[0.0 ]*self .n
            self .last_local_q_twin_gap_values_per_agent =[0.0 ]*self .n
            self .last_local_q_twin_gap_mean =0.0
            return 0

        s ,s2 =ctx ['s'],ctx ['s2']
        a_actual =ctx ['a_actual']
        r_local ,d =ctx ['r_local'],ctx ['d']
        actual_station_powers =ctx ['actual_station_powers']
        ev_padding_mask_s2 =ctx ['ev_padding_mask_s2']
        max_station_power =ctx ['max_station_power']
        current_socs_s2 =ctx ['current_socs_s2']
        max_power_kw_s2 =ctx ['max_power_kw_s2']
        soc_step_per_kw_s2 =ctx ['soc_step_per_kw_s2']
        kw_per_soc_step_s2 =ctx ['kw_per_soc_step_s2']
        inv_max_power_kw_s2 =ctx ['inv_max_power_kw_s2']
        local_critic_clip_count =0

        with torch .no_grad ():
            next_actions_all =torch .stack (
            [self .t_actors [i ](s2 [:,i ,:])for i in range (self .n )],dim =1
            )
            local_noise =torch .randn_like (next_actions_all )*self .td3_sigma
            local_noise =torch .clamp (local_noise ,-self .td3_clip ,self .td3_clip )
            next_actions_all =next_actions_all +local_noise
            next_actions_all =torch .clamp (next_actions_all ,-1.0 ,1.0 )

            next_actions_kw =next_actions_all *max_power_kw_s2
            next_actions_clamped ,next_agent_powers =self ._apply_soc_constraint (
            next_actions_kw ,current_socs_s2 ,ev_padding_mask_s2 ,
            max_power_kw =max_power_kw_s2 ,
            soc_step_per_kw =soc_step_per_kw_s2 ,kw_per_soc_step =kw_per_soc_step_s2 ,
            )
            next_powers_norm =torch .clamp (next_agent_powers /max_station_power ,-1.0 ,1.0 )
            next_actions_normalized =self ._normalize_power_by_limit (
            next_actions_clamped ,max_power_kw_s2 ,inv_max_power_kw_s2
            )
            target_qs =[]
            for i in range (self .n ):
                tq1 =self .t_critics [i ](
                s2 [:,i ,:],next_actions_normalized [:,i ,:],
                actual_station_powers =next_powers_norm [:,i ],
                )
                tq2 =self .t_critics2 [i ](
                s2 [:,i ,:],next_actions_normalized [:,i ,:],
                actual_station_powers =next_powers_norm [:,i ],
                )
                tq =torch .minimum (tq1 ,tq2 )
                tq =torch .nan_to_num_ (tq ,nan =0.0 ,posinf =0.0 ,neginf =0.0 )
                target_qs .append (tq )

        y_targets =[]
        for i in range (self .n ):
            reward =r_local [:,i :i +1 ]
            y =LOCAL_REWARD_SCALE *reward +self .gamma *target_qs [i ]*(1 -d [:,i :i +1 ])
            y_targets .append (torch .nan_to_num_ (y ,nan =0.0 ,posinf =0.0 ,neginf =0.0 ))

        if actual_station_powers is None :
            raise ValueError ("actual_station_powers must be provided.")
        powers_norm =torch .clamp (actual_station_powers /max_station_power ,-1.0 ,1.0 )

        q_vals =[]
        q_vals2 =[]
        for i in range (self .n ):
            q_val =self .critics [i ](
            s [:,i ,:],a_actual [:,i ,:],actual_station_powers =powers_norm [:,i ]
            )
            q_val2 =self .critics2 [i ](
            s [:,i ,:],a_actual [:,i ,:],actual_station_powers =powers_norm [:,i ]
            )
            q_vals .append (torch .nan_to_num_ (q_val ,nan =0.0 ,posinf =0.0 ,neginf =0.0 ))
            q_vals2 .append (torch .nan_to_num_ (q_val2 ,nan =0.0 ,posinf =0.0 ,neginf =0.0 ))

        self .critic_losses =[0.0 ]*self .n
        self .local_critic_clip_counts =[0 ]*self .n
        self .local_q_twin_gap_values_per_agent =[0.0 ]*self .n
        critic_loss_tensors =[]
        critic1_grad_norm_tensors =[]
        critic2_grad_norm_tensors =[]
        critic1_grad_finite_tensors =[]
        critic2_grad_finite_tensors =[]
        local_q_gap_tensors =[]
        critics1_ready_to_step =[]
        critics2_ready_to_step =[]
        for i in range (self .n ):
            per_sample_loss1 =F .smooth_l1_loss (q_vals [i ],y_targets [i ],beta =SMOOTHL1_BETA ,reduction ='none').squeeze (-1 )
            per_sample_loss2 =F .smooth_l1_loss (q_vals2 [i ],y_targets [i ],beta =SMOOTHL1_BETA ,reduction ='none').squeeze (-1 )
            loss_c1 =per_sample_loss1 .mean ()
            loss_c2 =per_sample_loss2 .mean ()
            loss_c1 =torch .nan_to_num_ (loss_c1 ,nan =0.0 ,posinf =0.0 ,neginf =0.0 )
            loss_c2 =torch .nan_to_num_ (loss_c2 ,nan =0.0 ,posinf =0.0 ,neginf =0.0 )
            loss_pair =loss_c1 +loss_c2
            critic_loss_tensors .append ((0.5 *(loss_c1 +loss_c2 )).detach ())

            with torch .no_grad ():
                local_q_gap_tensors .append ((q_vals [i ].sum (dim =1 )-q_vals2 [i ].sum (dim =1 )).abs ().mean ().detach ())

            self .opt_c [i ].zero_grad ()
            self .opt_c2 [i ].zero_grad ()
            loss_pair .backward ()

            critic1_grad_finite_tensors .append (
            _all_grads_finite (self .critics [i ],loss_c1 )
            )
            critic2_grad_finite_tensors .append (
            _all_grads_finite (self .critics2 [i ],loss_c2 )
            )

            self .clip_bias_gradients (self .critics [i ],max_norm =BIAS_GRAD_CLIP_MAX )
            self .clip_bias_gradients (self .critics2 [i ],max_norm =BIAS_GRAD_CLIP_MAX )
            gn1 =torch .nn .utils .clip_grad_norm_ (
            self .critics [i ].parameters (),max_norm =GRAD_CLIP_MAX
            )
            gn2 =torch .nn .utils .clip_grad_norm_ (
            self .critics2 [i ].parameters (),max_norm =GRAD_CLIP_MAX
            )
            if isinstance (gn1 ,torch .Tensor ):
                gn1_tensor =gn1 .detach ().to (device =device )
            else :
                gn1_tensor =torch .tensor (float (gn1 ),device =device )
            if isinstance (gn2 ,torch .Tensor ):
                gn2_tensor =gn2 .detach ().to (device =device )
            else :
                gn2_tensor =torch .tensor (float (gn2 ),device =device )
            critic1_grad_norm_tensors .append (gn1_tensor )
            critic2_grad_norm_tensors .append (gn2_tensor )

        if self .n >0 :
            # One read, not six. Each `.cpu()` on a device tensor blocks until
            # every kernel queued behind it has finished, so the cost is the
            # pipeline it drains rather than the six values it copies, and this
            # runs once per environment step. The rows carry the same numbers in
            # the same order the six separate reads produced.
            (
            critic_losses ,
            critic1_grad_finite ,
            critic2_grad_finite ,
            critic1_grad_norms ,
            critic2_grad_norms ,
            local_q_gaps ,
            )=torch .stack ([
            torch .stack (critic_loss_tensors ),
            torch .stack (critic1_grad_finite_tensors ).to (dtype =torch .float32 ),
            torch .stack (critic2_grad_finite_tensors ).to (dtype =torch .float32 ),
            torch .stack (critic1_grad_norm_tensors ),
            torch .stack (critic2_grad_norm_tensors ),
            torch .stack (local_q_gap_tensors ),
            ]).cpu ().tolist ()

            for i in range (self .n ):
                loss_val =float (critic_losses [i ])
                finite1 =bool (critic1_grad_finite [i ])
                finite2 =bool (critic2_grad_finite [i ])
                finite =finite1 and finite2 and math .isfinite (loss_val )
                self .critic_losses [i ]=loss_val if finite else 0.0

                grad_norm_before1 =float (critic1_grad_norms [i ])
                grad_norm_before2 =float (critic2_grad_norms [i ])
                grad_norm_before =max (
                grad_norm_before1 if finite1 else 0.0 ,
                grad_norm_before2 if finite2 else 0.0 ,
                )
                grad_norm_after =min (grad_norm_before ,GRAD_CLIP_MAX )

                clipped =(
                (finite1 and grad_norm_before1 >GRAD_CLIP_MAX )
                or (finite2 and grad_norm_before2 >GRAD_CLIP_MAX )
                )
                if clipped :
                    local_critic_clip_count +=1
                    self .local_critic_clip_counts [i ]=1

                if (finite1 or finite2 )and math .isfinite (grad_norm_after ):
                    self .critic_norms_before_clip [i ]=grad_norm_before
                    self .critic_norms [i ]=grad_norm_after
                    if finite1 and math .isfinite (min (grad_norm_before1 ,GRAD_CLIP_MAX )):
                        critics1_ready_to_step .append (i )
                    if finite2 and math .isfinite (min (grad_norm_before2 ,GRAD_CLIP_MAX )):
                        critics2_ready_to_step .append (i )
                else :
                    self .critic_norms_before_clip [i ]=0.0
                    self .critic_norms [i ]=0.0

                gap_val =float (local_q_gaps [i ])
                self .local_q_twin_gap_values_per_agent [i ]=gap_val if math .isfinite (gap_val )else 0.0

            self .last_local_q_twin_gap_values_per_agent =list (self .local_q_twin_gap_values_per_agent )
            self .last_local_q_twin_gap_mean =(
            float (sum (self .last_local_q_twin_gap_values_per_agent )/len (self .last_local_q_twin_gap_values_per_agent ))
            if self .last_local_q_twin_gap_values_per_agent else 0.0
            )

        for i in critics1_ready_to_step :
            self .opt_c [i ].step ()
        for i in critics2_ready_to_step :
            self .opt_c2 [i ].step ()

        return local_critic_clip_count

    def _set_critic_requires_grad_for_actor_update (
    self ,
    requires_grad :bool ,
    include_local :bool =True ,
    include_global :bool =True ,
    ):
        modules =[]
        if include_local :
            modules .extend (self .critics )
            modules .extend (self .critics2 )
        if include_global :
            modules .append (self .global_critic1 )
            modules .append (self .global_critic2 )

        states =[]
        for module in modules :
            for p in module .parameters ():
                states .append ((p ,p .requires_grad ))
                p .requires_grad_ (requires_grad )
        return states

    @staticmethod
    def _restore_requires_grad (states ):
        for p ,requires_grad in states :
            p .requires_grad_ (requires_grad )


    def _update_actors (self ,ctx ):
        """
        Update station actors with an explicit local/global gradient mixture.

        For station i, the local gradient is taken from its local critic. The
        global gradient is taken from the twin global critic, computed once on
        the shared global graph and split by actor parameters. The applied
        gradient is `g_mix = (1 - w_eff) * g_local + w_eff * g_global`; the
        separate norms and cosine are recorded to diagnose objective conflict.
        """
        s =ctx ['s']
        ev_padding_mask =ctx ['ev_padding_mask']
        key_padding_mask =ctx ['key_padding_mask']
        batch_size =ctx ['batch_size']
        skip_local =ctx ['skip_local']
        max_station_power =ctx ['max_station_power']
        w_eff =ctx ['w_eff']
        current_socs_all =ctx ['current_socs']
        max_power_kw_all =ctx ['max_power_kw']
        soc_step_per_kw_all =ctx ['soc_step_per_kw']
        kw_per_soc_step_all =ctx ['kw_per_soc_step']
        inv_max_power_kw_all =ctx ['inv_max_power_kw']
        actor_clip_count =0
        critic_grad_states =self ._set_critic_requires_grad_for_actor_update (
        False ,include_local =not skip_local ,include_global =True
        )

        current_actions =[self .actors [i ](s [:,i ,:])for i in range (self .n )]
        cur_a_all_new =torch .stack (current_actions ,dim =1 )
        actions_all =cur_a_all_new *max_power_kw_all
        actions_all =actions_all .masked_fill (ev_padding_mask ,0.0 )

        clamped_actions_all ,recomputed_actual_station_powers =self ._apply_soc_constraint (
        actions_all ,current_socs_all ,ev_padding_mask ,use_ste =True ,
        max_power_kw =max_power_kw_all ,
        soc_step_per_kw =soc_step_per_kw_all ,kw_per_soc_step =kw_per_soc_step_all ,
        )

        s_global_actor =self ._convert_to_global_critic_obs (
        s ,recomputed_actual_station_powers )
        recomputed_actual_station_powers_normalized =torch .clamp (
        recomputed_actual_station_powers /max_station_power ,-1.0 ,1.0
        )

        a_all_kw_masked =clamped_actions_all .masked_fill (ev_padding_mask ,0.0 )
        a_all_kw_normalized =self ._normalize_power_by_limit (
        a_all_kw_masked ,max_power_kw_all ,inv_max_power_kw_all
        )
        q1_shared ,_ =self .global_critic1 (
        s_global_actor ,a_all_kw_normalized ,key_padding_mask ,
        actual_station_powers =recomputed_actual_station_powers_normalized ,
        )
        q2_shared ,_ =self .global_critic2 (
        s_global_actor ,a_all_kw_normalized ,key_padding_mask ,
        actual_station_powers =recomputed_actual_station_powers_normalized ,
        )
        q_global_shared =torch .minimum (q1_shared ,q2_shared )
        q_global_shared =torch .nan_to_num_ (q_global_shared ,nan =0.0 ,posinf =0.0 ,neginf =0.0 )
        q_g_mean_shared =q_global_shared .mean ()

        self .actor_losses =[0.0 ]*self .n
        self .actor_clip_counts =[0 ]*self .n
        self .actor_source_local_norms_before_clip =[0.0 ]*self .n
        self .actor_source_global_norms_before_clip =[0.0 ]*self .n
        self .actor_source_global_ratio =[0.0 ]*self .n
        self .actor_source_cos =[0.0 ]*self .n
        self .actor_source_cos_valid =[0 ]*self .n
        actors_ready_to_step =[]
        actor_param_groups =[list (self .actors [i ].parameters ())for i in range (self .n )]
        global_grads_by_actor =[[None ]*len (params )for params in actor_param_groups ]
        all_actor_params =[p for params in actor_param_groups for p in params ]
        global_grads_flat =torch .autograd .grad (
        -q_g_mean_shared ,all_actor_params ,retain_graph =True ,allow_unused =True
        )
        offset =0
        for i ,params in enumerate (actor_param_groups ):
            width =len (params )
            global_grads_by_actor [i ]=list (global_grads_flat [offset :offset +width ])
            offset +=width

        source_local_norm_tensors =[]
        source_global_norm_tensors =[]
        source_global_ratio_tensors =[]
        source_cos_tensors =[]
        source_cos_valid_tensors =[]
        grad_finite_tensors =[]
        grad_norm_before_tensors =[]
        actor_loss_tensors =[]

        for i in range (self .n ):
            params =actor_param_groups [i ]

            agent_actual_power =recomputed_actual_station_powers [:,i ]
            agent_actual_power_normalized =torch .clamp (
            agent_actual_power /max_station_power ,-1.0 ,1.0
            )

            if skip_local :
                q_local =torch .zeros ((batch_size ,1 ),device =device )
            else :
                s_flat_i =s [:,i ,:]
                agent_a =torch .clamp (
                clamped_actions_all [:,i ,:]*inv_max_power_kw_all [:,i ,:],-1.0 ,1.0 )
                q_local =self .critics [i ](
                s_flat_i ,agent_a ,actual_station_powers =agent_actual_power_normalized )

            q_local =torch .nan_to_num_ (q_local ,nan =0.0 ,posinf =0.0 ,neginf =0.0 )

            q_l_mean =q_local .mean ()
            q_g_mean =q_g_mean_shared

            self ._ep_q_raw_local [i ].append (q_l_mean .detach ())
            if i ==0 :
                self ._ep_q_raw_global .append (q_g_mean .detach ())

            self .opt_a [i ].zero_grad ()
            local_grads =[None ]*len (params )
            global_grads =global_grads_by_actor [i ]

            if not skip_local :
                local_grads =list (torch .autograd .grad (
                -q_l_mean ,params ,retain_graph =True ,allow_unused =True
                ))

            sq_l =torch .zeros ((),device =device )
            sq_g =torch .zeros ((),device =device )
            dot_lg =torch .zeros ((),device =device )
            has_l ,has_g =False ,False
            for g_l ,g_g in zip (local_grads ,global_grads ):
                if g_l is not None :
                    sq_l =sq_l +(g_l .detach ()*g_l .detach ()).sum ()
                    has_l =True
                if g_g is not None :
                    sq_g =sq_g +(g_g .detach ()*g_g .detach ()).sum ()
                    has_g =True
                if g_l is not None and g_g is not None :
                    dot_lg =dot_lg +(g_l .detach ()*g_g .detach ()).sum ()

            norm_l_tensor =(
            torch .sqrt (torch .clamp (sq_l ,min =0.0 ))
            if has_l else torch .zeros ((),device =device )
            )
            norm_g_tensor =(
            torch .sqrt (torch .clamp (sq_g ,min =0.0 ))
            if has_g else torch .zeros ((),device =device )
            )
            ratio_g_tensor =norm_g_tensor /torch .clamp (norm_l_tensor +norm_g_tensor ,min =1e-12 )

            cos_lg_tensor =torch .zeros ((),device =device )
            cos_valid_tensor =torch .zeros ((),device =device ,dtype =torch .bool )
            if has_l and has_g :
                cos_tensor =dot_lg /(
                torch .sqrt (torch .clamp (sq_l ,min =1e-24 ))*
                torch .sqrt (torch .clamp (sq_g ,min =1e-24 ))
                )
                cos_valid_tensor =(norm_l_tensor >1e-12 )&(norm_g_tensor >1e-12 )
                cos_lg_tensor =torch .where (
                cos_valid_tensor ,
                torch .clamp (cos_tensor ,-1.0 ,1.0 ),
                torch .zeros_like (cos_tensor ),
                )

            source_local_norm_tensors .append (norm_l_tensor .detach ())
            source_global_norm_tensors .append (norm_g_tensor .detach ())
            source_global_ratio_tensors .append (ratio_g_tensor .detach ())
            source_cos_tensors .append (cos_lg_tensor .detach ())
            source_cos_valid_tensors .append (cos_valid_tensor .detach ())

            for idx ,p in enumerate (params ):
                g_l =local_grads [idx ]
                g_g =global_grads [idx ]
                if g_l is not None and g_g is not None :
                    g_mix =(1.0 -w_eff )*g_l +w_eff *g_g
                elif g_l is not None :
                    g_mix =g_l
                elif g_g is not None :
                    g_mix =g_g
                else :
                    g_mix =None
                p .grad =g_mix .clone ()if g_mix is not None else None

            # The actor's own check starts from a true constant rather than from
            # a loss, because the mixed gradient it just wrote is what is being
            # judged here and the two source losses are recorded separately.
            grad_finite_tensors .append (
            _all_grads_finite (self .actors [i ],torch .ones ((),device =device ))
            )

            self .clip_bias_gradients (self .actors [i ],max_norm =BIAS_GRAD_CLIP_MAX )
            grad_norm_before =torch .nn .utils .clip_grad_norm_ (
            self .actors [i ].parameters (),max_norm =GRAD_CLIP_MAX
            )
            if isinstance (grad_norm_before ,torch .Tensor ):
                grad_norm_before_tensor =grad_norm_before .detach ().to (device =device )
            else :
                grad_norm_before_tensor =torch .tensor (float (grad_norm_before ),device =device )
            grad_norm_before_tensors .append (grad_norm_before_tensor )

            actor_loss_tensors .append (-(
            (1.0 -w_eff )*q_l_mean .detach ()+w_eff *q_g_mean .detach ()
            ))

        if self .n >0 :
            # One read for the eight diagnostics, for the reason given over the
            # local critics' read.
            (
            source_local_norms ,
            source_global_norms ,
            source_global_ratios ,
            source_cos ,
            source_cos_valid ,
            grad_finite ,
            grad_norms_before ,
            actor_losses ,
            )=torch .stack ([
            torch .stack (source_local_norm_tensors ),
            torch .stack (source_global_norm_tensors ),
            torch .stack (source_global_ratio_tensors ),
            torch .stack (source_cos_tensors ),
            torch .stack (source_cos_valid_tensors ).to (dtype =torch .float32 ),
            torch .stack (grad_finite_tensors ).to (dtype =torch .float32 ),
            torch .stack (grad_norm_before_tensors ),
            torch .stack (actor_loss_tensors ),
            ]).cpu ().tolist ()

            for i in range (self .n ):
                self .actor_source_local_norms_before_clip [i ]=float (source_local_norms [i ])
                self .actor_source_global_norms_before_clip [i ]=float (source_global_norms [i ])
                self .actor_source_global_ratio [i ]=float (source_global_ratios [i ])
                self .actor_source_cos [i ]=float (source_cos [i ])
                self .actor_source_cos_valid [i ]=int (bool (source_cos_valid [i ]))
                self .actor_losses [i ]=float (actor_losses [i ])

                grad_norm_before =float (grad_norms_before [i ])
                grad_norm_after =min (grad_norm_before ,GRAD_CLIP_MAX )

                if bool (grad_finite [i ])and grad_norm_before >GRAD_CLIP_MAX :
                    actor_clip_count +=1
                    if i <len (self .actor_clip_counts ):
                        self .actor_clip_counts [i ]=1

                if bool (grad_finite [i ])and math .isfinite (grad_norm_after ):
                    self .actor_norms [i ]=grad_norm_after
                    self .actor_norms_before_clip [i ]=grad_norm_before
                    actors_ready_to_step .append (i )
                else :
                    self .actor_norms [i ]=0.0

        for i in actors_ready_to_step :
            self .opt_a [i ].step ()

        self ._restore_requires_grad (critic_grad_states )
        return actor_clip_count


    def _polyak_update_targets (self ):
        """
        Soft-update target actors, local critics, and global critics.

        Local networks use `self.tau`; the global critic uses
        `self.tau_global` so the shorter-horizon global value can track its
        online critic at an independently chosen rate.
        """
        src_local ,tgt_local =[],[]
        for i in range (self .n ):
            for p ,tp in zip (self .actors [i ].parameters (),self .t_actors [i ].parameters ()):
                src_local .append (p .data );tgt_local .append (tp .data )
            for p ,tp in zip (self .critics [i ].parameters (),self .t_critics [i ].parameters ()):
                src_local .append (p .data );tgt_local .append (tp .data )
            for p ,tp in zip (self .critics2 [i ].parameters (),self .t_critics2 [i ].parameters ()):
                src_local .append (p .data );tgt_local .append (tp .data )
        torch ._foreach_lerp_ (tgt_local ,src_local ,self .tau )

        src_global ,tgt_global =[],[]
        for p ,tp in zip (self .global_critic1 .parameters (),self .t_global_critic1 .parameters ()):
            src_global .append (p .data );tgt_global .append (tp .data )
        for p ,tp in zip (self .global_critic2 .parameters (),self .t_global_critic2 .parameters ()):
            src_global .append (p .data );tgt_global .append (tp .data )
        torch ._foreach_lerp_ (tgt_global ,src_global ,self .tau_global )


    def _aggregate_update_logs (self ,local_critic_clip_count ,actor_clip_count ):
        """Collect the latest losses, Q-values, gradient norms, and clip counts."""
        if self .critic_losses :
            self .last_critic_loss =sum (self .critic_losses )/len (self .critic_losses )
        if self .actor_losses :
            self .last_actor_loss =sum (self .actor_losses )/len (self .actor_losses )

        if self .critic_norms :
            avg_cn =sum (self .critic_norms )/len (self .critic_norms )
            self .last_local_critic_grad_norm =avg_cn if math .isfinite (avg_cn )else 0.0
        else :
            self .last_local_critic_grad_norm =0.0
        self .last_local_critic_clip_count =local_critic_clip_count

        if self .actor_norms :
            avg_an =sum (self .actor_norms )/len (self .actor_norms )
            self .last_actor_grad_norm =avg_an if math .isfinite (avg_an )else 0.0
        else :
            self .last_actor_grad_norm =0.0
        self .last_actor_clip_count =actor_clip_count

        src_local =self .actor_source_local_norms_before_clip
        self .last_actor_source_local_grad_norm_before_clip =(
        float (sum (src_local )/len (src_local ))if src_local else 0.0
        )
        src_global_norms =self .actor_source_global_norms_before_clip
        self .last_actor_source_global_grad_norm_before_clip =(
        float (sum (src_global_norms )/len (src_global_norms ))if src_global_norms else 0.0
        )
        self .last_actor_source_global_ratio =(
        float (sum (self .actor_source_global_ratio )/len (self .actor_source_global_ratio ))
        if self .actor_source_global_ratio else 0.0
        )

        cos_valid_count =int (sum (self .actor_source_cos_valid ))if self .actor_source_cos_valid else 0
        if cos_valid_count >0 :
            cos_sum =sum (
            float (v )for v ,flag in zip (self .actor_source_cos ,self .actor_source_cos_valid )
            if flag
            )
            self .last_actor_source_cos =cos_sum /float (cos_valid_count )
        else :
            self .last_actor_source_cos =0.0
        self .last_actor_source_cos_valid_fraction =(
        float (cos_valid_count )/float (len (self .actor_source_cos_valid ))
        if self .actor_source_cos_valid else 0.0
        )

        # The per-station values and the joint one are read together: two reads
        # cost two pipeline drains, and the global value is one more entry on
        # the end of the same vector.
        q_tensors =[
        self ._ep_q_raw_local [i ][-1 ]if self ._ep_q_raw_local [i ]else torch .zeros ((),device =device )
        for i in range (self .n )
        ]
        has_global =bool (self ._ep_q_raw_global )
        if has_global :
            q_tensors .append (self ._ep_q_raw_global [-1 ])
        if q_tensors :
            q_values =torch .stack (q_tensors ).cpu ().tolist ()
        else :
            q_values =[]
        self .last_global_q_value =float (q_values .pop ())if has_global else 0.0
        self .last_local_q_values_per_agent =[float (v )for v in q_values ]


    def _update_global_critic (self ,ctx ):
        """
        Update the global critic with a one-step target.

        The global critic observes all station EV states and all station
        actions simultaneously. Target policy smoothing is applied to target
        actor outputs, then actions are clipped to physical SoC limits before
        the target Q is computed. The target uses `GAMMA_GLOBAL`, which is
        shorter than the local discount because dispatch tracking is an
        immediate aggregate-power objective.
        """
        s =ctx ['s']
        a_actual =ctx ['a_actual']
        r_global_n =ctx ['r_global_n']
        s2_n =ctx ['s2_n']
        d_n =ctx ['d_n']
        actual_station_powers =ctx ['actual_station_powers']
        ev_padding_mask =ctx ['ev_padding_mask']
        key_padding_mask =ctx ['key_padding_mask']
        max_station_power =ctx ['max_station_power']

        # The replay sampler is called with n_step=1, so `s2_n` is the ordinary
        # next state.
        s2_for_target =s2_n

        # Build masks from the target next-state EV presence flags so target
        # actions are ignored for empty EV slots and empty stations.
        ev_block_for_target =s2_for_target [:,:,:self .max_evs *EV_FEAT_DIM ].reshape (
        s .size (0 ),self .n ,self .max_evs ,EV_FEAT_DIM )
        current_socs_s2_g ,capacity_kwh_s2_g ,max_power_kw_s2_g =self ._extract_ev_physics (
        ev_block_for_target
        )
        soc_step_per_kw_s2_g ,kw_per_soc_step_s2_g ,inv_max_power_kw_s2_g =self ._ev_physics_factors (
        capacity_kwh_s2_g ,max_power_kw_s2_g
        )
        presence_mask_for_target =(ev_block_for_target [...,0 ]<=0.5 )
        ev_padding_mask_for_target =presence_mask_for_target
        key_padding_mask_for_target =presence_mask_for_target .all (dim =2 )

        with torch .no_grad ():
            next_a_all =torch .stack (
            [self .t_actors [i ](s2_for_target [:,i ,:])for i in range (self .n )],dim =1
            )
            g_noise =torch .randn_like (next_a_all )*self .td3_sigma
            g_noise =torch .clamp (g_noise ,-self .td3_clip ,self .td3_clip )
            next_a_all =torch .clamp (next_a_all +g_noise ,-1.0 ,1.0 )

            next_a_kw =next_a_all *max_power_kw_s2_g
            clamped_actions_next ,next_station_powers =self ._apply_soc_constraint (
            next_a_kw ,current_socs_s2_g ,ev_padding_mask_for_target ,
            max_power_kw =max_power_kw_s2_g ,
            soc_step_per_kw =soc_step_per_kw_s2_g ,kw_per_soc_step =kw_per_soc_step_s2_g ,
            )
            s2_global =self ._convert_to_global_critic_obs (s2_for_target ,next_station_powers )

            next_a_kw_masked =clamped_actions_next .masked_fill (ev_padding_mask_for_target ,0.0 )
            next_a_kw_normalized =self ._normalize_power_by_limit (
            next_a_kw_masked ,max_power_kw_s2_g ,inv_max_power_kw_s2_g
            )
            next_station_powers_normalized =torch .clamp (
            next_station_powers /max_station_power ,-1.0 ,1.0
            )

            tq1_s ,_ =self .t_global_critic1 (
            s2_global ,next_a_kw_normalized ,key_padding_mask_for_target ,
            actual_station_powers =next_station_powers_normalized ,
            )
            tq2_s ,_ =self .t_global_critic2 (
            s2_global ,next_a_kw_normalized ,key_padding_mask_for_target ,
            actual_station_powers =next_station_powers_normalized ,
            )
            target_q_global =torch .min (tq1_s ,tq2_s )

            # One-step global TD target in the n-step sampler format:
            # y = c_global * (r_global - b_global) + GAMMA_GLOBAL * Q' * (1 - done_any).
            # `b_global` removes the action-independent success baseline from
            # the critic target. With one-step global targets this shifts Q_g
            # by a constant before scaling, preserving the optimal action while
            # reducing the centralized critic's raw gradient scale.
            # Q' is min(Q1', Q2') of the twin target global critics.
            centered_r_global_n =r_global_n -self .global_reward_baseline
            scaled_r_global_n =self .global_reward_scale *centered_r_global_n
            done_mask_global =d_n .max (dim =1 ,keepdim =True )[0 ]
            y_global =scaled_r_global_n +GAMMA_GLOBAL *target_q_global *(1 -done_mask_global )
            y_global =torch .nan_to_num_ (y_global ,nan =0.0 ,posinf =0.0 ,neginf =0.0 )

        s_global =self ._convert_to_global_critic_obs (s ,actual_station_powers )
        a_actual_global =a_actual .masked_fill (ev_padding_mask ,0.0 )
        actual_station_powers_normalized =torch .clamp (
        actual_station_powers /max_station_power ,-1.0 ,1.0
        )

        q1_s ,_ =self .global_critic1 (
        s_global ,a_actual_global ,key_padding_mask ,
        actual_station_powers =actual_station_powers_normalized ,
        )
        q1_s =torch .nan_to_num_ (q1_s ,nan =0.0 ,posinf =0.0 ,neginf =0.0 )
        q2_s ,_ =self .global_critic2 (
        s_global ,a_actual_global ,key_padding_mask ,
        actual_station_powers =actual_station_powers_normalized ,
        )
        q2_s =torch .nan_to_num_ (q2_s ,nan =0.0 ,posinf =0.0 ,neginf =0.0 )

        per_sample_g1 =F .smooth_l1_loss (q1_s ,y_global ,beta =SMOOTHL1_BETA ,reduction ='none').squeeze (-1 )
        per_sample_g2 =F .smooth_l1_loss (q2_s ,y_global ,beta =SMOOTHL1_BETA ,reduction ='none').squeeze (-1 )
        loss_g1 =per_sample_g1 .mean ()
        loss_g2 =per_sample_g2 .mean ()
        loss_g1 =torch .nan_to_num_ (loss_g1 ,nan =0.0 ,posinf =0.0 ,neginf =0.0 )
        loss_g2 =torch .nan_to_num_ (loss_g2 ,nan =0.0 ,posinf =0.0 ,neginf =0.0 )

        with torch .no_grad ():
            global_td_abs =((q1_s -y_global ).abs ().squeeze (-1 ))
            global_diag_t =torch .stack ([
            torch .as_tensor (self .global_reward_scale ,dtype =q1_s .dtype ,device =q1_s .device ),
            torch .as_tensor (self .global_reward_baseline ,dtype =q1_s .dtype ,device =q1_s .device ),
            r_global_n .detach ().abs ().mean (),
            centered_r_global_n .detach ().abs ().mean (),
            scaled_r_global_n .detach ().abs ().mean (),
            y_global .detach ().abs ().mean (),
            global_td_abs .mean (),
            q1_s .detach ().abs ().mean (),
            target_q_global .detach ().abs ().mean (),
            ]).cpu ().tolist ()
        (
        self .last_global_reward_scale ,
        self .last_global_reward_baseline ,
        self .last_global_reward_raw_abs_mean ,
        self .last_global_reward_centered_abs_mean ,
        self .last_global_reward_term_abs_mean ,
        self .last_global_td_target_abs_mean ,
        self .last_global_td_error_abs_mean ,
        self .last_global_current_q_abs_mean ,
        self .last_global_target_q_abs_mean ,
        )=[float (v )if math .isfinite (float (v ))else 0.0 for v in global_diag_t ]

        loss_g_total =loss_g1 +loss_g2
        self .opt_global_c1 .zero_grad ()
        self .opt_global_c2 .zero_grad ()
        loss_g_total .backward ()

        finite_g1 =torch .isfinite (loss_g1 .detach ())
        for p in self .global_critic1 .parameters ():
            if p .grad is not None :
                finite_g1 =finite_g1 &torch .isfinite (p .grad ).all ()
        finite_g2 =torch .isfinite (loss_g2 .detach ())
        for p in self .global_critic2 .parameters ():
            if p .grad is not None :
                finite_g2 =finite_g2 &torch .isfinite (p .grad ).all ()

        self .clip_bias_gradients (self .global_critic1 ,max_norm =BIAS_GRAD_CLIP_MAX )
        gn_b =torch .nn .utils .clip_grad_norm_ (
        self .global_critic1 .parameters (),max_norm =GRAD_CLIP_MAX_GLOBAL
        )
        self .clip_bias_gradients (self .global_critic2 ,max_norm =BIAS_GRAD_CLIP_MAX )
        gn_b2 =torch .nn .utils .clip_grad_norm_ (
        self .global_critic2 .parameters (),max_norm =GRAD_CLIP_MAX_GLOBAL
        )

        global_diag =torch .stack ([
        loss_g1 .detach (),
        gn_b .detach ()if isinstance (gn_b ,torch .Tensor )else torch .tensor (float (gn_b ),device =device ),
        gn_b2 .detach ()if isinstance (gn_b2 ,torch .Tensor )else torch .tensor (float (gn_b2 ),device =device ),
        finite_g1 .detach ().float (),
        finite_g2 .detach ().float (),
        ]).cpu ().tolist ()
        loss_g1_value ,gn_b_value ,gn_b2_value ,finite_g1_value ,finite_g2_value =global_diag

        gn_a =min (float (gn_b_value ),GRAD_CLIP_MAX_GLOBAL )
        self .last_global_critic_clip_count =1 if float (gn_b_value )>GRAD_CLIP_MAX_GLOBAL else 0
        self .last_global_critic_grad_norm =gn_a if math .isfinite (gn_a )else 0.0
        self .last_global_critic_grad_norm_before_clip =(
        float (gn_b_value )if math .isfinite (float (gn_b_value ))else 0.0
        )
        self .last_global_critic_loss =(
        float (loss_g1_value )if math .isfinite (float (loss_g1_value ))else 0.0
        )

        gn_a2 =min (float (gn_b2_value ),GRAD_CLIP_MAX_GLOBAL )
        if bool (finite_g1_value )and math .isfinite (gn_a ):
            self .opt_global_c1 .step ()
        if bool (finite_g2_value )and math .isfinite (gn_a2 ):
            self .opt_global_c2 .step ()


    def update (self):
        """
        Run one gradient update if replay memory has passed warmup.

        Training is disabled in test mode. After warmup, one batch updates local
        critics, the global critic, and, every `POLICY_DELAY` calls, the
        actors plus target networks. Global replay sampling is fixed to one-step
        targets and uses `GAMMA_GLOBAL`.
        """
        if self .test_mode :
            self ._zero_update_logs ()
            return

        if self .buf .size <self .warmup_steps :
            self ._zero_update_logs ()
            return

        self .update_step +=1
        actor_update_due =(self .update_step %self .policy_delay ==0 )

        (s ,s2 ,a ,r_local ,_r_global ,d ,actual_station_powers ,actual_ev_power_kw ,
        r_global_n ,s2_n ,d_n ,_n_eff )=self .buf .sample_with_nstep_global (
        self .batch ,1 ,GAMMA_GLOBAL )

        ctx =self ._build_update_ctx (s ,s2 ,a ,r_local ,d ,
        actual_station_powers ,actual_ev_power_kw )
        ctx ['r_global_n']=r_global_n
        ctx ['s2_n']=s2_n
        ctx ['d_n']=d_n

        local_critic_clip_count =self ._update_local_critics (ctx )
        self ._update_global_critic (ctx )
        self .actor_losses =[0.0 ]*self .n
        actor_clip_count =0
        if actor_update_due :
            try :
                actor_clip_count =self ._update_actors (ctx )
            finally :
                self ._set_critic_requires_grad_for_actor_update (True ,True ,True )
            self ._polyak_update_targets ()

        self ._aggregate_update_logs (local_critic_clip_count ,actor_clip_count )


    def episode_start (self ):
        """Advance exploration schedules and reset per-episode diagnostics."""
        if not self .test_mode :
            self .current_episode +=1

        self ._ep_q_raw_global =[]
        self ._ep_q_raw_local =[[]for _ in range (self .n )]

        if self .test_mode :
            self .epsilon =0.0
            self .ou_noise_scale =0.0
        else :
            ep_in_phase =self .current_episode
            self .epsilon =linear_epsilon_decay (
            ep_in_phase ,
            self .epsilon_start_episode ,self .epsilon_end_episode ,
            self .epsilon_initial ,self .epsilon_final ,
            )

            s0n ,s1n =self .ou_noise_start_episode ,self .ou_noise_end_episode
            n0 ,n1 =self .ou_noise_scale_initial ,self .ou_noise_scale_final
            if ep_in_phase <s0n :
                self .ou_noise_scale =0.0
            elif ep_in_phase >=s1n :
                self .ou_noise_scale =n1
            else :
                rn =(ep_in_phase -s0n )/max (1 ,(s1n -s0n ))
                self .ou_noise_scale =n0 +(n1 -n0 )*rn
            self .ou_noise_scale =max (self .ou_noise_scale_final ,self .ou_noise_scale )

        self .ou_noise .reset ()

    def episode_end (self ):
        """Finish one environment episode."""
        if self .test_mode :
            return

    def set_test_mode (self ,mode :bool ):
        """Switch all online and target networks between train and eval mode."""
        self .test_mode =mode
        self .training =not mode

        if mode :
            for actor in self .actors :
                actor .eval ()
            for critic in self .critics :
                critic .eval ()
            for critic in self .critics2 :
                critic .eval ()
            for t_actor in self .t_actors :
                t_actor .eval ()
            for t_critic in self .t_critics :
                t_critic .eval ()
            for t_critic in self .t_critics2 :
                t_critic .eval ()
            self .global_critic1 .eval ()
            self .t_global_critic1 .eval ()
            self .global_critic2 .eval ()
            self .t_global_critic2 .eval ()
        else :
            for actor in self .actors :
                actor .train ()
            for critic in self .critics :
                critic .train ()
            for critic in self .critics2 :
                critic .train ()
            for t_actor in self .t_actors :
                t_actor .train ()
            for t_critic in self .t_critics :
                t_critic .train ()
            for t_critic in self .t_critics2 :
                t_critic .train ()
            self .global_critic1 .train ()
            self .t_global_critic1 .train ()
            self .global_critic2 .train ()
            self .t_global_critic2 .train ()

    def cache_experience (self ,s ,s2 ,a ,r_local ,r_global ,d ,
    actual_station_powers =None ,actual_ev_power_kw =None ):
        """Store an executable environment transition in replay memory."""
        if self .test_mode :
            return
        self .buf .cache (s ,s2 ,a ,r_local ,r_global ,d ,actual_station_powers ,actual_ev_power_kw )

    def save_actors (self ,path ,episode ):
        """Save each station actor as a separate checkpoint file."""
        os .makedirs (path ,exist_ok =True )
        for i in range (self .n ):
            torch .save (self .actors [i ].state_dict (),os .path .join (path ,f"actor_{i}_ep{episode}.pth"))

    def load_actors (self ,path ,episode ,map_location =None ):
        """Load station actor checkpoints and synchronize target actors."""
        for i in range (self .n ):
            actor_path =os .path .join (path ,f"actor_{i}_ep{episode}.pth")
            try :
                sd =torch .load (
                actor_path ,
                map_location =map_location if map_location is not None else device ,
                weights_only =True ,
                )
            except TypeError :
                sd =torch .load (
                actor_path ,
                map_location =map_location if map_location is not None else device ,
                )
            self .actors [i ].load_state_dict (sd )
            if i <len (self .t_actors ):
                self .t_actors [i ].load_state_dict (self .actors [i ].state_dict ())

    def _training_resume_compatibility (self ):
        """Architecture/runtime fields that must match an exact resume."""
        return {
        "agent_type":type (self ).__name__ ,
        "s_dim":int (self .s_dim ),
        "a_dim":int (self .a_dim ),
        "n_agents":int (self .n ),
        "max_ev_per_station":int (self .max_ev_per_station ),
        "batch":int (self .batch ),
        "gamma":float (self .gamma ),
        "tau":float (self .tau ),
        "tau_global":float (self .tau_global ),
        "policy_delay":int (self .policy_delay ),
        "replay_capacity":int (self .buf .buf_size ),
        }

    def training_resume_state_dict (self ):
        """Return complete learner state for research-grade pretrain resume."""
        models ={
        "actors":[m .state_dict ()for m in self .actors ],
        "target_actors":[m .state_dict ()for m in self .t_actors ],
        "critics":[m .state_dict ()for m in self .critics ],
        "critics2":[m .state_dict ()for m in self .critics2 ],
        "target_critics":[m .state_dict ()for m in self .t_critics ],
        "target_critics2":[m .state_dict ()for m in self .t_critics2 ],
        "global_critic1":self .global_critic1 .state_dict (),
        "target_global_critic1":self .t_global_critic1 .state_dict (),
        "global_critic2":self .global_critic2 .state_dict (),
        "target_global_critic2":self .t_global_critic2 .state_dict (),
        }
        optimizers ={
        "actors":[o .state_dict ()for o in self .opt_a ],
        "critics":[o .state_dict ()for o in self .opt_c ],
        "critics2":[o .state_dict ()for o in self .opt_c2 ],
        "global_critic1":self .opt_global_c1 .state_dict (),
        "global_critic2":self .opt_global_c2 .state_dict (),
        }
        return {
        "format_version":1 ,
        "compatibility":self ._training_resume_compatibility (),
        "models":models ,
        "optimizers":optimizers ,
        "scalars":{
        "current_episode":int (self .current_episode ),
        "epsilon":float (self .epsilon ),
        "ou_noise_scale":float (self .ou_noise_scale ),
        "update_step":int (self .update_step ),
        "warmup_steps":int (self .warmup_steps ),
        "test_mode":bool (self .test_mode ),
        },
        "replay":self .buf .training_resume_state_dict (),
        }

    @staticmethod
    def _load_module_list (modules ,states ,label ):
        if len (modules )!=len (states ):
            raise ValueError (
            f"{label} count mismatch: saved={len(states)} current={len(modules)}"
            )
        for module ,state in zip (modules ,states ):
            module .load_state_dict (state )

    @staticmethod
    def _load_optimizer_list (optimizers ,states ,label ):
        if len (optimizers )!=len (states ):
            raise ValueError (
            f"{label} optimizer count mismatch: saved={len(states)} current={len(optimizers)}"
            )
        for optimizer ,state in zip (optimizers ,states ):
            optimizer .load_state_dict (state )

    def load_training_resume_state_dict (self ,state ):
        """Restore an exact state and reject architecture/config drift."""
        if int (state .get ("format_version",-1 ))!=1 :
            raise ValueError (f"unsupported agent resume format: {state.get('format_version')!r}")
        saved_compat =state .get ("compatibility")
        current_compat =self ._training_resume_compatibility ()
        if saved_compat !=current_compat :
            keys =sorted (set (saved_compat or {})|set (current_compat ))
            detail =", ".join (
            f"{key}: saved={(saved_compat or {}).get(key)!r} current={current_compat.get(key)!r}"
            for key in keys if (saved_compat or {}).get (key )!=current_compat .get (key )
            )
            raise ValueError (f"agent resume compatibility mismatch: {detail}")

        models =state ["models"]
        self ._load_module_list (self .actors ,models ["actors"],"actor")
        self ._load_module_list (self .t_actors ,models ["target_actors"],"target actor")
        self ._load_module_list (self .critics ,models ["critics"],"critic")
        self ._load_module_list (self .critics2 ,models ["critics2"],"critic2")
        self ._load_module_list (self .t_critics ,models ["target_critics"],"target critic")
        self ._load_module_list (self .t_critics2 ,models ["target_critics2"],"target critic2")
        self .global_critic1 .load_state_dict (models ["global_critic1"])
        self .t_global_critic1 .load_state_dict (models ["target_global_critic1"])
        self .global_critic2 .load_state_dict (models ["global_critic2"])
        self .t_global_critic2 .load_state_dict (models ["target_global_critic2"])

        optimizers =state ["optimizers"]
        self ._load_optimizer_list (self .opt_a ,optimizers ["actors"],"actor")
        self ._load_optimizer_list (self .opt_c ,optimizers ["critics"],"critic")
        self ._load_optimizer_list (self .opt_c2 ,optimizers ["critics2"],"critic2")
        self .opt_global_c1 .load_state_dict (optimizers ["global_critic1"])
        self .opt_global_c2 .load_state_dict (optimizers ["global_critic2"])

        scalars =state ["scalars"]
        self .current_episode =int (scalars ["current_episode"])
        self .epsilon =float (scalars ["epsilon"])
        self .ou_noise_scale =float (scalars ["ou_noise_scale"])
        self .update_step =int (scalars ["update_step"])
        self .warmup_steps =int (scalars ["warmup_steps"])
        self .buf .load_training_resume_state_dict (state ["replay"])

        self .set_test_mode (False )
        for module in (
        list (self .actors )+list (self .t_actors )+
        list (self .critics )+list (self .critics2 )+
        list (self .t_critics )+list (self .t_critics2 )+
        [self .global_critic1 ,self .t_global_critic1 ,self .global_critic2 ,self .t_global_critic2 ]
        ):
            module .train ()
        return {
        "current_episode":int (self .current_episode ),
        "update_step":int (self .update_step ),
        "replay_size":int (self .buf .size ),
        "replay_ptr":int (self .buf .ptr ),
        }
