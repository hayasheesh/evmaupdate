"""
noise.py - exploration noise and epsilon-greedy helpers.

This module provides:
- `OUNoise`: Ornstein-Uhlenbeck noise for continuous-control exploration
- `GaussianNoise`: independent Gaussian action-space exploration
- `linear_epsilon_decay`: linear epsilon schedule
- `sample_epsilon_random_action`: uniform random action sampling
"""
import torch

from Config import (
DEVICE ,OU_THETA ,OU_SIGMA ,OU_DT ,OU_INIT_X ,OU_CLIP ,
)


class OUNoise :
    """
    Ornstein-Uhlenbeck exploration noise.

    A separate OU process is maintained for each agent and EV slot in parallel
    on the configured device. This is useful for temporally correlated
    exploration in continuous control.

    Update rule:
        dx = -theta * (x - 0) * dt + sigma * sqrt(dt) * N(0, 1)
        x <- x + dx
    """
    def __init__ (self ,n_agents :int ,max_evs_per_station :int ,
    theta :float =OU_THETA ,sigma :float =OU_SIGMA ,
    dt :float =OU_DT ,x0 :float =OU_INIT_X ):
        self .n_agents =int (n_agents )
        self .max_evs =int (max_evs_per_station )
        self .theta =float (theta )
        self .sigma =float (sigma )
        self .dt =float (dt )
        self .x0 =float (x0 )
        self .device =DEVICE
        self .state =torch .full ((self .n_agents ,self .max_evs ),float (self .x0 ),
        dtype =torch .float32 ,device =self .device )

    def reset (self ):
        """Reset the OU state to the initial value at episode start."""
        self .state .fill_ (float (self .x0 ))

    @torch .no_grad ()
    def sample (self ,active_evs_per_agent ):
        """
        Advance the OU process by one step and return noise for active EV slots.

        `active_evs_per_agent` is a list or 1D tensor of length `n_agents`.
        Inactive slots are forced to zero. If `OU_CLIP` is enabled, the result
        is clamped to `[-OU_CLIP, OU_CLIP]`.
        """
        if not torch .is_tensor (active_evs_per_agent ):
            active =torch .as_tensor (active_evs_per_agent ,device =self .device )
        else :
            active =active_evs_per_agent .to (self .device )

        noise =torch .randn_like (self .state )
        dx =self .theta *(0.0 -self .state )*self .dt +self .sigma *(noise *(self .dt **0.5 ))
        self .state =self .state +dx

        if active .dtype ==torch .bool and active .dim ()==2 :
            active_mask =active
        else :
            active =active .to (dtype =torch .long )
            if active .dim ()==0 :
                active =active .expand (self .n_agents )
            slot_idx =torch .arange (self .max_evs ,device =self .device )
            active_mask =slot_idx .unsqueeze (0 )<active .view (-1 ,1 )
        out =self .state *active_mask .to (dtype =self .state .dtype )

        if OU_CLIP is not None and OU_CLIP >0 :
            out =torch .clamp (out ,-float (OU_CLIP ),float (OU_CLIP ))
        return out


class GaussianNoise :
    """
    Zero-mean i.i.d. Gaussian exploration noise per station and EV slot.

    This is a drop-in replacement for OUNoise with the same
    `sample(active_evs_per_agent)` API. It has no temporal correlation, so a
    station does not inherit a persistent episode-level charge/discharge bias
    from the noise process.
    """
    def __init__ (self ,n_agents :int ,max_evs_per_station :int ,
    sigma :float =1.0 ,clip :float =None ):
        self .n_agents =int (n_agents )
        self .max_evs =int (max_evs_per_station )
        self .sigma =float (sigma )
        self .clip =None if clip is None else float (clip )
        self .device =DEVICE

    def reset (self ):
        """Gaussian noise is stateless across steps and episodes."""
        return

    @torch .no_grad ()
    def sample (self ,active_evs_per_agent ):
        if not torch .is_tensor (active_evs_per_agent ):
            active =torch .as_tensor (active_evs_per_agent ,device =self .device )
        else :
            active =active_evs_per_agent .to (self .device )

        noise =torch .randn ((self .n_agents ,self .max_evs ),device =self .device )*self .sigma

        if active .dtype ==torch .bool and active .dim ()==2 :
            active_mask =active
        else :
            active =active .to (dtype =torch .long )
            if active .dim ()==0 :
                active =active .expand (self .n_agents )
            slot_idx =torch .arange (self .max_evs ,device =self .device )
            active_mask =slot_idx .unsqueeze (0 )<active .view (-1 ,1 )
        out =noise *active_mask .to (dtype =noise .dtype )

        if self .clip is not None and self .clip >0 :
            out =torch .clamp (out ,-self .clip ,self .clip )
        return out


def linear_epsilon_decay (current_episode :int ,
start_episode :int ,
end_episode :int ,
epsilon_initial :float ,
epsilon_final :float )->float :
    """
    Linearly decay epsilon between two episode indices.

    Before `start_episode`, return `epsilon_initial`.
    After `end_episode`, return `epsilon_final`.
    Between them, linearly interpolate.
    """
    ep =int (current_episode )
    s0 =int (start_episode )
    s1 =int (end_episode )
    e0 =float (epsilon_initial )
    e1 =float (epsilon_final )

    if ep <=s0 :
        eps =e0
    elif ep >=s1 :
        eps =e1
    else :
        r =(ep -s0 )/max (1 ,(s1 -s0 ))
        eps =e0 +(e1 -e0 )*r

    lo =min (e0 ,e1 )
    hi =max (e0 ,e1 )
    return max (lo ,min (hi ,eps ))


def sample_epsilon_random_action (num_active :int ,
action_range =(-1.0 ,1.0 ),
like_tensor :torch .Tensor =None )->torch .Tensor :
    """
    Sample a uniform random action tensor for epsilon-greedy exploration.
    """
    low ,high =float (action_range [0 ]),float (action_range [1 ])
    if like_tensor is not None :
        out =torch .empty (num_active ,device =like_tensor .device ,dtype =like_tensor .dtype )
    else :
        out =torch .empty (num_active ,device =DEVICE ,dtype =torch .float32 )
    out .uniform_ (low ,high )
    return out


@torch .no_grad ()
def sample_per_slot_random_mask (num_active :int ,epsilon :float ,
like_tensor :torch .Tensor =None )->torch .Tensor :
    """
    Per-slot Bernoulli mask of length `num_active`.

    Each entry is True with probability `epsilon`, independently. This
    randomizes a small subset of EV slots while letting the rest keep the
    policy action.
    """
    if num_active <=0 :
        if like_tensor is not None :
            return torch .zeros (0 ,dtype =torch .bool ,device =like_tensor .device )
        return torch .zeros (0 ,dtype =torch .bool ,device =DEVICE )
    eps =max (0.0 ,min (1.0 ,float (epsilon )))
    if like_tensor is not None :
        u =torch .rand (num_active ,device =like_tensor .device )
    else :
        u =torch .rand (num_active ,device =DEVICE )
    return u <eps
