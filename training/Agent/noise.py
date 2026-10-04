"""
noise.py - exploration noise and the epsilon schedule.

This module provides:
- `GaussianNoise`: independent Gaussian action-space exploration
- `linear_epsilon_decay`: linear epsilon schedule
"""
import torch

from Config import DEVICE


class GaussianNoise :
    """
    Zero-mean i.i.d. Gaussian exploration noise per station and EV slot.

    It has no temporal correlation, so a station does not inherit a persistent
    episode-level charge/discharge bias from the noise process.
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
    def sample (self ,active_slot_mask ):
        """Noise for every slot, zero where ``active_slot_mask`` [stations, slots] is False."""
        active_mask =active_slot_mask .to (self .device )
        noise =torch .randn ((self .n_agents ,self .max_evs ),device =self .device )*self .sigma
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
