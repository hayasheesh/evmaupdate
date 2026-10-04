"""
replay_buffer.py - experience replay buffer.

Provides `ReplayBuffer`, a uniform-sampling buffer.

All core transition tensors live on the configured device for fast minibatch
sampling.

Design notes:
- Lazy initialization: shapes are inferred on the first `cache()` call.
- Temporary CPU staging: data can be accumulated before GPU tensors exist.
- Circular overwrite: `ptr % buf_size` overwrites the oldest entries.
"""
import math
import random

import torch
import numpy as np

device =torch .device ("cuda"if torch .cuda .is_available ()else "cpu")


class ReplayBuffer :
    def __init__ (self ,cap =int (1e5 ),per_ev_local =False ):
        """Initialize the replay buffer with a fixed capacity.

        With per_ev_local each transition also keeps the per-EV local rewards
        and, per EV slot, the slot the same EV holds in s2 (-1 once it left).
        """
        self .buf_size =cap
        self .per_ev_local =bool (per_ev_local )
        self .r_ev_local =None
        self .next_slot =None
        self ._last_idxs =None
        if self .per_ev_local :
            self ._RESUME_TENSOR_NAMES =type (self )._RESUME_TENSOR_NAMES +("r_ev_local","next_slot")

        self .s_dim =None
        self .a_dim =None
        self .n_agents =None
        self .max_evs =None

        self .ptr =0
        self .size =0

        # Balanced replay (offline-to-online fine-tuning). When a fine-tune run
        # inherits the pretrain buffer, the first `offline_size` slots hold that
        # inherited data and online transitions are written after them. Updates
        # then draw from both, instead of refitting the warm-started critic on
        # whatever little online data exists (which degrades the restored
        # policy before it improves). offline_size = 0 keeps the plain
        # single-region behaviour used by pretraining.
        self .offline_size =0
        self .offline_sample_ratio =0.0
        self .offline_ratio_initial =0.0
        self .offline_ratio_final =0.0
        self .offline_decay_steps =0
        self ._online_added =0

        self .s =None
        self .s2 =None
        self .a =None

        self .r_local =None
        self .r_global =None
        self .d =None
        self .actual_station_powers =None
        self .actual_ev_soc_changes =None
        self .estimated_memory_gb =0.0

        self .obs_action_storage_dtype =torch .float16 if device .type =="cuda"else torch .float32
        self .temp_buf =[]

    _RESUME_TENSOR_NAMES =(
        "s",
        "s2",
        "a",
        "r_local",
        "r_global",
        "d",
        "actual_station_powers",
        "actual_ev_soc_changes",
    )

    def training_resume_state_dict (self ):
        """Return the complete physical replay layout for exact continuation.

        Fine-tune snapshots intentionally keep only the newest transitions.
        A pretrain resume cannot do that: removing older transitions changes
        the sampling distribution and therefore the next gradient.  This
        state keeps every populated slot plus the circular write pointer.
        """
        storage_count =int (self .buf_size if self .size >=self .buf_size else self .size )
        tensors ={}
        if self .s is not None :
            for name in self ._RESUME_TENSOR_NAMES :
                value =getattr (self ,name ,None )
                if value is None :
                    raise RuntimeError (f"replay tensor {name!r} is missing")
                tensors [name ]=value [:storage_count ].detach ().cpu ().clone ()
        return {
            "format_version":1 ,
            "buffer_type":type (self ).__name__ ,
            "buf_size":int (self .buf_size ),
            "ptr":int (self .ptr ),
            "size":int (self .size ),
            "storage_count":storage_count ,
            "s_dim":self .s_dim ,
            "a_dim":self .a_dim ,
            "n_agents":self .n_agents ,
            "max_evs":self .max_evs ,
            "obs_action_storage_dtype":str (self .obs_action_storage_dtype ),
            "offline_size":int (self .offline_size ),
            "offline_sample_ratio":float (self .offline_sample_ratio ),
            "offline_ratio_initial":float (self .offline_ratio_initial ),
            "offline_ratio_final":float (self .offline_ratio_final ),
            "offline_decay_steps":int (self .offline_decay_steps ),
            "online_added":int (self ._online_added ),
            "tensors":tensors ,
        }

    def load_training_resume_state_dict (self ,state ):
        """Restore a state produced by :meth:`training_resume_state_dict`."""
        if int (state .get ("format_version",-1 ))!=1 :
            raise ValueError (f"unsupported replay resume format: {state.get('format_version')!r}")
        if str (state .get ("buffer_type"))!=type (self ).__name__ :
            raise ValueError (
            f"replay type mismatch: saved={state.get('buffer_type')} current={type(self).__name__}"
            )
        saved_capacity =int (state ["buf_size"])
        if saved_capacity !=int (self .buf_size ):
            raise ValueError (
            f"replay capacity mismatch: saved={saved_capacity} current={self.buf_size}"
            )
        size =int (state ["size"])
        ptr =int (state ["ptr"])
        storage_count =int (state ["storage_count"])
        if not (0 <=size <=saved_capacity ):
            raise ValueError (f"invalid saved replay size: {size}")
        expected_storage =saved_capacity if size >=saved_capacity else size
        if storage_count !=expected_storage :
            raise ValueError (
            f"replay storage count mismatch: saved={storage_count} expected={expected_storage}"
            )
        if not (0 <=ptr <saved_capacity ):
            raise ValueError (f"invalid saved replay pointer: {ptr}")

        tensors =state .get ("tensors")or {}
        if storage_count >0 :
            for name in self ._RESUME_TENSOR_NAMES :
                saved =tensors .get (name )
                if not torch .is_tensor (saved )or int (saved .shape [0 ])!=storage_count :
                    raise ValueError (f"invalid saved replay tensor {name!r}")
                full_shape =(saved_capacity ,)+tuple (saved .shape [1 :])
                restored =torch .zeros (full_shape ,dtype =saved .dtype ,device =device )
                restored [:storage_count ].copy_ (saved .to (device =device ,dtype =saved .dtype ))
                setattr (self ,name ,restored )
            self .obs_action_storage_dtype =self .s .dtype
        else :
            for name in self ._RESUME_TENSOR_NAMES :
                setattr (self ,name ,None )

        self .ptr =ptr
        self .size =size
        self .s_dim =state .get ("s_dim")
        self .a_dim =state .get ("a_dim")
        self .n_agents =state .get ("n_agents")
        self .max_evs =state .get ("max_evs")
        self .offline_size =int (state .get ("offline_size",0 ))
        self .offline_sample_ratio =float (state .get ("offline_sample_ratio",0.0 ))
        self .offline_ratio_initial =float (state .get ("offline_ratio_initial",0.0 ))
        self .offline_ratio_final =float (state .get ("offline_ratio_final",0.0 ))
        self .offline_decay_steps =int (state .get ("offline_decay_steps",0 ))
        self ._online_added =int (state .get ("online_added",0 ))
        self .temp_buf =[]
        if storage_count >0 :
            self .estimated_memory_gb =sum (
            int (getattr (self ,name ).numel ()) *int (getattr (self ,name ).element_size ())
            for name in self ._RESUME_TENSOR_NAMES
            )/(1024 **3 )
        else :
            self .estimated_memory_gb =0.0
        return self

    def _to_device_tensor (self ,value ,dtype =torch .float32 ):
        if isinstance (value ,torch .Tensor ):
            return value .detach ().to (device =device ,dtype =dtype )
        return torch .as_tensor (value ,dtype =dtype ,device =device )

    def cache (self ,s ,s2 ,a ,r_local ,r_global ,d ,actual_station_powers =None ,actual_ev_soc_changes =None ,
    r_ev_local =None ,next_slot =None ):
        """
        Add one transition to the replay buffer.

        On the first call, the buffer infers tensor shapes, allocates device
        tensors, and flushes any temporarily staged entries.
        """
        if self .per_ev_local :
            if r_ev_local is None or next_slot is None :
                raise ValueError ("a per-EV local buffer needs r_ev_local and next_slot for every transition")
            if self .s is None :
                n_agents =len (s )
                max_evs =int (torch .as_tensor (r_ev_local ).shape [-1 ])
                self .r_ev_local =torch .zeros ((self .buf_size ,n_agents ,max_evs ),dtype =torch .float32 ,device =device )
                self .next_slot =torch .full ((self .buf_size ,n_agents ,max_evs ),-1 ,dtype =torch .int8 ,device =device )
            idx =self .ptr
            self .r_ev_local [idx ]=self ._to_device_tensor (r_ev_local ,torch .float32 )
            self .next_slot [idx ]=self ._to_device_tensor (next_slot ,torch .int8 )
        if self .s is None :
            self .n_agents =len (s )
            self .s_dim =s [0 ].shape [0 ]if isinstance (s [0 ],np .ndarray )else len (s [0 ])

            if isinstance (a ,np .ndarray ):
                self .max_evs =a .shape [2 ]if a .ndim ==3 else a .shape [1 ]
            else :
                try :
                    if isinstance (a ,tuple )or isinstance (a ,list ):
                        if len (a )>0 and hasattr (a [0 ],'shape')and len (a [0 ].shape )>0 :
                            self .max_evs =a [0 ].shape [1 ]
                        else :
                            self .max_evs =5
                    elif isinstance (a ,torch .Tensor ):
                        self .max_evs =a .shape [1 ]if len (a .shape )>1 else 5
                    else :
                        self .max_evs =5
                except (IndexError ,AttributeError ):
                    self .max_evs =5

            self .s =torch .zeros ((self .buf_size ,self .n_agents ,self .s_dim ),
            dtype =self .obs_action_storage_dtype ,device =device )
            self .s2 =torch .zeros ((self .buf_size ,self .n_agents ,self .s_dim ),
            dtype =self .obs_action_storage_dtype ,device =device )
            self .a =torch .zeros ((self .buf_size ,self .n_agents ,self .max_evs ),
            dtype =self .obs_action_storage_dtype ,device =device )
            self .r_local =torch .zeros ((self .buf_size ,self .n_agents ),
            dtype =torch .float32 ,device =device )
            self .r_global =torch .zeros ((self .buf_size ,1 ),
            dtype =torch .float32 ,device =device )
            self .d =torch .zeros ((self .buf_size ,self .n_agents ),
            dtype =torch .float32 ,device =device )
            self .actual_station_powers =torch .zeros ((self .buf_size ,self .n_agents ),
            dtype =torch .float32 ,device =device )
            self .actual_ev_soc_changes =torch .zeros ((self .buf_size ,self .n_agents ,self .max_evs ),
            dtype =torch .float32 ,device =device )
            allocated_bytes =sum (
            int (t .numel ()) *int (t .element_size ())
            for t in (
            self .s ,self .s2 ,self .a ,self .r_local ,self .r_global ,
            self .d ,self .actual_station_powers ,self .actual_ev_soc_changes
            )
            )
            self .estimated_memory_gb =allocated_bytes /(1024 **3 )

            for temp_data in self .temp_buf :
                if len (temp_data )==8 :
                    temp_s ,temp_s2 ,temp_a ,temp_r_local ,temp_r_global ,temp_d ,temp_actual_powers ,temp_actual_ev_soc_changes =temp_data
                elif len (temp_data )==7 :
                    temp_s ,temp_s2 ,temp_a ,temp_r_local ,temp_r_global ,temp_d ,temp_actual_powers =temp_data
                    temp_actual_ev_soc_changes =None
                else :
                    temp_s ,temp_s2 ,temp_a ,temp_r_local ,temp_r_global ,temp_d =temp_data
                    temp_actual_powers =None
                    temp_actual_ev_soc_changes =None
                self ._cache_to_tensor (temp_s ,temp_s2 ,temp_a ,temp_r_local ,temp_r_global ,temp_d ,temp_actual_powers ,temp_actual_ev_soc_changes )
            self .temp_buf =[]

        if self .s is not None :
            self ._cache_to_tensor (s ,s2 ,a ,r_local ,r_global ,d ,actual_station_powers ,actual_ev_soc_changes )
        else :
            if actual_ev_soc_changes is not None :
                self .temp_buf .append ((s ,s2 ,a ,r_local ,r_global ,d ,actual_station_powers ,actual_ev_soc_changes ))
            elif actual_station_powers is not None :
                self .temp_buf .append ((s ,s2 ,a ,r_local ,r_global ,d ,actual_station_powers ))
            else :
                self .temp_buf .append ((s ,s2 ,a ,r_local ,r_global ,d ))

    def _cache_to_tensor (self ,s ,s2 ,a ,r_local ,r_global ,d ,actual_station_powers =None ,actual_ev_soc_changes =None ):
        """Write one transition into the current buffer slot."""
        idx =self .ptr

        self .s [idx ]=self ._to_device_tensor (s ,self .obs_action_storage_dtype )
        self .s2 [idx ]=self ._to_device_tensor (s2 ,self .obs_action_storage_dtype )
        self .a [idx ]=self ._to_device_tensor (a ,self .obs_action_storage_dtype )
        self .r_local [idx ]=self ._to_device_tensor (r_local ,torch .float32 )
        self .r_global [idx ]=self ._to_device_tensor (r_global ,torch .float32 ).flatten ()[:1 ]
        self .d [idx ]=self ._to_device_tensor (d ,torch .float32 )

        if actual_station_powers is not None :
            self .actual_station_powers [idx ]=self ._to_device_tensor (actual_station_powers ,torch .float32 )
        else :
            raise ValueError ("actual_station_powers must be provided. Environment should provide actual station powers after applying SoC constraints.")

        if actual_ev_soc_changes is not None :
            self .actual_ev_soc_changes [idx ]=self ._to_device_tensor (actual_ev_soc_changes ,torch .float32 )
        else :
            self .actual_ev_soc_changes [idx ]=self .a [idx ].clone ()

        if self .offline_size >0 :
            # Online writes cycle through the tail region only, so the
            # inherited offline transitions are never overwritten.
            online_cap =max (1 ,self .buf_size -self .offline_size )
            next_ptr =self .ptr +1
            if next_ptr >=self .buf_size :
                next_ptr =self .offline_size
            self .ptr =next_ptr
            self ._online_added +=1
            self .size =min (
            self .offline_size +min (self ._online_added ,online_cap ),
            self .buf_size ,
            )
            self ._update_offline_ratio ()
        else :
            self .ptr =(self .ptr +1 )%self .buf_size
            self .size =min (self .size +1 ,self .buf_size )

    def _update_offline_ratio (self ):
        """Decay the offline sampling share as online data accumulates.

        Early in fine-tuning the online region is tiny, so drawing mostly from
        it refits the warm-started critic on a near-degenerate sample. The
        share therefore starts high and anneals toward `offline_ratio_final`
        over `offline_decay_steps` online transitions, which is the simple
        deterministic stand-in for the "on-policyness" weighting used by
        adaptive balanced-replay schemes.
        """
        if self .offline_size <=0 or self .offline_decay_steps <=0 :
            return
        frac =min (1.0 ,float (self ._online_added )/float (self .offline_decay_steps ))
        self .offline_sample_ratio =(
        self .offline_ratio_initial
        +(self .offline_ratio_final -self .offline_ratio_initial )*frac
        )

    def mark_offline_region (self ,ratio_initial =0.5 ,ratio_final =0.25 ,decay_steps =0 ):
        """Freeze everything currently stored as the offline (pretrain) region."""
        if self .s is None or self .size <=0 :
            return False
        self .offline_size =int (self .size )
        self .ptr =self .offline_size %self .buf_size
        self ._online_added =0
        self .offline_ratio_initial =float (ratio_initial )
        self .offline_ratio_final =float (ratio_final )
        self .offline_decay_steps =int (decay_steps )
        self .offline_sample_ratio =float (ratio_initial )
        return True

    @property
    def pending_size (self ):
        """Transitions that count toward the warmup gate.

        Without an offline region this is the whole buffer, which is what a
        pretrain wants.  After a warm start the buffer already holds the
        pretrain's own transitions, so counting them clears the gate on the
        first step of the new day -- and the batch drawn there is half offline
        and half the single new transition repeated, because the online region
        has exactly one index to draw from.  Counting only the new region makes
        the gate mean what it meant before: wait until there is something to
        learn from.
        """
        if self .offline_size <=0 :
            return int (self .size )
        return max (0 ,int (self .size )-int (self .offline_size ))

    def _mixed_start_indices (self ,batch ,n ):
        """Draw start indices from the offline and online regions.

        Returns None when no offline region is active so callers keep their
        original single-region sampling path.
        """
        if self .offline_size <=0 :
            return None
        online_len =max (0 ,int (self .size )-int (self .offline_size ))
        offline_max_start =max (1 ,int (self .offline_size )-n +1 )
        if online_len <n :
            # Not enough online data to form an n-step window yet.
            return torch .randint (0 ,offline_max_start ,(batch ,),device =device )
        ratio =float (min (max (self .offline_sample_ratio ,0.0 ),1.0 ))
        n_offline =int (round (batch *ratio ))
        n_offline =max (0 ,min (batch ,n_offline ))
        n_online =batch -n_offline
        parts =[]
        if n_offline >0 :
            parts .append (torch .randint (0 ,offline_max_start ,(n_offline ,),device =device ))
        if n_online >0 :
            online_max_start =max (1 ,online_len -n +1 )
            parts .append (
            torch .randint (0 ,online_max_start ,(n_online ,),device =device )
            +int (self .offline_size )
            )
        return torch .cat (parts )if len (parts )>1 else parts [0 ]

    def sample (self ,batch ):
        """Sample a random minibatch directly from device tensors."""
        if self .s is None :
            raise RuntimeError ("ReplayBuffer is not initialized. cache() must be called to initialize GPU buffers before sampling.")

        mixed =self ._mixed_start_indices (batch ,1 )
        idxs =mixed if mixed is not None else torch .randint (0 ,self .size ,(batch ,),device =device )
        return self ._gather_batch (idxs )

    def per_ev_batch (self ):
        """Per-EV local rewards and next slots of the last sampled batch."""
        if not self .per_ev_local or self ._last_idxs is None :
            raise RuntimeError ("no per-EV local data for the last sample")
        idxs =self ._last_idxs
        return self .r_ev_local [idxs ],self .next_slot [idxs ].to (dtype =torch .long )

    def _gather_batch (self ,idxs ):
        """Gather one batch given pre-selected indices (1-step semantics)."""
        self ._last_idxs =idxs
        s_batch =self .s [idxs ].to (dtype =torch .float32 )
        s2_batch =self .s2 [idxs ].to (dtype =torch .float32 )
        a_batch =self .a [idxs ].to (dtype =torch .float32 )
        r_local_batch =self .r_local [idxs ]
        r_global_batch =self .r_global [idxs ]
        d_batch =self .d [idxs ]
        actual_station_powers_batch =self .actual_station_powers [idxs ]
        actual_ev_soc_changes_batch =self .actual_ev_soc_changes [idxs ]
        return tuple (t .to (device ,non_blocking =True )for t in (
        s_batch ,s2_batch ,a_batch ,r_local_batch ,r_global_batch ,d_batch ,
        actual_station_powers_batch ,actual_ev_soc_changes_batch ))

    def sample_with_nstep_global (self ,batch ,n_step ,gamma ):
        """
        Sample a minibatch and additionally compute n-step global rewards.

        Returns the standard 1-step batch plus three extra tensors for the
        global critic target:
            r_global_n  : sum_{k=0..k_eff-1} gamma^k * r_global[idx+k]
            s2_n        : s2[idx + k_eff - 1]                  (state after k_eff steps)
            d_n         : done flag at idx + k_eff - 1         (per-agent)
            n_eff       : effective n used per sample          (1..n_step)
        where k_eff = min(n_step, k of first done in window) so that bootstrap
        cleanly stops at episode boundaries. The local 1-step quantities
        (s2, r_local, d) are returned unchanged so the caller can use them
        for the local critic.

        Buffer wrap-around is avoided by restricting starting indices to
        positions whose chronological window of length n_step does not cross
        the write pointer.
        """
        if self .s is None :
            raise RuntimeError ("ReplayBuffer is not initialized.")
        n =max (1 ,int (n_step ))

        mixed =self ._mixed_start_indices (batch ,n )
        if mixed is not None :
            # Balanced replay: offline (pretrain) and online regions are
            # sampled separately, and neither wraps into the other.
            idxs =mixed
        elif self .size <self .buf_size :
            # Buffer not yet wrapped: chronological order matches index order.
            max_start =max (1 ,self .size -n +1 )
            idxs =torch .randint (0 ,max_start ,(batch ,),device =device )
        else :
            # Buffer wrapped. Forbidden start = last n-1 chronological positions
            # whose lookahead window would cross the write pointer (oldest).
            # Allowed offsets from `ptr`: [0, buf_size - n + 1).
            offsets =torch .randint (0 ,self .buf_size -n +1 ,(batch ,),device =device )
            idxs =(offsets +self .ptr )%self .buf_size

        # Standard 1-step batch (used for local critic and as base for global)
        std_batch =self ._gather_batch (idxs )
        s_b ,s2_b ,a_b ,r_local_b ,r_global_b ,d_b ,asp_b ,aevsc_b =std_batch

        # Compute n-step global return with done-truncation
        B =idxs .size (0 )
        r_n =torch .zeros ((B ,1 ),dtype =torch .float32 ,device =device )
        # Track if we've already passed an episode boundary (then stop accumulating)
        stopped =torch .zeros ((B ,1 ),dtype =torch .float32 ,device =device )
        n_eff =torch .ones ((B ,),dtype =torch .long ,device =device )
        s2_n =s2_b .clone ()
        d_n =d_b .clone ()
        discount =1.0

        for k in range (n ):
            cur_idx =(idxs +k )%self .buf_size
            r_k =self .r_global [cur_idx ]
            d_k_per_agent =self .d [cur_idx ]
            d_k_any =d_k_per_agent .max (dim =1 ,keepdim =True )[0 ]

            # Add this step's reward only if not yet stopped
            active =1.0 -stopped
            r_n =r_n +discount *r_k *active

            # Update state-after-window and done-flag-of-window for samples still active
            active_b =(active >0.5 ).squeeze (-1 )  # bool [B]
            if active_b .any ():
                s2_n [active_b ]=self .s2 [cur_idx [active_b ]].to (dtype =torch .float32 )
                d_n [active_b ]=d_k_per_agent [active_b ]
                n_eff [active_b ]=k +1

            # Update stopped flag for next iteration
            stopped =torch .maximum (stopped ,d_k_any *active )
            discount *=gamma

        return (
        s_b ,s2_b ,a_b ,r_local_b ,r_global_b ,d_b ,asp_b ,aevsc_b ,
        # extra n-step augmentation for the global critic
        r_n ,s2_n ,d_n ,n_eff ,
        )
