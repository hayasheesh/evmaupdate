"""
replay_buffer.py - experience replay buffer.

Provides `ReplayBuffer`, a uniform-sampling buffer.

All core transition tensors live on the configured device for fast minibatch
sampling.

Design notes:
- Lazy initialization: shapes are inferred on the first `cache()` call.
- Circular overwrite: `ptr % buf_size` overwrites the oldest entries.
"""
import torch
import numpy as np

device =torch .device ("cuda"if torch .cuda .is_available ()else "cpu")


class ReplayBuffer :
    def __init__ (self ,cap =int (1e5 )):
        """Initialize the replay buffer with a fixed capacity."""
        self .buf_size =cap

        self .s_dim =None
        self .a_dim =None
        self .n_agents =None
        self .max_evs =None

        self .ptr =0
        self .size =0

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

        Removing older transitions would change the sampling distribution and
        therefore the next gradient, so this state keeps every populated slot
        plus the circular write pointer.
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

    def cache (self ,s ,s2 ,a ,r_local ,r_global ,d ,actual_station_powers ,actual_ev_soc_changes =None ):
        """
        Add one transition to the replay buffer.

        On the first call, the buffer infers tensor shapes and allocates device
        tensors.
        """
        if self .s is None :
            self .n_agents =len (s )
            self .s_dim =s [0 ].shape [0 ]if isinstance (s [0 ],np .ndarray )else len (s [0 ])
            self .max_evs =int (a .shape [-1 ])

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

        self ._cache_to_tensor (s ,s2 ,a ,r_local ,r_global ,d ,actual_station_powers ,actual_ev_soc_changes )

    def _cache_to_tensor (self ,s ,s2 ,a ,r_local ,r_global ,d ,actual_station_powers ,actual_ev_soc_changes =None ):
        """Write one transition into the current buffer slot."""
        idx =self .ptr

        self .s [idx ]=self ._to_device_tensor (s ,self .obs_action_storage_dtype )
        self .s2 [idx ]=self ._to_device_tensor (s2 ,self .obs_action_storage_dtype )
        self .a [idx ]=self ._to_device_tensor (a ,self .obs_action_storage_dtype )
        self .r_local [idx ]=self ._to_device_tensor (r_local ,torch .float32 )
        self .r_global [idx ]=self ._to_device_tensor (r_global ,torch .float32 ).flatten ()[:1 ]
        self .d [idx ]=self ._to_device_tensor (d ,torch .float32 )
        self .actual_station_powers [idx ]=self ._to_device_tensor (actual_station_powers ,torch .float32 )

        if actual_ev_soc_changes is not None :
            self .actual_ev_soc_changes [idx ]=self ._to_device_tensor (actual_ev_soc_changes ,torch .float32 )
        else :
            self .actual_ev_soc_changes [idx ]=self .a [idx ].clone ()

        self .ptr =(self .ptr +1 )%self .buf_size
        self .size =min (self .size +1 ,self .buf_size )

    def sample (self ,batch ):
        """Sample a random minibatch directly from device tensors."""
        if self .s is None :
            raise RuntimeError ("ReplayBuffer is not initialized. cache() must be called to initialize GPU buffers before sampling.")

        idxs =torch .randint (0 ,self .size ,(batch ,),device =device )
        return self ._gather_batch (idxs )

    def _gather_batch (self ,idxs ):
        """Gather one batch given pre-selected indices (1-step semantics)."""
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

        if self .size <self .buf_size :
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
