"""
Critic networks for station-local and system-global value estimation.

LocalEvMLPCritic evaluates one charging station at a time. It receives the
normalized station observation, that station's EV-slot actions, and the realized
station power after EVEnv applies SoC and station-power constraints. It returns a
scalar local Q value for SoC/departure-oriented learning.

GlobalMLPCritic evaluates the joint multi-station action. It receives all
station EV blocks, global demand/time context, all EV-slot actions, and realized
station powers. It returns a scalar global Q value for dispatch tracking plus
per-station intermediate utilities for diagnostics.
"""
import torch
import torch .nn as nn
from Config import (
GLOBAL_CRITIC_KEEP_COMMON_MODE ,
LOCAL_CRITIC_HIDDEN_SIZE ,
GLOBAL_CRITIC_HIDDEN_SIZE ,
MAX_EV_PER_STATION ,
MIXER_B_MAX ,
MIXER_B_MAX_ENABLE ,
)
from environment.observation_config import (
EV_FEAT_DIM ,
LOCAL_TAIL_DIM ,
GLOBAL_TAIL_DIM ,
)


class LocalEvMLPCritic (nn .Module ):
    def __init__ (self ,ev_feat_dim ,a_dim ,max_evs ,hid =LOCAL_CRITIC_HIDDEN_SIZE ,station_state_dim =None ,init_gain =1.0 ):
        """
        Build the local critic for one station.

        Each EV token is concatenated with its action, encoded independently,
        masked by presence, and mean-pooled into a station embedding.
        """
        super ().__init__ ()
        self .ev_feat_dim =ev_feat_dim
        self .a_dim =a_dim
        self .max_evs =max_evs
        self .hid =hid
        self .station_state_dim =station_state_dim if station_state_dim is not None else (ev_feat_dim *max_evs +LOCAL_TAIL_DIM )
        self .init_gain =init_gain

        additional_features =1 +int (LOCAL_TAIL_DIM )
        set_hid =max (hid //2 ,32 )

        self .token_encoder =nn .Sequential (
        nn .Linear (self .ev_feat_dim +1 ,set_hid ),
        nn .LayerNorm (set_hid ),
        nn .LeakyReLU (0.1 ),
        nn .Linear (set_hid ,set_hid ),
        nn .LayerNorm (set_hid ),
        nn .LeakyReLU (0.1 ),
        )

        self .q_head =nn .Sequential (
        nn .Linear (set_hid +additional_features ,hid //2 ),
        nn .LayerNorm (hid //2 ),
        nn .LeakyReLU (0.1 ),
        nn .Linear (hid //2 ,1 ),
        )

        self .apply (self ._init_weights )

    def _init_weights (self ,m ):
        if isinstance (m ,nn .Linear ):
            nn .init .xavier_uniform_ (m .weight ,gain =self .init_gain )
            nn .init .constant_ (m .bias ,0 )

    def forward (self ,s_flat ,a_flat ,key_padding_mask =None ,return_attn =False ,actual_station_powers =None ):
        """
        Evaluate a station-level state/action pair.

        `s_flat` has shape `(state_dim,)` or `(batch, state_dim)`. `a_flat` has
        shape `(max_evs,)` or `(batch, max_evs)`. `actual_station_powers` is the
        realized, normalized station power used as an additional physical signal.
        """
        if s_flat .dim ()==1 :
            s_flat =s_flat .unsqueeze (0 )
        if a_flat .dim ()==1 :
            a_flat =a_flat .unsqueeze (0 )

        B =s_flat .size (0 )

        if actual_station_powers is None :
            raise ValueError ("actual_station_powers must be provided for LocalEvMLPCritic")
        if actual_station_powers .dim ()>1 :
            total_ev_power =actual_station_powers .sum (dim =1 ,keepdim =True )
        else :
            total_ev_power =actual_station_powers .unsqueeze (-1 )

        ev_dim =self .max_evs *self .ev_feat_dim
        ev_flat =s_flat [:,:ev_dim ]
        ev_tokens =ev_flat .view (B ,self .max_evs ,self .ev_feat_dim )
        presence =(ev_tokens [...,0 :1 ]>0.5 ).float ()

        if a_flat .size (1 )!=self .max_evs :
            raise ValueError (f"Invalid local action shape: {tuple(a_flat.shape)} expected second dim {self.max_evs}")
        a_tokens =torch .clamp (a_flat ,-1.0 ,1.0 ).unsqueeze (-1 )

        token_input =torch .cat ([ev_tokens ,a_tokens ],dim =-1 )
        token_feat =self .token_encoder (token_input )

        # Presence masking removes padded EV slots from the station embedding.
        # If a station is empty, denom is clamped so the pooled vector remains
        # finite and contributes no active-EV signal.
        denom =presence .sum (dim =1 ,keepdim =False ).clamp (min =1.0 )
        pooled =(token_feat *presence ).sum (dim =1 )/denom

        tail_start =self .max_evs *self .ev_feat_dim
        tail =s_flat [:,tail_start :tail_start +int (LOCAL_TAIL_DIM )]
        if tail .size (1 )<int (LOCAL_TAIL_DIM ):
            tail =torch .cat (
            [tail ,s_flat .new_zeros (B ,int (LOCAL_TAIL_DIM )-tail .size (1 ))],dim =1
            )

        scaled_total_ev_power =torch .clamp (total_ev_power ,-1.0 ,1.0 )
        feat =torch .cat ([pooled ,scaled_total_ev_power ,tail ],dim =1 )
        out =self .q_head (feat )
        out =torch .nan_to_num_ (out ,nan =0.0 ,posinf =0.0 ,neginf =0.0 )
        return out


class LocalPerEvCritic (nn .Module ):
    """Station-local value as the sum of per-EV values.

    EV j's value is Q_j = head(token(x_j, a_j) + context(s)): its own features
    and action, plus a station context built from the EV features and local
    tail only. The context carries no action, because an EV's SoC reward
    depends on that EV's actions alone; so d(sum_k Q_k)/d a_j = d Q_j/d a_j and
    each action is credited with its own EV's value. One set of weights serves
    every EV slot.
    """

    def __init__ (self ,ev_feat_dim ,max_evs ,hid =LOCAL_CRITIC_HIDDEN_SIZE ,station_state_dim =None ,init_gain =1.0 ):
        super ().__init__ ()
        self .ev_feat_dim =ev_feat_dim
        self .max_evs =max_evs
        self .station_state_dim =station_state_dim if station_state_dim is not None else (ev_feat_dim *max_evs +LOCAL_TAIL_DIM )
        self .init_gain =init_gain
        set_hid =max (hid //2 ,32 )
        head_hid =max (hid //4 ,16 )

        self .token_encoder =nn .Sequential (
        nn .Linear (self .ev_feat_dim +1 ,set_hid ),
        nn .LayerNorm (set_hid ),
        nn .LeakyReLU (0.1 ),
        nn .Linear (set_hid ,set_hid ),
        nn .LayerNorm (set_hid ),
        nn .LeakyReLU (0.1 ),
        )
        self .context_encoder =nn .Sequential (
        nn .Linear (self .ev_feat_dim ,set_hid ),
        nn .LayerNorm (set_hid ),
        nn .LeakyReLU (0.1 ),
        )
        # Computed once per station and added to every EV's token.
        self .context_proj =nn .Linear (set_hid +int (LOCAL_TAIL_DIM ),set_hid )
        self .q_head =nn .Sequential (
        nn .LayerNorm (set_hid ),
        nn .LeakyReLU (0.1 ),
        nn .Linear (set_hid ,head_hid ),
        nn .LayerNorm (head_hid ),
        nn .LeakyReLU (0.1 ),
        nn .Linear (head_hid ,1 ),
        )
        self .apply (self ._init_weights )

    def _init_weights (self ,m ):
        if isinstance (m ,nn .Linear ):
            nn .init .xavier_uniform_ (m .weight ,gain =self .init_gain )
            nn .init .constant_ (m .bias ,0 )

    def per_ev (self ,s_flat ,a_flat ):
        """Per-slot values, [batch, max_evs], zero on empty slots."""
        if s_flat .dim ()==1 :
            s_flat =s_flat .unsqueeze (0 )
        if a_flat .dim ()==1 :
            a_flat =a_flat .unsqueeze (0 )
        B =s_flat .size (0 )
        if a_flat .size (1 )!=self .max_evs :
            raise ValueError (f"Invalid local action shape: {tuple(a_flat.shape)} expected second dim {self.max_evs}")
        ev_dim =self .max_evs *self .ev_feat_dim
        ev_tokens =s_flat [:,:ev_dim ].view (B ,self .max_evs ,self .ev_feat_dim )
        presence =(ev_tokens [...,0 :1 ]>0.5 ).float ()

        tail =s_flat [:,ev_dim :ev_dim +int (LOCAL_TAIL_DIM )]
        if tail .size (1 )<int (LOCAL_TAIL_DIM ):
            tail =torch .cat ([tail ,s_flat .new_zeros (B ,int (LOCAL_TAIL_DIM )-tail .size (1 ))],dim =1 )
        context_feat =self .context_encoder (ev_tokens )
        denom =presence .sum (dim =1 ).clamp (min =1.0 )
        pooled =(context_feat *presence ).sum (dim =1 )/denom
        context =self .context_proj (torch .cat ([pooled ,tail ],dim =1 ))

        a_tokens =torch .clamp (a_flat ,-1.0 ,1.0 ).unsqueeze (-1 )
        token_feat =self .token_encoder (torch .cat ([ev_tokens ,a_tokens ],dim =-1 ))
        q =self .q_head (token_feat +context .unsqueeze (1 )).squeeze (-1 )
        q =torch .nan_to_num (q ,nan =0.0 ,posinf =0.0 ,neginf =0.0 )
        return q *presence .squeeze (-1 )

    def forward (self ,s_flat ,a_flat ,key_padding_mask =None ,return_attn =False ,actual_station_powers =None ):
        """The station value, [batch, 1]: the sum of the per-EV values."""
        return self .per_ev (s_flat ,a_flat ).sum (dim =1 ,keepdim =True )


class GlobalMLPCritic (nn .Module ):
    def __init__ (self ,s_dim ,a_dim ,n_agent ,hid =GLOBAL_CRITIC_HIDDEN_SIZE ,station_state_dim =None ,init_gain =1.0 ):
        """
        Build the global critic with a QMIX-style mixer.

        Per-station state/action embeddings produce utilities `u_i`. The mixer
        combines them with non-negative context-dependent weights conditioned on
        demand/time features and realized station powers.
        """
        super ().__init__ ()
        self .s_dim =s_dim
        self .a_dim =a_dim
        self .n_agent =n_agent
        self .hid =hid
        self .ev_features_per_station =EV_FEAT_DIM *MAX_EV_PER_STATION
        self .station_state_dim =self .ev_features_per_station
        self .init_gain =init_gain

        self .station_emb_dim =hid //2
        self .per_station_sa =nn .Sequential (
        nn .Linear (self .ev_features_per_station +a_dim ,self .station_emb_dim ),
        nn .LayerNorm (self .station_emb_dim ),
        nn .LeakyReLU (0.1 ),
        )

        self .per_agent_head =nn .Sequential (
        nn .Linear (self .station_emb_dim ,self .station_emb_dim //2 ),
        nn .LayerNorm (self .station_emb_dim //2 ),
        nn .LeakyReLU (0.1 ),
        nn .Linear (self .station_emb_dim //2 ,1 ),
        )

        # Mixer context: total EV power, current step, global demand lookahead,
        # and the vector of realized station powers.
        common_in_dim =int (GLOBAL_TAIL_DIM )+n_agent
        self .mixer_w =nn .Sequential (
        nn .Linear (common_in_dim ,hid //2 ),
        nn .LayerNorm (hid //2 ),
        nn .LeakyReLU (0.1 ),
        nn .Linear (hid //2 ,n_agent ),
        )
        self .mixer_b =nn .Sequential (
        nn .Linear (common_in_dim ,hid //2 ),
        nn .LayerNorm (hid //2 ),
        nn .LeakyReLU (0.1 ),
        nn .Linear (hid //2 ,1 ),
        )
        self .softplus =nn .Softplus ()
        self .keep_common_mode =bool (GLOBAL_CRITIC_KEEP_COMMON_MODE )
        if self .keep_common_mode :
            # softplus(0.5413) == 1, so the station mean starts passing through
            # at full weight and the mixer can shrink it if it is not useful.
            self .common_mode_gain =nn .Parameter (torch .tensor (0.5413 ))

        self .apply (self ._init_weights )

    def _init_weights (self ,m ):
        if isinstance (m ,nn .Linear ):
            nn .init .xavier_uniform_ (m .weight ,gain =self .init_gain )
            nn .init .constant_ (m .bias ,0 )

    def forward (self ,s ,a ,key_padding_mask =None ,return_attn =False ,actual_station_powers =None ,attn_per_head =False ):
        """
        Evaluate the joint multi-station state/action.

        `s` is the global critic observation. `a` must be shaped
        `(batch, n_agent, max_evs)`. `actual_station_powers` must be shaped
        `(batch, n_agent)` after normalization. The return value is
        `(q_global, u)` unless `return_attn=True`, in which case a placeholder
        third value is kept for compatibility.
        """
        if s .dim ()==1 :
            s =s .unsqueeze (0 )
        B =s .size (0 )

        ev_features_total =self .n_agent *self .ev_features_per_station
        ev_features_flat =s [:,:ev_features_total ]

        common_info_start =ev_features_total
        global_context =s [:,common_info_start :common_info_start +int (GLOBAL_TAIL_DIM )]
        if global_context .size (1 )<int (GLOBAL_TAIL_DIM ):
            global_context =torch .cat (
            [global_context ,s .new_zeros (B ,int (GLOBAL_TAIL_DIM )-global_context .size (1 ))],dim =1
            )

        if actual_station_powers is None :
            raise ValueError ("actual_station_powers must be provided for GlobalMLPCritic")
        station_power_vec =torch .clamp (actual_station_powers ,-1.0 ,1.0 )
        if station_power_vec .dim ()==1 :
            station_power_vec =station_power_vec .unsqueeze (0 )
        if station_power_vec .dim ()!=2 or station_power_vec .size (1 )!=self .n_agent :
            raise ValueError (
            f"Invalid actual_station_powers shape: {tuple(station_power_vec.shape)} "
            f"expected (B, {self.n_agent})"
            )

        if a is None :
            raise ValueError ("action tensor 'a' must be provided for GlobalMLPCritic")
        actions_vec =torch .clamp (a ,-1.0 ,1.0 )
        if actions_vec .dim ()!=3 or actions_vec .size (1 )!=self .n_agent or actions_vec .size (2 )!=self .a_dim :
            raise ValueError (f"Invalid action shape: {tuple(actions_vec.shape)} expected (B, n, {self.a_dim})")

        ev_per_station =ev_features_flat .reshape (B ,self .n_agent ,self .ev_features_per_station )
        station_input =torch .cat ([ev_per_station ,actions_vec ],dim =2 )
        station_feats =self .per_station_sa (
        station_input .reshape (B *self .n_agent ,-1 )
        ).reshape (B ,self .n_agent ,self .station_emb_dim )

        u_tokens =self .per_agent_head (station_feats )
        u =u_tokens .squeeze (-1 )
        if key_padding_mask is not None :
            u =u .masked_fill (key_padding_mask ,0.0 )

        common_features =torch .cat ([global_context ,station_power_vec ],dim =1 )

        w =self .mixer_w (common_features )
        w =self .softplus (w )
        b_raw =self .mixer_b (common_features )
        # Bound the mixer bias with a scaled tanh:
        #     b = K * tanh(b_raw / K).
        # This preserves near-zero gradient scale while preventing the bias term
        # from absorbing unbounded global-Q drift.
        if MIXER_B_MAX_ENABLE :
            b =MIXER_B_MAX *torch .tanh (b_raw /MIXER_B_MAX )
        else :
            b =b_raw

        # Mean-centered mixer: the global value depends on station utilities as
        # advantages relative to the station mean. Normalizing weights to mean 1
        # keeps the mixer scale stable while still allowing station-specific
        # credit assignment.
        u_mean =u .mean (dim =1 ,keepdim =True )
        u_centered =u -u_mean
        w_mean =w .mean (dim =1 ,keepdim =True ).clamp (min =1e-6 )
        w_norm =w /w_mean
        q_global =(w_norm *u_centered ).mean (dim =1 ,keepdim =True )+b
        if self .keep_common_mode :
            # The centred term is untouched, so station-specific credit is
            # unchanged; this only adds back the channel centring removes.
            # Softplus keeps the share positive, which also gives every station
            # a non-negative partial derivative the centred term alone lacks.
            q_global =q_global +self .softplus (self .common_mode_gain )*u_mean

        if not torch .isfinite (q_global ).all ():
            q_global =torch .nan_to_num (q_global ,nan =0.0 ,posinf =0.0 ,neginf =0.0 )
        if not torch .isfinite (u ).all ():
            u =torch .nan_to_num (u ,nan =0.0 ,posinf =0.0 ,neginf =0.0 )

        if return_attn :
            return q_global ,u ,None

        return q_global ,u
