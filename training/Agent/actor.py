"""
Actor network for one charging station.

Input:
- A normalized station observation vector with layout
  `[EV slots flattened][local tail features]` (environment.observation_config).
- Each EV slot uses `EV_FEAT_DIM` features; feature 0 is the presence flag.

Output:
- One continuous action per EV slot in `[-1, 1]`.
- Positive means charging; negative means discharging.

Architecture:
- Encode each EV slot as a token.
- Pool active EV tokens to summarize station context.
- Concatenate per-EV token, station pooled context, local tail features and
  the station's active-EV count.
- Produce per-EV actions with a tanh output head.
"""
import torch
import torch .nn as nn
from Config import ACTOR_HIDDEN_SIZE
from environment.observation_config import (
EV_FEAT_DIM ,
LOCAL_TAIL_DIM ,
)


class Actor (nn .Module ):
    def __init__ (self ,s_dim ,max_evs_per_station ,hid =ACTOR_HIDDEN_SIZE ,station_state_dim =None ,init_gain =0.1 ):
        """
        Build the station actor.

        `s_dim` is the normalized station observation width. The implementation
        derives the EV block and tail dimensions from observation_config so the
        policy stays aligned with EVEnv's observation layout.
        """
        super (Actor ,self ).__init__ ()
        self .s_dim =s_dim
        self .max_evs =max_evs_per_station
        self .hidden_size =hid
        self .station_state_dim =station_state_dim if station_state_dim is not None else s_dim
        self .init_gain =init_gain

        self .ev_feat_dim =EV_FEAT_DIM
        self .ev_state_dim =self .ev_feat_dim *self .max_evs
        self .tail_dim =LOCAL_TAIL_DIM

        set_hid =max (hid //2 ,32 )
        self .ev_token_encoder =nn .Sequential (
        nn .Linear (self .ev_feat_dim ,set_hid ),
        nn .LayerNorm (set_hid ),
        nn .LeakyReLU (0.1 ),
        nn .Linear (set_hid ,set_hid ),
        nn .LayerNorm (set_hid ),
        nn .LeakyReLU (0.1 ),
        )
        # Per-EV token, pooled station context, tail, active-EV count.
        self .ev_action_head =nn .Sequential (
        nn .Linear (set_hid *2 +self .tail_dim +1 ,set_hid ),
        nn .LayerNorm (set_hid ),
        nn .LeakyReLU (0.1 ),
        nn .Linear (set_hid ,1 ),
        )

        self .apply (self ._init_weights )

    def _init_weights (self ,m ):
        if isinstance (m ,nn .Linear ):
            nn .init .xavier_uniform_ (m .weight ,gain =self .init_gain )
            nn .init .constant_ (m .bias ,0 )

    def forward (self ,x ):
        """
        Map a normalized station observation to per-slot actions.

        Args:
            x: Tensor shaped `(state_dim,)` or `(batch, state_dim)`.

        Returns:
            One value per EV slot.
        """
        if x .dim ()==1 :
            x =x .unsqueeze (0 )
            squeeze_output =True
        else :
            squeeze_output =False

        batch_size =x .shape [0 ]
        x_normalized =x

        ev_flat =x_normalized [:,:self .ev_state_dim ]
        ev_tokens =ev_flat .view (batch_size ,self .max_evs ,self .ev_feat_dim )
        presence =(ev_tokens [...,0 :1 ]>0.5 ).float ()

        tail =x_normalized [:,self .ev_state_dim :self .ev_state_dim +self .tail_dim ]

        token_feat =self .ev_token_encoder (ev_tokens )

        # Mean-pool only active EV slots. Empty padded slots have presence=0 and
        # therefore do not affect the station-level context vector.
        denom =presence .sum (dim =1 ,keepdim =True ).clamp (min =1.0 )
        pooled =(token_feat *presence ).sum (dim =1 ,keepdim =True )/denom

        pooled_expand =pooled .expand (-1 ,self .max_evs ,-1 )
        tail_expand =tail .unsqueeze (1 ).expand (-1 ,self .max_evs ,-1 )
        # The divisor the pooling already computed, scaled to [0, 1]. It is
        # the one thing the head cannot recover: it sees one token at a
        # time and a mean that is invariant to how many went into it.
        count =(denom /float (max (self .max_evs ,1 ))).expand (-1 ,self .max_evs ,-1 )
        token_input =torch .cat ([token_feat ,pooled_expand ,tail_expand ,count ],dim =-1 )

        raw_actions =torch .tanh (self .ev_action_head (token_input ).squeeze (-1 ))
        # Keep padded slots inert. EVEnv applies actions only to active slots,
        # but zeroing here also keeps critic inputs and diagnostics clean.
        raw_actions =raw_actions *presence .squeeze (-1 )

        if squeeze_output :
            raw_actions =raw_actions .squeeze (0 )

        return raw_actions
