"""
Actor network for one charging station.

Input:
- A normalized station observation vector with layout
  `[EV slots flattened][local demand lookahead][current step]`.
- Each EV slot uses `EV_FEAT_DIM` features; feature 0 is the presence flag.

Output:
- Normally one continuous action per EV slot in `[-1, 1]`.
- With `station_total=True`, one scalar in `[-1, 1]` for the station's
  feasible charge or discharge range. An empty station emits zero.
- Positive means charging; negative means discharging.

Architecture:
- Encode each EV slot as a token.
- Pool active EV tokens to summarize station context.
- Concatenate per-EV token, station pooled context, and local tail features.
- Produce per-EV actions or one station-total action with a tanh output head.
"""
import torch
import torch .nn as nn
from Config import ACTOR_HIDDEN_SIZE ,ACTOR_USE_ACTIVE_EV_COUNT
from environment.observation_config import (
EV_FEAT_DIM ,
LOCAL_TAIL_DIM ,
)


class Actor (nn .Module ):
    def __init__ (self ,s_dim ,max_evs_per_station ,hid =ACTOR_HIDDEN_SIZE ,station_state_dim =None ,init_gain =0.1 ,station_total =False ):
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
        self .station_total =bool (station_total )

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
        self .use_active_ev_count =bool (ACTOR_USE_ACTIVE_EV_COUNT )
        count_dim =1 if self .use_active_ev_count else 0
        if self .station_total :
            self .station_action_head =nn .Sequential (
            nn .Linear (set_hid +self .tail_dim +count_dim ,set_hid ),
            nn .LayerNorm (set_hid ),
            nn .LeakyReLU (0.1 ),
            nn .Linear (set_hid ,1 ),
            )
        else :
            self .ev_action_head =nn .Sequential (
            nn .Linear (set_hid *2 +self .tail_dim +count_dim ,set_hid ),
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
            One station-total scalar when `station_total` is set; otherwise
            one value per EV slot.
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

        tail_start =self .ev_state_dim
        tail_end =min (tail_start +self .tail_dim ,x_normalized .size (1 ))
        if tail_end >tail_start :
            tail =x_normalized [:,tail_start :tail_end ]
            if tail .size (1 )<self .tail_dim :
                pad =x_normalized .new_zeros (batch_size ,self .tail_dim -tail .size (1 ))
                tail =torch .cat ([tail ,pad ],dim =1 )
        else :
            tail =x_normalized .new_zeros (batch_size ,self .tail_dim )

        token_feat =self .ev_token_encoder (ev_tokens )

        # Mean-pool only active EV slots. Empty padded slots have presence=0 and
        # therefore do not affect the station-level context vector.
        denom =presence .sum (dim =1 ,keepdim =True ).clamp (min =1.0 )
        pooled =(token_feat *presence ).sum (dim =1 ,keepdim =True )/denom

        if self .station_total :
            station_parts =[pooled .squeeze (1 ),tail ]
            if self .use_active_ev_count :
                station_parts .append (denom .squeeze (1 )/float (max (self .max_evs ,1 )))
            station_input =torch .cat (station_parts ,dim =-1 )
            active =(presence .sum (dim =1 )>0 ).to (dtype =x .dtype )
            station_action =torch .tanh (self .station_action_head (station_input ))*active
            return station_action .squeeze (0 )if squeeze_output else station_action

        pooled_expand =pooled .expand (-1 ,self .max_evs ,-1 )
        tail_expand =tail .unsqueeze (1 ).expand (-1 ,self .max_evs ,-1 )
        head_parts =[token_feat ,pooled_expand ,tail_expand ]
        if self .use_active_ev_count :
            # The divisor the pooling already computed, scaled to [0, 1]. It is
            # the one thing the head cannot recover: it sees one token at a
            # time and a mean that is invariant to how many went into it.
            count =(denom /float (max (self .max_evs ,1 ))).expand (-1 ,self .max_evs ,-1 )
            head_parts .append (count )
        token_input =torch .cat (head_parts ,dim =-1 )

        raw_actions =torch .tanh (self .ev_action_head (token_input ).squeeze (-1 ))
        # Keep padded slots inert. EVEnv applies actions only to active slots,
        # but zeroing here also keeps critic inputs and diagnostics clean.
        raw_actions =raw_actions *presence .squeeze (-1 )

        if squeeze_output :
            raw_actions =raw_actions .squeeze (0 )

        return raw_actions
