
"""
Plotting, TensorBoard, and experiment-output utilities.

This module centralizes artifacts that are produced during training,
checkpoint evaluation, and runtime execution:
- reward and performance CSV/PNG summaries;
- station-level power cooperation plots;
- EV-level SoC trajectory plots;
- demand-tracking mismatch and reward-decomposition diagnostics;
- TensorBoard scalar writers for reward, Q-value, gradient, loss, clipping, and
  replay/exploration state;
- archive snapshots of source code for reproducibility.

Data conventions:
- `all_episode_data` maps episode index -> per-step series. Common keys are
  `ag_requests`, `total_ev_transport`, `actual_ev{station}`, `soc_data`,
  `arrivals_per_step`, and reward component arrays.
- `performance_metrics` stores one list per metric, aligned by episode or
  checkpoint index.
- Plot functions always write CSV outputs where practical, so paper/review
  figures can be regenerated or audited without reading PNG pixels.
"""

import os
import shutil
import csv
import warnings
import numpy as np
import matplotlib
matplotlib .use ("Agg")
import matplotlib .pyplot as plt
import matplotlib as mpl


warnings .filterwarnings ("ignore",category =UserWarning ,module ="matplotlib")
warnings .filterwarnings ("ignore",message ="Glyph .* missing from font")
import logging
logging .getLogger ('matplotlib.font_manager').setLevel (logging .ERROR )
logging .getLogger ('matplotlib.ticker').setLevel (logging .ERROR )

from torch .utils .tensorboard import SummaryWriter
from Config import (
TOL_NARROW_METRICS ,
POWER_TO_ENERGY ,
NUM_STATIONS ,
)


def _savefig_with_retry (fig ,path ,attempts :int =5 ,wait_s :float =0.5 ,**kwargs )->bool :
    """Save a figure over a file another process may briefly hold open.

    Each interim test writes the same history PNG twice within a second, and a
    viewer reloading it or the indexer scanning it makes the second write fail
    with EINVAL/EACCES on Windows. Retry for a moment; if the file stays held,
    report it and leave the old picture so the caller can still write its CSV.
    """
    import sys
    import time
    for attempt in range (attempts ):
        try :
            fig .savefig (path ,**kwargs )
            return True
        except OSError as exc :
            if attempt ==attempts -1 :
                print (f"[plot] could not write {path}: {exc}",file =sys .stderr ,flush =True )
                return False
            time .sleep (wait_s )
    return False


mpl .rcParams ['font.family']='sans-serif'
mpl .rcParams ['font.sans-serif']=['Arial','Helvetica','Liberation Sans','FreeSans','sans-serif']
mpl .rcParams ['font.size']=30
mpl .rcParams ['axes.titlesize']=36
mpl .rcParams ['axes.labelsize']=30
mpl .rcParams ['xtick.labelsize']=24
mpl .rcParams ['ytick.labelsize']=24
mpl .rcParams ['legend.fontsize']=24
mpl .rcParams ['lines.linewidth']=3.0
mpl .rcParams ['axes.linewidth']=3.0
mpl .rcParams ['grid.linewidth']=1.5
mpl .rcParams ['path.simplify']=True
mpl .rcParams ['path.simplify_threshold']=0.5
mpl .rcParams ['agg.path.chunksize']=10000


def dispatch_background_color (dispatch_kw ,intensity :float =1.0 ,alpha_min :float =0.06 ,alpha_max :float =0.40 ):
    """Map aggregate dispatch sign/magnitude to an SoC-plot background tint.

    Color follows sign (charge vs discharge); ``intensity`` is the 0-1 fraction
    of the largest dispatch magnitude in the plotted window, so a bigger
    instruction paints a visibly darker span instead of every nonzero step
    getting the same flat tint. Returns ``(color, alpha)``, or ``(None, 0.0)``
    for a zero instruction.
    """
    value =float (dispatch_kw )
    if value ==0.0 :
        return None ,0.0
    color ="lightgreen"if value >0.0 else "lightcoral"
    frac =min (max (float (intensity ),0.0 ),1.0 )
    alpha =alpha_min +(alpha_max -alpha_min )*frac
    return color ,alpha





def plot_daily_rewards (local_rewards ,
global_rewards ,
results_dir ,
episode_num =None ,
performance_metrics =None ,
title_prefix :str ="",
skip_png :bool =False ,
x_values =None ):
    """
    Save reward curves and the corresponding reward CSV.

    `local_rewards` is the station-averaged local objective, while
    `global_rewards` is the demand-tracking objective. When `x_values` is given,
    the x-axis represents checkpoint episodes rather than dense episode indices.
    """

    base_results_dir =os .path .dirname (results_dir )if "TEST"in os .path .basename (results_dir )else results_dir
    os .makedirs (base_results_dir ,exist_ok =True )
    os .makedirs (results_dir ,exist_ok =True )


    rewards_filename ="episode_rewards_all.png"
    if title_prefix and ("test" in title_prefix .lower ()):
        rewards_filename ="test_episode_rewards_all.png"


    n_local =len (local_rewards )if local_rewards else 0
    n_global =len (global_rewards )if global_rewards else 0
    if x_values is not None :
        x_ep =list (x_values )
    else :
        x_ep =list (range (1 ,max (n_local ,n_global ,1 )+1 ))

    if not skip_png :

        fig ,(ax1 ,ax2 )=plt .subplots (2 ,1 ,figsize =(14 ,12 ))


        if local_rewards and n_local >0 :
            ax1 .plot (x_ep [:n_local ],local_rewards ,label ="Local Reward",linewidth =6 ,color ='green',alpha =0.8 )


            win =20
            if n_local >=win :
                local_mov =[sum (local_rewards [i :i +win ])/win for i in range (n_local -win +1 )]
                ax1 .plot (x_ep [win -1 :n_local ],local_mov ,
                color ='green',label =f"Local MA ({win}ep)",
                linewidth =3 ,linestyle ='--',alpha =0.6 )

        ax1 .axhline (0 ,color ="k",alpha =.3 ,linewidth =3 )
        ax1 .set_xlabel ("Episode",fontsize =24 )
        ax1 .set_ylabel ("Local Reward (Station Avg)",fontsize =24 )

        ax1 .grid (alpha =.3 ,linewidth =3 )
        ax1 .legend (fontsize =18 )


        if global_rewards and n_global >0 :
            ax2 .plot (x_ep [:n_global ],global_rewards ,label ="Global Reward",linewidth =6 ,color ='orange',alpha =0.8 )


            win =20
            if n_global >=win :
                global_mov =[sum (global_rewards [i :i +win ])/win for i in range (n_global -win +1 )]
                ax2 .plot (x_ep [win -1 :n_global ],global_mov ,
                color ='orange',label =f"Global MA ({win}ep)",
                linewidth =3 ,linestyle ='--',alpha =0.6 )

        ax2 .axhline (0 ,color ="k",alpha =.3 ,linewidth =3 )
        ax2 .set_xlabel ("Episode",fontsize =24 )
        ax2 .set_ylabel ("Global Reward (Total)",fontsize =24 )

        ax2 .grid (alpha =.3 ,linewidth =3 )
        ax2 .legend (fontsize =18 )

        fig .tight_layout ()

        fig .savefig (os .path .join (base_results_dir ,rewards_filename ),dpi =140 ,bbox_inches ='tight')
        plt .close (fig )

    csv_filename =rewards_filename .replace ('.png','.csv')
    csv_path =os .path .join (base_results_dir ,csv_filename )
    with open (csv_path ,'w',newline ='',encoding ='utf-8')as f :
        writer =csv .writer (f )
        writer .writerow (['Episode','Local_Reward','Global_Reward'])
        for i in range (n_local ):
            writer .writerow ([x_ep [i ],local_rewards [i ],global_rewards [i ]])







def plot_performance_metrics (performance_metrics ,results_dir ,title_prefix :str ="",x_values =None ,skip_png :bool =False ):
    """
    Save the main policy-quality metrics used in the paper-style evaluation.

    SoC performance is plotted as target-SoC satisfaction rate
    (`100 - soc_miss_count`). Dispatch tracking is the share of assessed steps
    inside the band: up (shortage), down (surplus) and no-instruction steps
    together, as the market assesses them. A row without the no-instruction
    counts (histories written before they were stored) has no such rate and
    is written as NaN rather than as a rate over a subset of the steps.
    """
    if not performance_metrics or len (performance_metrics .get ('soc_miss_count',[]))==0 :
        return


    base_results_dir =os .path .dirname (results_dir )if "TEST"in os .path .basename (results_dir )else results_dir
    os .makedirs (base_results_dir ,exist_ok =True )


    soc_miss_rates =performance_metrics .get ('soc_miss_count',[])


    surplus_steps_list =performance_metrics .get ('surplus_steps',[])
    surplus_within_list =performance_metrics .get ('surplus_within_narrow',[])
    shortage_steps_list =performance_metrics .get ('shortage_steps',[])
    shortage_within_list =performance_metrics .get ('shortage_within_narrow',[])
    zero_steps_list =performance_metrics .get ('zero_request_steps',[])
    zero_within_list =performance_metrics .get ('zero_request_within_narrow',[])


    soc_hit_rates =[100 -rate for rate in soc_miss_rates ]


    def _count_or_none (seq ,idx ):
        if idx >=len (seq )or seq [idx ]is None :
            return None
        return seq [idx ]

    dispatch_tracking_rates =[]
    for i in range (len (soc_hit_rates )):
        s_steps =surplus_steps_list [i ]if i <len (surplus_steps_list )else 0
        s_within =surplus_within_list [i ]if i <len (surplus_within_list )else 0
        sh_steps =shortage_steps_list [i ]if i <len (shortage_steps_list )else 0
        sh_within =shortage_within_list [i ]if i <len (shortage_within_list )else 0
        z_steps =_count_or_none (zero_steps_list ,i )
        z_within =_count_or_none (zero_within_list ,i )
        if z_steps is None or z_within is None :
            dispatch_tracking_rates .append (float ('nan'))
            continue

        denom =s_steps +sh_steps +z_steps
        numer =s_within +sh_within +z_within
        rate =(numer /denom *100.0 )if denom >0 else 0.0
        dispatch_tracking_rates .append (rate )


    if x_values is not None :
        x_plot =np .asarray (x_values )
    else :
        episodes =np .arange (1 ,len (soc_hit_rates )+1 )

        x_plot =episodes


    metrics_filename ="train_performance_metrics.png"
    if title_prefix and ("test" in title_prefix .lower ()):
        metrics_filename ="test_performance_metrics.png"

    if not skip_png :

        plt .rcParams .update ({
        "font.size":24 ,
        "axes.labelsize":26 ,
        "axes.titlesize":28 ,
        "legend.fontsize":20 ,
        "xtick.labelsize":20 ,
        "ytick.labelsize":20 ,
        })

        soc_color ="tab:blue"
        compliance_color ="tab:red"

        fig ,ax1 =plt .subplots (figsize =(12 ,7 ))


        soc_line ,=ax1 .plot (x_plot ,soc_hit_rates ,linestyle ="-",linewidth =5 ,
        label ="Target SoC Satisfaction Rate",color =soc_color )
        ax1 .set_xlabel ("Episode")
        ax1 .set_ylabel ("Target SoC Satisfaction Rate (%)",color =soc_color )
        ax1 .tick_params (axis ="y",labelcolor =soc_color )
        ax1 .set_ylim (0 ,105 )
        ax1 .spines ["left"].set_color (soc_color )


        ax2 =ax1 .twinx ()
        dt_line ,=ax2 .plot (x_plot ,dispatch_tracking_rates ,linestyle ="-",linewidth =5 ,
        label ="Dispatch Tracking Rate",color =compliance_color )
        ax2 .set_ylabel ("Dispatch Tracking Rate (%)",color =compliance_color )
        ax2 .tick_params (axis ="y",labelcolor =compliance_color )
        ax2 .set_ylim (0 ,105 )
        ax2 .spines ["right"].set_color (compliance_color )


        lines =[soc_line ,dt_line ]
        labels =["Target SoC Satisfaction Rate","Dispatch Tracking Rate"]




        ax1 .grid (True ,linewidth =0.7 ,alpha =0.7 )


        if len (x_plot )>10 :
            step =max (1 ,len (x_plot )//10 )
            tick_idx =np .arange (0 ,len (x_plot ),step )
            ax1 .set_xticks (x_plot [tick_idx ])
        else :
            ax1 .set_xticks (x_plot )

        fig .tight_layout ()


        _savefig_with_retry (fig ,os .path .join (base_results_dir ,metrics_filename ),dpi =140 ,bbox_inches ='tight')
        plt .close (fig )


        leg_fig ,leg_ax =plt .subplots (figsize =(8 ,3 ))
        leg_ax .axis ("off")
        leg_ax .legend (lines ,labels ,loc ="center",frameon =True )
        leg_fig .tight_layout ()
        legend_filename =metrics_filename .replace (".png","_legend.png")
        _savefig_with_retry (leg_fig ,os .path .join (base_results_dir ,legend_filename ),dpi =140 )
        plt .close (leg_fig )


    csv_filename =metrics_filename .replace ('.png','.csv')
    csv_path =os .path .join (base_results_dir ,csv_filename )


    departing_evs =performance_metrics .get ('departing_evs',[])
    departing_evs_soc_met =performance_metrics .get ('departing_evs_soc_met',[])
    avg_soc_deficit_list =performance_metrics .get ('avg_soc_deficit',[])
    central_tracking_rate_list =performance_metrics .get ('central_tracking_success_rate',[])
    system_tracking_rate_list =performance_metrics .get ('system_tracking_success_rate',[])
    raw_actor_mae_list =performance_metrics .get ('raw_actor_mae_kw',[])
    pre_bess_mae_list =performance_metrics .get ('pre_bess_mae_kw',[])
    post_bess_mae_list =performance_metrics .get ('post_bess_mae_kw',[])
    corrected_ev_steps_list =performance_metrics .get ('central_corrected_ev_steps',[])
    corrected_station_steps_list =performance_metrics .get ('central_corrected_station_steps',[])
    max_corrected_stations_list =performance_metrics .get ('central_max_corrected_stations_per_step',[])
    station_target_error_list =performance_metrics .get ('central_max_station_target_error_kw',[])
    central_correction_kwh_list =performance_metrics .get ('central_absolute_correction_kwh',[])
    bess_throughput_list =performance_metrics .get ('bess_throughput_kwh',[])

    with open (csv_path ,'w',newline ='',encoding ='utf-8')as f :
        writer =csv .writer (f )
        header =[
        'Episode',
        'SoC_Hit_Rate_%',
        'Dispatch_Tracking_Rate_%',
        'Avg_SoC_Deficit_kWh',
        ]
        header .extend ([
        'SoC_Departing_EVs',
        'SoC_Hit_EVs',
        'Surplus_Steps',
        'Surplus_Steps_Within_Narrow',
        'Shortage_Steps',
        'Shortage_Steps_Within_Narrow',
        'Idle_Steps',
        'Idle_Steps_Within_Narrow',
        'Central_Tracking_Rate_%',
        'System_Tracking_Rate_%',
        'Raw_Actor_MAE_kW',
        'Pre_BESS_MAE_kW',
        'Post_BESS_MAE_kW',
        'Central_Corrected_EV_Steps',
        'Central_Corrected_Station_Steps',
        'Central_Max_Corrected_Stations_Per_Step',
        'Central_Max_Station_Target_Error_kW',
        'Central_Absolute_Correction_kWh',
        'BESS_Throughput_kWh',
        ])
        writer .writerow (header )

        def _safe_get (seq ,idx ):
            return seq [idx ]if (isinstance (seq ,(list ,tuple ,np .ndarray ))and idx <len (seq ))else ""

        for i in range (len (soc_hit_rates )):
            row =[
            i +1 ,
            soc_hit_rates [i ],
            dispatch_tracking_rates [i ],
            _safe_get (avg_soc_deficit_list ,i ),
            ]
            row .extend ([
            _safe_get (departing_evs ,i ),
            _safe_get (departing_evs_soc_met ,i ),
            _safe_get (surplus_steps_list ,i ),
            _safe_get (surplus_within_list ,i ),
            _safe_get (shortage_steps_list ,i ),
            _safe_get (shortage_within_list ,i ),
            _safe_get (zero_steps_list ,i ),
            _safe_get (zero_within_list ,i ),
            _safe_get (central_tracking_rate_list ,i ),
            _safe_get (system_tracking_rate_list ,i ),
            _safe_get (raw_actor_mae_list ,i ),
            _safe_get (pre_bess_mae_list ,i ),
            _safe_get (post_bess_mae_list ,i ),
            _safe_get (corrected_ev_steps_list ,i ),
            _safe_get (corrected_station_steps_list ,i ),
            _safe_get (max_corrected_stations_list ,i ),
            _safe_get (station_target_error_list ,i ),
            _safe_get (central_correction_kwh_list ,i ),
            _safe_get (bess_throughput_list ,i ),
            ])
            writer .writerow (row )





def plot_station_cooperation_full (all_episode_data :dict ,
results_dir :str ,
random_window :bool =False ,
title_prefix :str ="")->None :
    """
    Visualize how stations collectively satisfy the grid request over time.

    For each episode, station powers are stacked by sign so charging and
    discharging contributions remain visible. The grid request and total station
    power are overlaid with the dispatch tolerance band.
    """
    title_prefix_en =title_prefix


    episode_keys =sorted (all_episode_data .keys ())
    if len (episode_keys )>10 :
        episode_keys =episode_keys [-10 :]

    for ep_key in episode_keys :
        ep =all_episode_data [ep_key ]
        total_steps =len (ep ["ag_requests"])


        start =0
        end =total_steps
        rng =np .arange (end -start )


        num_stations =0
        while f"actual_ev{num_stations+1}"in ep :
            num_stations +=1
        if num_stations ==0 :
            raise ValueError ("Visualization requires actual_ev* series. Found none in episode data.")


        stations =[]
        for i in range (1 ,num_stations +1 ):
            key =f"actual_ev{i}"
            if key not in ep :
                raise KeyError (f"Missing required series '{key}' in episode data")
            arr =np .asarray (ep [key ][start :end ],dtype =float )
            stations .append (arr )

        ag_req =np .asarray (ep ["ag_requests"][start :end ],dtype =float )


        if "total_ev_transport"not in ep :
            raise KeyError ("Visualization requires 'total_ev_transport' series. Found none in episode data.")
        total_station_power =np .asarray (ep ["total_ev_transport"][start :end ],dtype =float )

        # What the learner itself emitted, before the central residual
        # allocator rewrote the actions. The stacked bars and the total above
        # are the physical outcome and so include that rewrite; without this
        # line a plot of a run whose actor tracks at 84% looks like one that
        # tracks at 98%. Absent for episode data saved before it was recorded.
        raw_actor_raw =ep .get ("raw_actor_total_power_kw")
        raw_actor_power =(
        None if raw_actor_raw is None or len (raw_actor_raw )==0
        else np .asarray (raw_actor_raw [start :end ],dtype =float )
        )
        if raw_actor_power is not None and raw_actor_power .size <len (rng ):
            raw_actor_power =None




        tol_raw =ep .get ("tol_narrow",TOL_NARROW_METRICS )
        if isinstance (tol_raw ,(list ,tuple ,np .ndarray )):
            tol_narrow =np .asarray (tol_raw [start :end ],dtype =float )
            if tol_narrow .size <len (rng ):
                fill =float (tol_narrow [-1 ])if tol_narrow .size else float (TOL_NARROW_METRICS )
                tol_narrow =np .pad (tol_narrow ,(0 ,len (rng )-tol_narrow .size ),constant_values =fill )
        else :
            tol_narrow =np .full (len (rng ),float (tol_raw ),dtype =float )

        tracking_raw =ep .get ("tracking_enabled")
        if tracking_raw is None :
            tracking_enabled =np .ones (len (rng ),dtype =bool )
        else :
            tracking_enabled =np .asarray (tracking_raw [start :end ],dtype =bool )
            if tracking_enabled .size <len (rng ):
                tracking_enabled =np .pad (
                tracking_enabled ,(0 ,len (rng )-tracking_enabled .size ),constant_values =True
                )

        # What the bid actually committed to: baseline plus the full up/down
        # award either side of it. The instruction can never fall outside this
        # -- it is what the previous questions were checking by hand. Absent
        # for episode data saved before this was tracked, so drawing it is
        # opt-in on all three arrays being present.
        baseline_raw =ep .get ("baseline")
        envelope_lower_raw =ep .get ("bid_envelope_lower")
        envelope_upper_raw =ep .get ("bid_envelope_upper")
        has_envelope =bool (
        baseline_raw is not None and envelope_lower_raw is not None
        and envelope_upper_raw is not None and len (baseline_raw )>0
        )
        if has_envelope :
            baseline_series =np .asarray (baseline_raw [start :end ],dtype =float )
            envelope_lower =np .asarray (envelope_lower_raw [start :end ],dtype =float )
            envelope_upper =np .asarray (envelope_upper_raw [start :end ],dtype =float )

            # Instruction on or off: the target sits on baseline while nothing
            # is being asked for, so any departure from it is an instruction.
            # The target is recorded float32, so "on baseline" means within a
            # few ULPs, not exact -- at 1e3 kW that is ~1e-4, and a fixed 1e-6
            # test paints idle steps as live instructions. Direction is kept
            # for the CSV; the plot only needs the on/off.
            regulation =ag_req -baseline_series
            idle_eps =np .maximum (1e-3 ,1e-6 *np .abs (baseline_series ))
            up_mask =regulation <-idle_eps
            down_mask =regulation >idle_eps
            instruction_on =up_mask |down_mask

        # Per-step tracking verdict: does the achieved power sit inside the
        # tolerance box already drawn around the request.
        step_pass =np .abs (total_station_power -ag_req )<=tol_narrow +1e-6



        fig ,ax =plt .subplots (figsize =(20 ,12 ))
        width =.9

        labels =[f"Station {i}"for i in range (1 ,num_stations +1 )]


        colors =plt .cm .tab20 (np .linspace (0 ,1 ,num_stations ))
        if num_stations >20 :

            cmap1 =plt .cm .tab20 (np .linspace (0 ,1 ,20 ))
            cmap2 =plt .cm .tab20b (np .linspace (0 ,1 ,20 ))
            cmap3 =plt .cm .tab20c (np .linspace (0 ,1 ,20 ))
            cmap4 =plt .cm .Set3 (np .linspace (0 ,1 ,20 ))
            cmap5 =plt .cm .Pastel1 (np .linspace (0 ,1 ,20 ))
            colors =np .vstack ([cmap1 ,cmap2 ,cmap3 ,cmap4 ,cmap5 ])[:num_stations ]


        pos_bottoms =np .zeros_like (rng ,dtype =float )
        neg_bottoms =np .zeros_like (rng ,dtype =float )
        for s ,lbl ,col in zip (stations ,labels ,colors ):
            s =np .asarray (s ,dtype =float )
            pos =np .clip (s ,0 ,None )
            neg =np .clip (s ,None ,0 )
            pos_drawn =np .any (pos !=0 )
            neg_drawn =np .any (neg !=0 )


            if pos_drawn :
                ax .bar (rng ,pos ,width ,bottom =pos_bottoms ,label =lbl ,color =col ,alpha =.7 )


            if neg_drawn :
                label_for_neg =lbl if not pos_drawn else "_nolegend_"
                ax .bar (rng ,neg ,width ,bottom =neg_bottoms ,label =label_for_neg ,color =col ,alpha =.7 )


            if not pos_drawn and not neg_drawn :
                ax .plot ([],[],color =col ,label =lbl ,linestyle ='-')

            pos_bottoms +=pos
            neg_bottoms +=neg





        if has_envelope :
            # The bid's full up/down range for this block, in a color family
            # (blue) that neither the black request line, the gray tolerance
            # fill, nor the orange unawarded spans use, so it cannot be
            # mistaken for either. Low zorder keeps it a backdrop: the request
            # line staying inside it is the thing to check, not a foreground
            # signal competing with the request/tolerance/station layers.
            ax .fill_between (rng ,envelope_lower ,envelope_upper ,step ='mid',
            color ="tab:blue",alpha =0.08 ,edgecolor ='none',zorder =0.5 ,
            label ="Bid envelope (baseline±award)")
            ax .step (rng ,baseline_series ,where ='mid',color ="tab:blue",
            lw =1.6 ,alpha =0.8 ,zorder =1 ,label ="Baseline")

        # Tracking well means these two coincide, so the pair has to stay
        # readable exactly where it matters most. The request is a thin pale
        # underlay with a thin dashed core; the achieved power is a narrow
        # opaque line on top. Where they agree the magenta sits inside the grey
        # band with the dashes showing through, and neither one disappears.
        # The underlay is a fixed few points wide regardless of data, so it
        # must stay visibly narrower than the real tolerance-band fill below
        # or it reads as a second, competing band around the line it is only
        # meant to make legible.
        ax .step (rng ,ag_req ,where ='mid',color ="#111111",lw =3 ,alpha =0.16 ,
        label ="Grid Request",zorder =2 )
        ax .step (rng ,ag_req ,where ='mid',color ="#111111",lw =1.3 ,alpha =0.85 ,
        linestyle =(0 ,(4 ,3 )),zorder =4 )
        ax .plot (rng ,total_station_power ,color ="m",lw =2.2 ,alpha =0.95 ,
        label ="Total Station Power",zorder =5 )

        if raw_actor_power is not None :
            # Dashed and thinner than the physical total, so where the
            # allocator did nothing the two coincide and only the magenta
            # reads; where it intervened the gap is the intervention.
            ax .plot (rng ,raw_actor_power ,color ="#1a7f37",lw =1.5 ,alpha =0.9 ,
            linestyle =(0 ,(5 ,2 )),zorder =4.5 ,
            label ="MARL actor total (before central allocator)")



        for i ,step_idx in enumerate (rng ):
            step_req =ag_req [i ]
            step_upper =step_req +tol_narrow [i ]
            step_lower =step_req -tol_narrow [i ]


            ax .fill_between ([step_idx -0.4 ,step_idx +0.4 ],
            [step_lower ,step_lower ],
            [step_upper ,step_upper ],
            color ='gray',alpha =0.25 ,edgecolor ='none')

        unawarded =np .logical_not (tracking_enabled )
        unawarded_starts =np .flatnonzero (unawarded &np .r_ [True ,~unawarded [:-1 ]])
        unawarded_stops =np .flatnonzero (unawarded &np .r_ [~unawarded [1 :],True ])+1
        for span_start ,span_stop in zip (unawarded_starts ,unawarded_stops ):
            ax .axvspan (
            float (span_start )-0.5 ,float (span_stop )-0.5 ,
            color ='#E69F00',alpha =0.18 ,linewidth =0 ,zorder =0
            )


        import matplotlib .patches as mpatches
        narrow_patch =mpatches .Patch (color ='gray',alpha =0.25 ,label ="Tolerance band")
        unawarded_patch =mpatches .Patch (
        color ='#E69F00',alpha =0.18 ,label ="No awarded capacity (not assessed)"
        )

        instruction_patch =None
        if has_envelope and np .any (instruction_on ):
            # Tint the envelope band itself rather than adding a third
            # full-height wash: a plain pale-blue band means the fleet is only
            # holding baseline, a green-tinted one means something is actually
            # being asked of it. Same geometry, so the eye compares color
            # alone, and the two states cannot drift apart.
            ax .fill_between (rng ,envelope_lower ,envelope_upper ,step ='mid',
            where =instruction_on ,color ="#2ca02c",alpha =0.20 ,
            edgecolor ='none',zorder =0.6 )
            instruction_patch =mpatches .Patch (
            color ="#2ca02c",alpha =0.20 ,label ="Instruction on"
            )

        fail_step_mask =(~step_pass )&tracking_enabled
        if np .any (fail_step_mask ):
            ax .plot (
            rng [fail_step_mask ],total_station_power [fail_step_mask ],
            linestyle ='None',marker ='x',markersize =11 ,markeredgewidth =2.5 ,
            color ='red',alpha =0.5 ,zorder =6 ,
            label ="Tracking miss (step outside band)",
            )

        ax .axhline (0 ,color ="k",alpha =.3 ,linewidth =3 )
        tick_indices =np .arange (0 ,len (rng ),24 ,dtype =int )
        if len (rng )and (not len (tick_indices )or tick_indices [-1 ]!=len (rng )-1 ):
            tick_indices =np .append (tick_indices ,len (rng )-1 )
        ax .set_xticks (rng [tick_indices ])
        ax .set_xticklabels ([start +int (i )+1 for i in tick_indices ])
        ax .set_xlabel ("Step")
        ax .set_ylabel ("Power ")



        ax .grid (alpha =.3 ,linewidth =3 )


        # Patch-only legend entries (never drawn as a single labeled artist,
        # so get_legend_handles_labels() below cannot find them on its own).
        # Tracking-miss markers and the request/baseline/envelope lines are
        # real labeled artists and are already in that call's output.
        extra_handles =[narrow_patch ]
        extra_labels =["Tolerance band"]
        if np .any (unawarded ):
            extra_handles .append (unawarded_patch )
            extra_labels .append ("No awarded capacity (not assessed)")
        if instruction_patch is not None :
            extra_handles .append (instruction_patch )
            extra_labels .append (instruction_patch .get_label ())

        if num_stations <=10 :

            handles ,labels =ax .get_legend_handles_labels ()
            handles =handles +extra_handles
            labels =labels +extra_labels
            ax .legend (handles =handles ,labels =labels ,loc ="upper left",fontsize =16 )
        else :

            handles ,labels =ax .get_legend_handles_labels ()

            # Station bars are BarContainers and the Grid Request/Total Station
            # Power lines are Line2D artists; matplotlib's own legend-handle
            # order groups by artist type, not draw order, so it lists the two
            # lines before any bar regardless of where num_stations is. Picking
            # "the first 3 slots" as a stand-in for "3 station samples" used to
            # assume station entries start at index 0, which put Grid
            # Request/Total Station Power right back in as duplicates. Select
            # each side by its actual label instead.
            non_station_items =[i for i ,lbl in enumerate (labels )if not lbl .startswith ("Station")]
            station_items =[i for i ,lbl in enumerate (labels )if lbl .startswith ("Station")][:3 ]
            selected_items =non_station_items +station_items


            selected_handles =[handles [i ]for i in selected_items ]+extra_handles
            selected_labels =[labels [i ]for i in selected_items ]+extra_labels

            ax .legend (handles =selected_handles ,labels =selected_labels ,loc ="upper left",fontsize =16 )


        fig .tight_layout ()




        base_fname =f"zz_station_cooperation_full_episode_{ep_key}"+(
        "_random_window"if random_window else ""
        )
        fname =base_fname
        if title_prefix :
            fname =f"{title_prefix_en.lower().replace(' ', '_')}_{base_fname}"
        fig .savefig (os .path .join (results_dir ,f"{fname}.png"))
        plt .close (fig )


        base_csv_fname =f"zz_station_cooperation_full_episode_{ep_key}"
        csv_fname =base_csv_fname
        if title_prefix :
            csv_fname =f"{title_prefix_en.lower().replace(' ', '_')}_{base_csv_fname}"
        csv_filename =f"{csv_fname}.csv"
        csv_path =os .path .join (results_dir ,csv_filename )


        all_ag_req =np .asarray (ep ["ag_requests"],dtype =float )
        all_stations =[]
        for i in range (1 ,num_stations +1 ):
            key =f"actual_ev{i}"
            all_stations .append (np .asarray (ep [key ],dtype =float ))
        all_total_station_power =np .asarray (ep ["total_ev_transport"],dtype =float )

        with open (csv_path ,'w',newline ='',encoding ='utf-8')as f :
            writer =csv .writer (f )

            all_tol_raw =ep .get ("tol_narrow",TOL_NARROW_METRICS )
            if isinstance (all_tol_raw ,(list ,tuple ,np .ndarray )):
                all_tol =np .asarray (all_tol_raw ,dtype =float )
                if all_tol .size <total_steps :
                    fill =float (all_tol [-1 ])if all_tol .size else float (TOL_NARROW_METRICS )
                    all_tol =np .pad (all_tol ,(0 ,total_steps -all_tol .size ),constant_values =fill )
            else :
                all_tol =np .full (total_steps ,float (all_tol_raw ),dtype =float )
            all_tracking =np .asarray (
            ep .get ("tracking_enabled",np .ones (total_steps ,dtype =bool )),dtype =bool
            ).reshape (-1 )
            if all_tracking .size <total_steps :
                all_tracking =np .pad (
                all_tracking ,(0 ,total_steps -all_tracking .size ),constant_values =True
                )

            def _padded_or_nan (key ):
                raw =ep .get (key )
                if raw is None or len (raw )==0 :
                    return np .full (total_steps ,np .nan ,dtype =float )
                arr =np .asarray (raw ,dtype =float )
                if arr .size <total_steps :
                    arr =np .pad (arr ,(0 ,total_steps -arr .size ),constant_values =np .nan )
                return arr
            all_raw_actor =_padded_or_nan ("raw_actor_total_power_kw")
            all_baseline =_padded_or_nan ("baseline")
            all_envelope_lower =_padded_or_nan ("bid_envelope_lower")
            all_envelope_upper =_padded_or_nan ("bid_envelope_upper")

            # rng spans the whole episode (start=0, end=total_steps above), so
            # step_pass and the instruction masks already cover every row here.
            all_step_state =np .full (total_steps ,"",dtype =object )
            if has_envelope :
                all_step_state =np .where (up_mask ,"up",np .where (down_mask ,"down","idle"))

            header =(
            ['Step','Grid_Request','Tolerance_kW','Tracking_Enabled',
            'Baseline_kW','Bid_Envelope_Lower_kW','Bid_Envelope_Upper_kW',
            'Instruction_State','Step_Tracking_Pass']
            +[f'Station_{i}'for i in range (1 ,num_stations +1 )]
            +['Total_Station_Power','Raw_Actor_Total_Power_kW']
            )
            writer .writerow (header )

            for i in range (total_steps ):
                row =[
                i +1 ,all_ag_req [i ],all_tol [i ],int (all_tracking [i ]),
                all_baseline [i ],all_envelope_lower [i ],all_envelope_upper [i ],
                all_step_state [i ],int (step_pass [i ]),
                ]
                for s in all_stations :
                    row .append (s [i ])
                row .append (all_total_station_power [i ])
                row .append (all_raw_actor [i ])
                writer .writerow (row )





def plot_ev_detailed_soc (all_episode_data :dict ,
results_dir :str ,
display_steps :int =48 ,
random_window :bool =False ,
title_prefix :str ="")->None :
    """
    Plot representative EV SoC trajectories against their target SoC.

    The function selects long-dwell EVs from recent episodes so reviewers can
    inspect whether the policy gradually closes charging deficits before
    departure, rather than only checking aggregate hit rates.
    """
    import random


    title_prefix_en =title_prefix


    episode_keys =sorted (all_episode_data .keys ())
    if len (episode_keys )>2 :
        episode_keys =episode_keys [-2 :]

    for ep_key in episode_keys :
        ep =all_episode_data [ep_key ]
        total_steps =len (ep ["ag_requests"])

        soc_data =ep .get ("soc_data",{})
        if not soc_data :
            continue


        long_stay_evs =find_long_stay_evs (soc_data ,min_stay =10 ,max_evs =3 )

        for idx ,long_stay_ev in enumerate (long_stay_evs ):

            long_stay_fname =f"test_results_ev_soc_long_stay_{idx+1}_episode_{ep_key}"
            if title_prefix :
                prefix =title_prefix_en .lower ().replace (' ','_')
                if not long_stay_fname .startswith (f"{prefix}_"):
                    long_stay_fname =f"{prefix}_{long_stay_fname}"

            if long_stay_ev :
                station_id ,ev_id ,ev_data =long_stay_ev


                csv_filename =f"{long_stay_fname}.csv"
                csv_path =os .path .join (results_dir ,csv_filename )
                with open (csv_path ,'w',newline ='',encoding ='utf-8')as f :
                    writer =csv .writer (f )
                    writer .writerow (['Step','EV_ID','Station','SoC_%','Target_%','Departure_Step'])
                    ts =np .asarray (ev_data ["times"])
                    soc_values =np .asarray (ev_data ["soc"])

                    for j ,step in enumerate (ts ):
                        if j <len (soc_values ):
                            writer .writerow ([step ,ev_id ,station_id ,soc_values [j ],ev_data ['target'],ev_data .get ('depart','N/A')])


                fig_long ,ax_long =plt .subplots (figsize =(14 ,8 ))
                soc_color ="tab:blue"


                ts =np .asarray (ev_data ["times"],dtype =float )
                soc_values =np .asarray (ev_data ["soc"],dtype =float )
                departure_time =ev_data .get ("depart_step",ev_data .get ("depart",None ))

                # Window = this EV's own stay, arrival to departure, not the
                # fixed 288-step episode. Every EV that appears in this figure
                # is only ever connected during this span, so a step outside it
                # cannot be attributed to this EV either way.
                arrival_time =float (np .min (ts ))if ts .size else 0.0
                last_time =(
                float (departure_time )if departure_time is not None
                else float (np .max (ts ))if ts .size else arrival_time
                )
                start =int (max (0 ,np .floor (arrival_time )))
                end =int (min (total_steps ,np .floor (last_time )+1 ))
                end =max (end ,start +1 )

                ag_req =np .asarray (ep ["ag_requests"][start :end ],dtype =float )
                # Intensity is relative to the loudest instruction during this
                # EV's own stay, so the shading still spans light-to-dark even
                # when the whole day's peak instruction falls outside the window.
                max_abs =float (np .max (np .abs (ag_req )))if ag_req .size else 0.0

                # Positive EV power charges the fleet; negative power discharges it.
                for i in range (len (ag_req )):
                    x =start +i
                    value =ag_req [i ]
                    intensity =abs (value )/max_abs if max_abs >1e-9 else 0.0
                    background_color ,alpha =dispatch_background_color (value ,intensity )
                    if background_color is not None :
                        ax_long .axvspan (
                        x -0.5 ,x +0.5 ,color =background_color ,alpha =alpha
                        )

                mask =(ts >=start )&(ts <end )
                ts_plot =ts [mask ]
                soc_plot =soc_values [mask ]


                target_soc =float (ev_data .get ("target",np .nan ))
                target_achieved =True
                depart_soc =None
                if departure_time is not None :
                    dep_idx =np .where (ts ==float (departure_time ))[0 ]
                    if dep_idx .size >0 :
                        depart_soc =float (soc_values [int (dep_idx [0 ])])
                    elif soc_plot .size >0 :
                        depart_soc =float (soc_plot [-1 ])
                    if np .isfinite (target_soc )and depart_soc is not None :
                        target_achieved =(depart_soc >=target_soc )

                line_style ="-"if target_achieved else "--"
                ax_long .plot (
                ts_plot ,soc_plot ,
                color =soc_color ,lw =6 ,ls =line_style ,
                label =f"EV {ev_id} (Target {target_soc:.0f}%)"
                )


                if np .isfinite (target_soc ):
                    ax_long .axhline (target_soc ,color ="green",ls ="--",alpha =0.6 ,lw =4.5 ,label ="Target SoC")


                if departure_time is not None and start <=float (departure_time )<end :
                    ax_long .axvline (float (departure_time ),color =soc_color ,ls ="--",alpha =0.8 ,lw =4.5 )
                    if depart_soc is not None :
                        ax_long .plot ([float (departure_time )],[depart_soc ],'o',color =soc_color ,markersize =8 *3 )
                        if np .isfinite (target_soc ):
                            ax_long .plot ([float (departure_time )],[target_soc ],'s',color ="green",markersize =6 *3 ,alpha =0.85 )
                            label_txt ="Goal met!"if depart_soc >=target_soc else "Goal not met"
                            label_col ="green"if depart_soc >=target_soc else "red"
                            ax_long .text (float (departure_time ),depart_soc +5.0 ,label_txt ,color =label_col ,ha ="center",fontsize =9 *3 )


                ax_long .set_title (
                f"Long Stay EV #{idx+1} - {station_id} / EV {ev_id}",
                fontsize =30
                )
                ax_long .set_xlabel ("Time Step",fontsize =22 )
                ax_long .set_ylabel ("SoC [%]",fontsize =22 ,color =soc_color )
                ax_long .tick_params (axis ='both',labelsize =16 )
                ax_long .tick_params (axis ='y',labelcolor =soc_color )
                ax_long .set_xlim (start ,max (end -1 ,start +1 ))
                ax_long .set_ylim (0 ,105 )
                ax_long .grid (alpha =0.3 ,linewidth =2 )

                h1 ,l1 =ax_long .get_legend_handles_labels ()
                h2 =[
                mpl .patches .Patch (color ="lightgreen",alpha =0.35 ,label ="Charge request (+)"),
                mpl .patches .Patch (color ="lightcoral",alpha =0.35 ,label ="Discharge request (-)"),
                ]
                l2 =["Charge request (+)","Discharge request (-)"]
                if h1 or h2 :
                    ax_long .legend (h1 +h2 ,l1 +l2 ,loc ="upper left",fontsize =14 )

                fig_long .tight_layout ()
                out_png =os .path .join (results_dir ,f"{long_stay_fname}.png")
                fig_long .savefig (out_png ,dpi =180 ,bbox_inches ='tight')
                plt .close (fig_long )



        max_stay_duration =0
        if soc_data :
            for station_evs in soc_data .values ():
                for ev in station_evs .values ():
                    if "times"in ev and len (ev ["times"])>0 :
                        ev_start =min (ev ["times"])
                        ev_end =max (ev ["times"])
                        stay_duration =ev_end -ev_start
                        max_stay_duration =max (max_stay_duration ,stay_duration )


        fig_random ,ax_random =plt .subplots (figsize =(14 ,8 ))


        soc_color ="tab:blue"


        if soc_data :
            # A station can hold zero EVs this episode; try others before
            # settling for the "no data" placeholder.
            station_keys =list (soc_data .keys ())
            random .shuffle (station_keys )
            station_key =station_keys [0 ]
            evs ={}
            for candidate in station_keys :
                if soc_data [candidate ]:
                    station_key =candidate
                    evs =soc_data [candidate ]
                    break


            valid_evs =evs
            ev_ids_to_show =list (evs .keys ())
            if len (ev_ids_to_show )>5 :
                random .shuffle (ev_ids_to_show )
                ev_ids_to_show =ev_ids_to_show [:5 ]

            # Window = arrival of the earliest shown EV to the departure of the
            # last, not the fixed 288-step episode. Picking the window from the
            # EVs actually drawn (rather than the other way around) keeps every
            # shown trajectory fully inside the plotted span.
            window_starts ,window_ends =[],[]
            for ev_id in ev_ids_to_show :
                ev_ts =np .asarray (evs [ev_id ]["times"],dtype =float )
                if ev_ts .size ==0 :
                    continue
                window_starts .append (float (np .min (ev_ts )))
                ev_depart =evs [ev_id ].get ("depart_step",evs [ev_id ].get ("depart",None ))
                window_ends .append (
                float (ev_depart )if ev_depart is not None else float (np .max (ev_ts ))
                )
            if window_starts :
                start =int (max (0 ,np .floor (min (window_starts ))))
                end =int (min (total_steps ,np .floor (max (window_ends ))+1 ))
                end =max (end ,start +1 )
            else :
                start =0
                end =min (total_steps ,display_steps )


            ag_req =np .asarray (ep ["ag_requests"][start :end ],dtype =float )
            max_abs =float (np .max (np .abs (ag_req )))if ag_req .size else 0.0


            for i in range (len (ag_req )):
                value =ag_req [i ]
                intensity =abs (value )/max_abs if max_abs >1e-9 else 0.0
                background_color ,alpha =dispatch_background_color (value ,intensity )
                if background_color is not None :
                    ax_random .axvspan (
                    start +i -0.5 ,start +i +0.5 ,
                    color =background_color ,alpha =alpha
                    )


            if valid_evs and ev_ids_to_show :


                colors =plt .cm .tab10 (np .linspace (0 ,1 ,len (ev_ids_to_show )))


                for i ,ev_id in enumerate (ev_ids_to_show ):
                    ev =valid_evs [ev_id ]
                    ts =np .asarray (ev ["times"])
                    soc_vals =np .asarray (ev ["soc"])
                    color =colors [i ]


                    target_achieved =True
                    if "soc"in ev and "depart"in ev and "target"in ev :
                        depart_idx =[j for j ,t in enumerate (ev ["times"])if t ==ev ["depart"]]
                        if depart_idx and ev ["soc"][depart_idx [0 ]]<ev ["target"]:
                            target_achieved =False

                    line_style ="--"if not target_achieved else "-"
                    line_color =color


                    if len (ts )>0 and len (soc_vals )>0 :
                        ax_random .plot (ts ,soc_vals ,
                        label =f"EV {ev_id} (Target {ev['target']:.0f}%)",
                        color =line_color ,lw =6 ,ls =line_style )


                    depart_time_rand =ev .get ("depart_step",ev .get ("depart",None ))
                    if depart_time_rand is not None :
                        ax_random .axvline (depart_time_rand ,color =color ,ls ="--",alpha =0.7 ,lw =4.5 )

                        depart_idx =[j for j ,t in enumerate (ev ["times"])if t ==depart_time_rand ]

                        depart_soc =None
                        if "final_soc"in ev :

                            depart_soc =ev ["final_soc"]

                            ax_random .plot ([depart_time_rand ],[depart_soc ],'o',color =color ,markersize =8 *3 )

                            if "target_soc"in ev :
                                ax_random .axhline (ev ["target_soc"],color ="green",ls ="--",alpha =0.3 ,lw =4.5 )
                                ax_random .plot ([depart_time_rand ],[ev ["target_soc"]],'s',color ="green",markersize =6 *3 ,alpha =0.7 )

                                if depart_soc >=ev ["target_soc"]:
                                    ax_random .text (depart_time_rand ,depart_soc +5 ,"Goal met!",color ="green",ha ="center",fontsize =9 *3 )
                                else :
                                    ax_random .text (depart_time_rand ,depart_soc +5 ,"Goal not met",color ="red",ha ="center",fontsize =9 *3 )
                        elif depart_idx :

                            depart_soc =ev ["soc"][depart_idx [0 ]]
                            ax_random .plot ([depart_time_rand ],[depart_soc ],'o',color =color ,markersize =6 *3 )

                            if "target"in ev :
                                target_soc =ev ["target"]
                                ax_random .axhline (target_soc ,color ="green",ls ="--",alpha =0.3 ,lw =4.5 )
                                ax_random .plot ([depart_time_rand ],[target_soc ],'s',color ="green",markersize =6 *3 ,alpha =0.7 )

                                if depart_soc >=target_soc :
                                    ax_random .text (depart_time_rand ,depart_soc +5 ,"Goal met!",color ="green",ha ="center",fontsize =9 *3 )
                                else :
                                    ax_random .text (depart_time_rand ,depart_soc +5 ,"Goal not met",color ="red",ha ="center",fontsize =9 *3 )

                        if depart_soc is not None and not any (ts ==depart_time_rand ):

                            plot_ts =np .append (ts ,[depart_time_rand ])
                            plot_soc =np .append (soc_vals ,[depart_soc ])

                            sort_idx =np .argsort (plot_ts )
                            ax_random .plot (plot_ts [sort_idx ],plot_soc [sort_idx ],color =line_color ,lw =6 ,ls =line_style ,alpha =0.7 )


                ax_random .spines ["left"].set_color (soc_color )
                ax_random .spines ["bottom"].set_color ("black")
                ax_random .tick_params (axis ="x",colors ="black")
                ax_random .tick_params (axis ="y",labelcolor =soc_color )



                ax_random .set_xlim (start ,end -1 )
            else :
                ax_random .text (0.5 ,0.5 ,"No EVs connected in this time window",
                ha ='center',va ='center',transform =ax_random .transAxes ,fontsize =24 )
        else :
            ax_random .text (0.5 ,0.5 ,"No station data available",
            ha ='center',va ='center',transform =ax_random .transAxes ,fontsize =24 )

        ax_random .set_ylabel ("SoC [%]",color =soc_color )
        ax_random .set_ylim (0 ,105 )
        ax_random .grid (alpha =.3 ,linewidth =3 )

        lines1 ,labels1 =ax_random .get_legend_handles_labels ()
        background_handles =[
        mpl .patches .Patch (color ="lightgreen",alpha =0.35 ,label ="Charge request (+)"),
        mpl .patches .Patch (color ="lightcoral",alpha =0.35 ,label ="Discharge request (-)"),
        ]
        background_labels =["Charge request (+)","Discharge request (-)"]
        ax_random .legend (
        lines1 +background_handles ,labels1 +background_labels ,loc ="best"
        )


        fig_random .tight_layout ()


        random_window_fname =f"ev_soc_random_window_episode_{ep_key}"
        if title_prefix :
            random_window_fname =f"{title_prefix_en.lower().replace(' ', '_')}_{random_window_fname}"
        fig_random .savefig (os .path .join (results_dir ,f"{random_window_fname}.png"))
        plt .close (fig_random )


        if soc_data and valid_evs :
            csv_filename =f"{random_window_fname}.csv"
            csv_path =os .path .join (results_dir ,csv_filename )
            with open (csv_path ,'w',newline ='',encoding ='utf-8')as f :
                writer =csv .writer (f )
                writer .writerow (['Step','EV_ID','Station','SoC_%','Target_%'])
                for ev_id in (ev_ids_to_show if 'ev_ids_to_show'in locals ()and ev_ids_to_show else valid_evs .keys ()):
                    ev =valid_evs [ev_id ]
                    ts =np .asarray (ev ["times"])
                    soc_vals =np .asarray (ev ["soc"])

                    for j ,step in enumerate (ts ):
                        if j <len (soc_vals ):
                            writer .writerow ([step ,ev_id ,station_key ,soc_vals [j ],ev ['target']])



        pass



def find_long_stay_evs (soc_data ,min_stay =10 ,max_evs =3 ):
    """Select representative long-dwell EVs from `soc_data` for SoC plots."""
    long_stay_evs =[]
    for station_id ,evs in sorted (soc_data .items ()):
        for ev_id ,ev in evs .items ():
            if "times"in ev and "depart"in ev :
                arrival_time =min (ev ["times"])
                departure_time =ev ["depart"]
                stay_duration =departure_time -arrival_time
                if stay_duration >=min_stay :
                    long_stay_evs .append ((station_id ,ev_id ,ev ))


    return long_stay_evs [:max_evs ]if len (long_stay_evs )>max_evs else long_stay_evs

def create_tensorboard_writer (log_dir ="temp/timing",comment =None ,purge_step =None ,max_queue =10 ,flush_secs =120 ,filename_suffix =''):
    """Create a TensorBoard SummaryWriter after ensuring the log directory exists."""

    os .makedirs (log_dir ,exist_ok =True )


    return SummaryWriter (
    log_dir =log_dir ,
    comment =comment ,
    purge_step =purge_step ,
    max_queue =max_queue ,
    flush_secs =flush_secs ,
    filename_suffix =filename_suffix
    )







class GradientLossVisualizer :
    """
    Accumulate training diagnostics during an episode and emit TensorBoard means.

    The trainer calls `update_*()` once per environment step. At episode end,
    `record_to_tensorboard()` writes averaged Q values, gradient norms, losses,
    and clipping counts. The class supports both distributed multi-agent critics
    and centralized-joint baselines by switching layout based on agent metadata.
    """

    def __init__ (self ,num_stations ,tb_writer =None ):
        """Initialize per-episode accumulators for `num_stations` agents."""
        self .num_stations =num_stations
        self .tb_writer =tb_writer
        self .mode =None
        self .reset_episode_data ()

    def _set_mode (self ,mode ):
        if self .mode is None :
            self .mode =mode
        elif self .mode !=mode :
            self .mode =mode

    def update_q_values (self ,q_values_per_agent ,q_mean ,q_global ):
        """Accumulate local/global critic Q values for distributed agents."""
        self ._set_mode ("distributed")
        for i ,q_val in enumerate (q_values_per_agent ):
            if i <len (self .local_q_agents_sums ):
                self .local_q_agents_sums [i ]+=q_val
        self .local_q_mean_sum +=q_mean
        self .global_q_sum +=q_global
        self .q_step_count +=1

    def update_central_q_value (self ,q_value ):
        """Accumulate a centralized critic Q value for joint-policy baselines."""
        self ._set_mode ("centralized_joint")
        self .central_q_sum +=float (q_value )
        self .q_step_count +=1

    def update_gradients (self ,agent ):
        """Read the latest gradient diagnostics exposed by the agent."""
        is_central =getattr (agent ,"visualizer_layout","")=="centralized_joint"
        if is_central :
            self ._set_mode ("centralized_joint")
            self .central_critic_grad_sum_before_clip +=float (
            getattr (agent ,"last_central_critic_grad_norm_before_clip",0.0 )
            )
            if getattr (agent ,"last_central_actor_updated",True ):
                self .central_actor_grad_sum_before_clip +=float (
                getattr (agent ,"last_central_actor_grad_norm_before_clip",0.0 )
                )
                self .central_actor_grad_step_count +=1
            self .central_joint_reward_sum +=float (
            getattr (agent ,"last_joint_reward_mean",0.0 )
            )
            self .central_joint_local_reward_sum +=float (
            getattr (agent ,"last_joint_local_reward_mean",0.0 )
            )
            self .central_joint_global_reward_sum +=float (
            getattr (agent ,"last_joint_global_reward_mean",0.0 )
            )
            self .central_reward_step_count +=1
        else :
            self ._set_mode ("distributed")
            self .global_critic_grad_sum_before_clip +=getattr (agent ,'last_global_critic_grad_norm_before_clip',0.0 )
            self .actor_source_local_grad_sum_before_clip +=float (
            getattr (agent ,"last_actor_source_local_grad_norm_before_clip",0.0 )
            )
            self .actor_source_global_grad_sum_before_clip +=float (
            getattr (agent ,"last_actor_source_global_grad_norm_before_clip",0.0 )
            )
            self .actor_source_global_ratio_sum +=float (
            getattr (agent ,"last_actor_source_global_ratio",0.0 )
            )
            self .actor_source_cos_sum +=float (getattr (agent ,"last_actor_source_cos",0.0 ))
            self .actor_source_cos_valid_fraction_sum +=float (
            getattr (agent ,"last_actor_source_cos_valid_fraction",0.0 )
            )

            if hasattr (agent ,'actor_source_local_norms_before_clip')and len (agent .actor_source_local_norms_before_clip )==self .num_stations :
                for i ,v in enumerate (agent .actor_source_local_norms_before_clip ):
                    if i <len (self .actor_source_local_norm_sums_before_clip ):
                        self .actor_source_local_norm_sums_before_clip [i ]+=float (v )

            if hasattr (agent ,'actor_source_global_norms_before_clip')and len (agent .actor_source_global_norms_before_clip )==self .num_stations :
                for i ,v in enumerate (agent .actor_source_global_norms_before_clip ):
                    if i <len (self .actor_source_global_norm_sums_before_clip ):
                        self .actor_source_global_norm_sums_before_clip [i ]+=float (v )

            if hasattr (agent ,'actor_source_global_ratio')and len (agent .actor_source_global_ratio )==self .num_stations :
                for i ,v in enumerate (agent .actor_source_global_ratio ):
                    if i <len (self .actor_source_global_ratio_sums ):
                        self .actor_source_global_ratio_sums [i ]+=float (v )

            if hasattr (agent ,'actor_source_cos')and len (agent .actor_source_cos )==self .num_stations :
                for i ,v in enumerate (agent .actor_source_cos ):
                    if i <len (self .actor_source_cos_sums ):
                        self .actor_source_cos_sums [i ]+=float (v )

            if hasattr (agent ,'actor_source_cos_valid')and len (agent .actor_source_cos_valid )==self .num_stations :
                for i ,v in enumerate (agent .actor_source_cos_valid ):
                    if i <len (self .actor_source_cos_valid_sums ):
                        self .actor_source_cos_valid_sums [i ]+=int (v )

            if hasattr (agent ,'critic_norms_before_clip')and len (agent .critic_norms_before_clip )==self .num_stations :
                for i ,critic_norm_before in enumerate (agent .critic_norms_before_clip ):
                    if i <len (self .critic_norms_sums_before_clip ):
                        self .critic_norms_sums_before_clip [i ]+=critic_norm_before

            if hasattr (agent ,'actor_norms_before_clip')and len (agent .actor_norms_before_clip )==self .num_stations :
                for i ,actor_norm_before in enumerate (agent .actor_norms_before_clip ):
                    if i <len (self .actor_norms_sums_before_clip ):
                        self .actor_norms_sums_before_clip [i ]+=actor_norm_before

            self .local_critic_grad_sum +=getattr (agent ,'last_local_critic_grad_norm',0.0 )
            self .global_critic_grad_sum +=getattr (agent ,'last_global_critic_grad_norm',0.0 )

        self .grad_step_count +=1

    def update_losses (self ,agent ):
        """Read the latest critic/actor loss diagnostics exposed by the agent."""
        is_central =getattr (agent ,"visualizer_layout","")=="centralized_joint"
        if is_central :
            self ._set_mode ("centralized_joint")
            self .central_critic_loss_sum +=float (getattr (agent ,"last_central_critic_loss",0.0 ))
            if getattr (agent ,"last_central_actor_updated",True ):
                self .central_actor_loss_sum +=float (getattr (agent ,"last_central_actor_loss",0.0 ))
                self .central_actor_loss_step_count +=1
        else :
            self ._set_mode ("distributed")
            if hasattr (agent ,'critic_losses')and len (agent .critic_losses )==self .num_stations :
                for i ,loss in enumerate (agent .critic_losses ):
                    if i <len (self .critic_loss_sums ):
                        self .critic_loss_sums [i ]+=loss

            self .global_critic_loss_sum +=getattr (agent ,'last_global_critic_loss',0.0 )

            if hasattr (agent ,'actor_losses')and len (agent .actor_losses )==self .num_stations :
                for i ,loss in enumerate (agent .actor_losses ):
                    if i <len (self .actor_loss_sums ):
                        self .actor_loss_sums [i ]+=loss

        self .loss_step_count +=1

    def update_clipping (self ,agent ):
        """Accumulate per-step gradient clipping counts."""
        is_central =getattr (agent ,"visualizer_layout","")=="centralized_joint"
        if is_central :
            self ._set_mode ("centralized_joint")
            self .central_critic_clip_sum +=int (getattr (agent ,"last_central_critic_clip_count",0 ))
            if getattr (agent ,"last_central_actor_updated",True ):
                self .central_actor_clip_sum +=int (getattr (agent ,"last_central_actor_clip_count",0 ))
                self .central_actor_clip_step_count +=1
        else :
            self ._set_mode ("distributed")
            self .global_critic_clip_sum +=getattr (agent ,'last_global_critic_clip_count',0 )

            if hasattr (agent ,'local_critic_clip_counts')and len (agent .local_critic_clip_counts )==self .num_stations :
                for i ,clip_count in enumerate (agent .local_critic_clip_counts ):
                    if i <len (self .local_critic_clip_sums ):
                        self .local_critic_clip_sums [i ]+=clip_count

            if hasattr (agent ,'actor_clip_counts')and len (agent .actor_clip_counts )==self .num_stations :
                for i ,clip_count in enumerate (agent .actor_clip_counts ):
                    if i <len (self .actor_clip_sums ):
                        self .actor_clip_sums [i ]+=clip_count

        self .clip_step_count +=1

    def record_to_tensorboard (self ,episode ):
        """Write aggregated Q/gradient/loss/clipping metrics to TensorBoard."""
        if not self .tb_writer :
            return

        w =self .tb_writer
        try :
            from Config import TB_VERBOSE as _verbose
        except Exception :
            _verbose =False

        if self .mode =="centralized_joint":
            if self .q_step_count >0 :
                w .add_scalar ("Q/central_mean",self .central_q_sum /self .q_step_count ,episode )
            if self .grad_step_count >0 :
                w .add_scalar ("Gradient/central_critic_raw",
                self .central_critic_grad_sum_before_clip /self .grad_step_count ,episode )
            if self .central_actor_grad_step_count >0 :
                w .add_scalar ("Gradient/central_actor_raw",
                self .central_actor_grad_sum_before_clip /self .central_actor_grad_step_count ,episode )
            if self .central_reward_step_count >0 :
                n_reward =self .central_reward_step_count
                w .add_scalar ("Reward/joint_objective",
                self .central_joint_reward_sum /n_reward ,episode )
                w .add_scalar ("Reward/joint_local_term",
                self .central_joint_local_reward_sum /n_reward ,episode )
                w .add_scalar ("Reward/joint_global_term",
                self .central_joint_global_reward_sum /n_reward ,episode )
            if self .loss_step_count >0 :
                w .add_scalar ("Loss/central_critic",
                self .central_critic_loss_sum /self .loss_step_count ,episode )
            if self .central_actor_loss_step_count >0 :
                w .add_scalar ("Loss/central_actor",
                self .central_actor_loss_sum /self .central_actor_loss_step_count ,episode )
            if self .clip_step_count >0 :
                w .add_scalar ("Clipping/central_critic",
                self .central_critic_clip_sum /self .clip_step_count ,episode )
            if self .central_actor_clip_step_count >0 :
                w .add_scalar ("Clipping/central_actor",
                self .central_actor_clip_sum /self .central_actor_clip_step_count ,episode )
            return

        if self .q_step_count >0 :
            n =self .q_step_count
            w .add_scalar ("Q/local_mean",self .local_q_mean_sum /n ,episode )
            w .add_scalar ("Q/global",self .global_q_sum /n ,episode )
            if _verbose :
                for i ,s in enumerate (self .local_q_agents_sums ):
                    w .add_scalar (f"Q/local_agent{i+1}",s /n ,episode )

        if self .grad_step_count >0 :
            n =self .grad_step_count
            w .add_scalar ("Gradient/global_critic_raw",
            self .global_critic_grad_sum_before_clip /n ,episode )
            w .add_scalar ("Gradient/actor_source_local_raw_mean",
            self .actor_source_local_grad_sum_before_clip /n ,episode )
            w .add_scalar ("Gradient/actor_source_global_raw_mean",
            self .actor_source_global_grad_sum_before_clip /n ,episode )
            w .add_scalar ("Gradient/actor_source_global_ratio_mean",
            self .actor_source_global_ratio_sum /n ,episode )
            w .add_scalar ("Gradient/actor_source_cos_mean",
            self .actor_source_cos_sum /n ,episode )
            w .add_scalar ("Gradient/actor_source_cos_valid_fraction_mean",
            self .actor_source_cos_valid_fraction_sum /n ,episode )
            w .add_scalar ("Gradient/local_critic_mean_after_clip",
            self .local_critic_grad_sum /n ,episode )
            w .add_scalar ("Gradient/global_critic_after_clip",
            self .global_critic_grad_sum /n ,episode )
            if _verbose :
                for i ,v in enumerate (self .critic_norms_sums_before_clip ):
                    w .add_scalar (f"Gradient/local_critic_raw_agent{i+1}",v /n ,episode )
                for i ,v in enumerate (self .actor_norms_sums_before_clip ):
                    w .add_scalar (f"Gradient/actor_raw_agent{i+1}",v /n ,episode )
                for i ,v in enumerate (self .actor_source_local_norm_sums_before_clip ):
                    w .add_scalar (f"Gradient/actor_source_local_raw_agent{i+1}",v /n ,episode )
                for i ,v in enumerate (self .actor_source_global_norm_sums_before_clip ):
                    w .add_scalar (f"Gradient/actor_source_global_raw_agent{i+1}",v /n ,episode )
                for i ,v in enumerate (self .actor_source_global_ratio_sums ):
                    w .add_scalar (f"Gradient/actor_source_global_ratio_agent{i+1}",v /n ,episode )
                for i ,v in enumerate (self .actor_source_cos_sums ):
                    w .add_scalar (f"Gradient/actor_source_cos_agent{i+1}",v /n ,episode )
                for i ,v in enumerate (self .actor_source_cos_valid_sums ):
                    w .add_scalar (f"Gradient/actor_source_cos_valid_fraction_agent{i+1}",v /n ,episode )
        if self .loss_step_count >0 :
            n =self .loss_step_count
            w .add_scalar ("Loss/global_critic",self .global_critic_loss_sum /n ,episode )
            w .add_scalar ("Loss/local_critic_mean",
            sum (self .critic_loss_sums )/len (self .critic_loss_sums )/n
            if self .critic_loss_sums else 0.0 ,episode )
            if _verbose :
                for i ,v in enumerate (self .critic_loss_sums ):
                    w .add_scalar (f"Loss/local_critic_agent{i+1}",v /n ,episode )
                for i ,v in enumerate (self .actor_loss_sums ):
                    w .add_scalar (f"Loss/actor_agent{i+1}",v /n ,episode )

        if self .clip_step_count >0 :
            n =self .clip_step_count
            w .add_scalar ("Clipping/global_critic",self .global_critic_clip_sum /n ,episode )
            if _verbose :
                for i ,v in enumerate (self .local_critic_clip_sums ):
                    w .add_scalar (f"Clipping/local_critic_agent{i+1}",v /n ,episode )
                for i ,v in enumerate (self .actor_clip_sums ):
                    w .add_scalar (f"Clipping/actor_agent{i+1}",v /n ,episode )

    def record_agent_state (self ,agent ,training_ep ):
        """Write replay-buffer and exploration-state metrics to TensorBoard."""
        if self .tb_writer is None :
            return
        if hasattr (agent ,'buf'):
            buf =agent .buf
            buf_used =int (getattr (buf ,'size',0 ))
            buf_cap =int (getattr (buf ,'buf_size',0 ))
            if buf_cap >0 :
                self .tb_writer .add_scalar ("Training/buffer_fill_rate",buf_used /buf_cap ,training_ep )
            if buf_used >0 :
                self .tb_writer .add_scalar ("Training/buffer_size",buf_used ,training_ep )
            if hasattr (buf ,'per_diagnostics'):
                try :
                    for key ,value in buf .per_diagnostics ().items ():
                        if value is None :
                            continue
                        value_f =float (value )
                        if np .isfinite (value_f ):
                            self .tb_writer .add_scalar (f"PER/{key}",value_f ,training_ep )
                except Exception as exc :
                    warnings .warn (f"Failed to write PER diagnostics: {exc}")
        if hasattr (agent ,'epsilon'):
            self .tb_writer .add_scalar ("Training/epsilon",float (agent .epsilon ),training_ep )
        if hasattr (agent ,'ou_noise_scale'):
            self .tb_writer .add_scalar ("Training/ou_noise_scale",float (agent .ou_noise_scale ),training_ep )
        if hasattr (agent ,'update_step'):
            self .tb_writer .add_scalar ("Training/update_step",int (agent .update_step ),training_ep )
        independent_reward_tags =(
        ("last_independent_local_reward_mean","Reward/independent_local_term"),
        ("last_independent_global_reward_mean","Reward/independent_global_term"),
        ("last_independent_joint_reward_mean","Reward/independent_joint_objective"),
        )
        for attr ,tag in independent_reward_tags :
            if hasattr (agent ,attr ):
                try :
                    value =float (getattr (agent ,attr ))
                except (TypeError ,ValueError ):
                    continue
                if np .isfinite (value ):
                    self .tb_writer .add_scalar (tag ,value ,training_ep )
        regular_diagnostic_tags =(
        ("last_regular_reward_scale","RegularMADDPG/reward_scale"),
        ("last_regular_td_target_abs_mean","RegularMADDPG/td_target_abs_mean"),
        ("last_regular_td_error_abs_mean","RegularMADDPG/td_error_abs_mean"),
        ("last_regular_current_q_abs_mean","RegularMADDPG/current_q_abs_mean"),
        ("last_regular_target_q_abs_mean","RegularMADDPG/target_q_abs_mean"),
        )
        for attr ,tag in regular_diagnostic_tags :
            if hasattr (agent ,attr ):
                try :
                    value =float (getattr (agent ,attr ))
                except (TypeError ,ValueError ):
                    continue
                if np .isfinite (value ):
                    self .tb_writer .add_scalar (tag ,value ,training_ep )
        global_critic_diagnostic_tags =(
        ("last_global_reward_scale","GlobalCritic/reward_scale"),
        ("last_global_reward_baseline","GlobalCritic/reward_baseline"),
        ("last_global_reward_raw_abs_mean","GlobalCritic/reward_raw_abs_mean"),
        ("last_global_reward_centered_abs_mean","GlobalCritic/reward_centered_abs_mean"),
        ("last_global_reward_term_abs_mean","GlobalCritic/reward_term_abs_mean"),
        ("last_global_td_target_abs_mean","GlobalCritic/td_target_abs_mean"),
        ("last_global_td_error_abs_mean","GlobalCritic/td_error_abs_mean"),
        ("last_global_current_q_abs_mean","GlobalCritic/current_q_abs_mean"),
        ("last_global_target_q_abs_mean","GlobalCritic/target_q_abs_mean"),
        )
        for attr ,tag in global_critic_diagnostic_tags :
            if hasattr (agent ,attr ):
                try :
                    value =float (getattr (agent ,attr ))
                except (TypeError ,ValueError ):
                    continue
                if np .isfinite (value ):
                    self .tb_writer .add_scalar (tag ,value ,training_ep )
        if hasattr (agent ,'last_local_q_twin_gap_mean'):
            self .tb_writer .add_scalar (
            "Q/local_twin_gap_mean",
            float (getattr (agent ,'last_local_q_twin_gap_mean',0.0 )),
            training_ep ,
            )
            try :
                from Config import TB_VERBOSE as _verbose
            except Exception :
                _verbose =False
            if _verbose and hasattr (agent ,'last_local_q_twin_gap_values_per_agent'):
                for i ,value in enumerate (agent .last_local_q_twin_gap_values_per_agent ):
                    self .tb_writer .add_scalar (f"Q/local_twin_gap_agent{i+1}",float (value ),training_ep )

    def reset_episode_data (self ):
        """Clear all per-episode accumulators after TensorBoard emission."""
        self .mode =None

        self .local_q_agents_sums =[0.0 ]*self .num_stations
        self .local_q_mean_sum =0.0
        self .global_q_sum =0.0
        self .central_q_sum =0.0
        self .q_step_count =0

        self .global_critic_grad_sum_before_clip =0.0
        self .critic_norms_sums_before_clip =[0.0 ]*self .num_stations
        self .actor_norms_sums_before_clip =[0.0 ]*self .num_stations
        self .actor_source_local_norm_sums_before_clip =[0.0 ]*self .num_stations
        self .actor_source_global_norm_sums_before_clip =[0.0 ]*self .num_stations
        self .actor_source_global_ratio_sums =[0.0 ]*self .num_stations
        self .actor_source_cos_sums =[0.0 ]*self .num_stations
        self .actor_source_cos_valid_sums =[0 ]*self .num_stations
        self .actor_source_local_grad_sum_before_clip =0.0
        self .actor_source_global_grad_sum_before_clip =0.0
        self .actor_source_global_ratio_sum =0.0
        self .actor_source_cos_sum =0.0
        self .actor_source_cos_valid_fraction_sum =0.0
        self .central_critic_grad_sum_before_clip =0.0
        self .central_actor_grad_sum_before_clip =0.0
        self .grad_step_count =0
        self .central_actor_grad_step_count =0
        self .central_joint_reward_sum =0.0
        self .central_joint_local_reward_sum =0.0
        self .central_joint_global_reward_sum =0.0
        self .central_reward_step_count =0

        self .critic_loss_sums =[0.0 ]*self .num_stations
        self .global_critic_loss_sum =0.0
        self .actor_loss_sums =[0.0 ]*self .num_stations
        self .central_critic_loss_sum =0.0
        self .central_actor_loss_sum =0.0
        self .loss_step_count =0
        self .central_actor_loss_step_count =0

        self .local_critic_clip_sums =[0 ]*self .num_stations
        self .global_critic_clip_sum =0
        self .actor_clip_sums =[0 ]*self .num_stations
        self .central_critic_clip_sum =0
        self .central_actor_clip_sum =0
        self .clip_step_count =0
        self .central_actor_clip_step_count =0

        self .local_critic_grad_sum =0.0
        self .global_critic_grad_sum =0.0





def plot_arrival_counts (all_episode_data :dict ,
results_dir :str ,
title_prefix :str ="")->None :
    """
    Plot and export the number of EV arrivals per step.

    If arrivals are station-resolved, the CSV includes per-station counts and
    the plot highlights representative station profiles. Initial seeded EVs are
    subtracted from step 1 so the figure reflects new arrivals only.
    """
    episode_keys =sorted (all_episode_data .keys ())
    if len (episode_keys )==0 :
        return

    ep_key =episode_keys [-1 ]
    ep =all_episode_data [ep_key ]

    if 'arrivals_per_step'not in ep or len (ep ['arrivals_per_step'])==0 :
        return

    raw_arrivals =ep ['arrivals_per_step']


    is_vector =any (isinstance (v ,(list ,tuple ,np .ndarray ))for v in raw_arrivals )
    if is_vector :

        rows =[]
        max_s =0
        for v in raw_arrivals :
            if isinstance (v ,(list ,tuple ,np .ndarray )):
                arr =np .asarray (v ,dtype =float ).reshape (-1 )
            else :
                arr =np .asarray ([float (v )],dtype =float )
            max_s =max (max_s ,int (arr .size ))
            rows .append (arr )
        T =len (rows )
        S =max_s
        arrivals_by_station =np .zeros ((T ,S ),dtype =float )
        for t ,arr in enumerate (rows ):
            n =min (S ,int (arr .size ))
            if n >0 :
                arrivals_by_station [t ,:n ]=arr [:n ]


        arrivals_total =arrivals_by_station .sum (axis =1 )
        steps =np .arange (1 ,arrivals_total .shape [0 ]+1 )
    else :
        arrivals_list =[]
        for value in raw_arrivals :
            try :
                arrivals_list .append (float (value ))
            except (TypeError ,ValueError ):
                arrivals_list .append (0.0 )
        arrivals_total =np .asarray (arrivals_list ,dtype =float )
        steps =np .arange (1 ,len (arrivals_total )+1 )

    title_prefix_en =title_prefix

    if is_vector :

        from Config import PER_STATION_SESSION_IDS

        S =int (arrivals_by_station .shape [1 ])
        if len (PER_STATION_SESSION_IDS )<S :
            print ("[plot_arrival_counts] the station roster is shorter than the arrival data; skipping grouped plot.")
            return
        profile_paths =list (PER_STATION_SESSION_IDS [:S ])


        groups ={}
        for st ,pth in enumerate (profile_paths ):
            groups .setdefault (str (pth ),[]).append (st )


        group_items =[]
        for pth ,sts in groups .items ():
            group_total =float (arrivals_by_station [:,sts ].sum ())if sts else 0.0
            rep =int (sts [0 ])if sts else 0
            group_items .append ((group_total ,pth ,rep ,sts ))
        group_items .sort (reverse =True ,key =lambda x :x [0 ])
        selected =group_items [:5 ]

        fig ,axes =plt .subplots (5 ,1 ,figsize =(20 ,18 ),sharex =True )
        for i in range (5 ):
            ax =axes [i ]
            if i <len (selected ):
                _ ,pth ,rep ,sts =selected [i ]
                y =arrivals_by_station [:,rep ]
                label_name =str (pth )
                ax .bar (steps ,y ,color ='skyblue',alpha =0.55 )
                ax .plot (steps ,y ,color ='navy',linewidth =2 )
                ax .set_title (f"Station {rep} (rep of {len(sts)} stations) | {label_name}",fontsize =14 )
                ax .grid (alpha =0.3 ,linewidth =1.0 )
                ax .set_ylabel ('Arrivals',fontsize =12 )
            else :
                ax .axis ('off')

        axes [-1 ].set_xlabel ('Step',fontsize =14 )
        fig .suptitle (f"{title_prefix_en} Arrivals per Step (by station, Episode {ep_key})".strip (),fontsize =18 )
        fig .tight_layout ()
    else :

        fig ,ax =plt .subplots (figsize =(20 ,10 ))
        ax .bar (steps ,arrivals_total ,color ='skyblue',alpha =0.6 ,label ='New Arrivals (bar)')
        ax .plot (steps ,arrivals_total ,color ='navy',linewidth =3 ,marker ='o',markersize =6 ,label ='New Arrivals (line)')

        ax .set_xlabel ('Step',fontsize =26 )
        ax .set_ylabel ('Newly Arrived EVs [count]',fontsize =26 )
        ax .grid (alpha =0.3 ,linewidth =1.5 )
        ax .legend (fontsize =18 )
        fig .tight_layout ()

    fname =f"arrive_EV_per_step_episode_{ep_key}.png"
    if title_prefix :
        prefix_clean =title_prefix .lower ().replace (' ','_')
        fname =f"{prefix_clean}_{fname}"

    fig .savefig (os .path .join (results_dir ,fname ),dpi =140 ,bbox_inches ='tight')
    plt .close (fig )



    csv_filename =fname .replace ('.png','.csv')
    csv_path =os .path .join (results_dir ,csv_filename )
    with open (csv_path ,'w',newline ='',encoding ='utf-8')as f :
        writer =csv .writer (f )
        writer .writerow (['Step','New_Arrivals'])
        for i ,value in enumerate (arrivals_total ):
            writer .writerow ([i +1 ,int (value )])


    if is_vector :
        by_station_csv =csv_filename .replace ('.csv','_by_station.csv')
        by_station_path =os .path .join (results_dir ,by_station_csv )
        with open (by_station_path ,'w',newline ='',encoding ='utf-8')as f :
            writer =csv .writer (f )
            header =['Step']+[f'Station_{i}'for i in range (int (arrivals_by_station .shape [1 ]))]+['Total']
            writer .writerow (header )
            for t in range (int (arrivals_by_station .shape [0 ])):
                row =[t +1 ]+[int (x )for x in arrivals_by_station [t ,:]]+[int (arrivals_total [t ])]
                writer .writerow (row )






def plot_power_mismatch_analysis (all_episode_data :dict ,
results_dir :str ,
title_prefix :str ="")->None :
    """
    Plot per-step and cumulative dispatch mismatch.

    Mismatch is `grid request - actual EV power`. Positive values indicate an
    unmet request; negative values indicate excess EV response. Non-awarded
    steps are shown but excluded from extrema and cumulative assessed energy.
    """

    episode_keys =sorted (all_episode_data .keys ())
    if len (episode_keys )==0 :
        return

    ep_key =episode_keys [-1 ]
    ep =all_episode_data [ep_key ]

    if 'power_mismatch'not in ep or len (ep ['power_mismatch'])==0 :
        return

    mismatches =np .array (ep ['power_mismatch'],dtype =float )
    steps =np .arange (1 ,len (mismatches )+1 )

    tracking_raw =ep .get ('tracking_enabled')
    if tracking_raw is None :
        tracking_enabled =np .ones (len (mismatches ),dtype =bool )
    else :
        tracking_enabled =np .asarray (tracking_raw ,dtype =bool ).reshape (-1 )[:len (mismatches )]
        if tracking_enabled .size <len (mismatches ):
            tracking_enabled =np .pad (
            tracking_enabled ,(0 ,len (mismatches )-tracking_enabled .size ),constant_values =True
            )
    assessed_mismatches =np .where (tracking_enabled ,mismatches ,0.0 )
    assessed_indices =np .flatnonzero (tracking_enabled )


    if assessed_indices .size :
        max_over_idx =int (assessed_indices [np .argmax (mismatches [assessed_indices ])])
        max_under_idx =int (assessed_indices [np .argmin (mismatches [assessed_indices ])])
        max_over_value =mismatches [max_over_idx ]
        max_under_value =mismatches [max_under_idx ]
    else :
        max_over_idx =max_under_idx =None
        max_over_value =max_under_value =0.0


    raw_cumulative_energy =np .cumsum (mismatches *POWER_TO_ENERGY )
    energy_mismatches =assessed_mismatches *POWER_TO_ENERGY
    cumulative_energy =np .cumsum (energy_mismatches )


    if assessed_indices .size :
        max_cumulative_idx =int (np .argmax (np .abs (cumulative_energy )))
        max_cumulative_value =cumulative_energy [max_cumulative_idx ]
    else :
        max_cumulative_idx =None
        max_cumulative_value =0.0


    fig ,(ax1 ,ax2 )=plt .subplots (2 ,1 ,figsize =(20 ,16 ))


    colors =[
    ('#B8B8B8'if not enabled else ('red'if mismatch >0 else 'blue'))
    for mismatch ,enabled in zip (mismatches ,tracking_enabled )
    ]
    ax1 .bar (steps ,mismatches ,color =colors ,alpha =0.6 ,width =0.8 )


    if max_over_idx is not None :
        max_over_step =steps [max_over_idx ]
        max_under_step =steps [max_under_idx ]
        ax1 .bar (max_over_step ,max_over_value ,color ='darkred',alpha =0.9 ,width =0.8 )
        ax1 .bar (max_under_step ,max_under_value ,color ='darkblue',alpha =0.9 ,width =0.8 )


        ax1 .annotate (
        f'Max Shortage\nStep {max_over_idx+1}\n{max_over_value:.1f} kW',
        xy =(max_over_step ,max_over_value ),xytext =(0 ,-8 ),textcoords ='offset points',
        ha ='center',va ='top',fontsize =20 ,color ='darkred',fontweight ='bold')
        ax1 .annotate (
        f'Max Excess\nStep {max_under_idx+1}\n{max_under_value:.1f} kW',
        xy =(max_under_step ,max_under_value ),xytext =(0 ,8 ),textcoords ='offset points',
        ha ='center',va ='bottom',fontsize =20 ,color ='darkblue',fontweight ='bold')

    unawarded =np .logical_not (tracking_enabled )
    unawarded_starts =np .flatnonzero (unawarded &np .r_ [True ,~unawarded [:-1 ]])
    unawarded_stops =np .flatnonzero (unawarded &np .r_ [~unawarded [1 :],True ])+1
    for span_start ,span_stop in zip (unawarded_starts ,unawarded_stops ):
        for axis in (ax1 ,ax2 ):
            axis .axvspan (
            float (span_start )+0.5 ,float (span_stop )+0.5 ,
            color ='#E69F00',alpha =0.18 ,linewidth =0 ,zorder =0
            )
    if np .any (unawarded ):
        import matplotlib .patches as mpatches
        ax1 .legend (
        handles =[mpatches .Patch (
        color ='#E69F00',alpha =0.18 ,label ='No awarded capacity (not assessed)'
        )],loc ='lower left'
        )

    ax1 .axhline (0 ,color ='black',linewidth =2 ,alpha =0.5 )
    ax1 .set_xlabel ('Step',fontsize =30 )
    ax1 .set_ylabel ('Mismatch [kW]\n(Request - Actual)',fontsize =30 )


    ax1 .grid (alpha =0.3 ,linewidth =2 )


    ax2 .plot (steps ,cumulative_energy ,linewidth =4 ,color ='purple',alpha =0.8 )
    ax2 .fill_between (steps ,0 ,cumulative_energy ,alpha =0.3 ,color ='purple')


    if max_cumulative_idx is not None :
        max_cumulative_step =steps [max_cumulative_idx ]
        ax2 .plot (max_cumulative_step ,max_cumulative_value ,'o',
        markersize =20 ,color ='darkred',zorder =5 )
        cumulative_near_right =max_cumulative_idx >=int (0.8 *len (steps ))
        ax2 .annotate (
        f'Max Cumulative\nStep {max_cumulative_idx+1}\n{max_cumulative_value:.2f} kWh',
        xy =(max_cumulative_step ,max_cumulative_value ),
        xytext =(-8 if cumulative_near_right else 0 ,8 if max_cumulative_value <0 else -8 ),
        textcoords ='offset points',
        ha ='right'if cumulative_near_right else 'center',
        va ='bottom'if max_cumulative_value <0 else 'top',
        fontsize =20 ,color ='darkred',fontweight ='bold')

    ax2 .axhline (0 ,color ='black',linewidth =2 ,alpha =0.5 )
    ax2 .set_xlabel ('Step',fontsize =30 )
    ax2 .set_ylabel ('Cumulative Assessed Energy Mismatch [kWh]',fontsize =30 )


    ax2 .grid (alpha =0.3 ,linewidth =2 )

    fig .tight_layout ()


    fname =f"power_mismatch_analysis_episode_{ep_key}.png"
    if title_prefix :
        prefix_clean =title_prefix .lower ().replace (' ','_')
        fname =f"{prefix_clean}_{fname}"

    fig .savefig (os .path .join (results_dir ,fname ),dpi =140 ,bbox_inches ='tight')
    plt .close (fig )



    csv_filename =fname .replace ('.png','.csv')
    csv_path =os .path .join (results_dir ,csv_filename )
    with open (csv_path ,'w',newline ='',encoding ='utf-8')as f :
        writer =csv .writer (f )
        writer .writerow ([
        'Step','Tracking_Enabled','Mismatch_kW','Assessed_Mismatch_kW',
        'Cumulative_Assessed_Energy_kWh','Cumulative_Energy_kWh'
        ])
        for i in range (len (steps )):
            writer .writerow ([
            i +1 ,int (tracking_enabled [i ]),mismatches [i ],assessed_mismatches [i ],
            cumulative_energy [i ],raw_cumulative_energy [i ]
            ])



def plot_reward_breakdown (all_episode_data :dict ,
results_dir :str ,
title_prefix :str ="")->None :
    """
    Plot local and global reward components over time.

    Local components are progress shaping and the departure reward. The global
    component is the demand-balance reward.
    """

    title_prefix_en =title_prefix


    episode_keys =sorted (all_episode_data .keys ())
    if len (episode_keys )>2 :
        episode_keys =episode_keys [-2 :]

    for ep_key in episode_keys :
        ep =all_episode_data [ep_key ]


        if 'rewards_global_balance'not in ep :
            continue

        steps =np .arange (1 ,len (ep ['rewards_global_balance'])+1 )
        local_shaping =np .asarray (ep ['rewards_local_shaping'],dtype =float )
        local_departure =np .asarray (ep ['rewards_local_departure'],dtype =float )
        global_balance =np .asarray (ep ['rewards_global_balance'],dtype =float )

        fig ,(ax1 ,ax2 )=plt .subplots (2 ,1 ,figsize =(20 ,16 ),sharex =True )


        ax1 .plot (steps ,local_shaping ,label ='Local Shaping',color ='cyan',lw =3 ,alpha =0.8 )
        ax1 .plot (steps ,local_departure ,label ='Local Departure',color ='orange',lw =4 ,marker ='o',markersize =4 ,ls ='None')


        local_total =local_shaping +local_departure
        ax1 .plot (steps ,local_total ,label ='Local Total (Sum)',color ='green',lw =2 ,ls ='--',alpha =0.5 )


        all_local_vals =np .concatenate ([
        local_shaping ,
        local_departure ,
        local_total ,
        ])
        if len (all_local_vals )>0 :
            v_min ,v_max =np .min (all_local_vals ),np .max (all_local_vals )

            if v_min ==v_max :
                ax1 .set_ylim (v_min -1.0 ,v_max +1.0 )
            else :
                margin =(v_max -v_min )*0.1
                ax1 .set_ylim (v_min -margin ,v_max +margin )

        ax1 .set_ylabel ('Local Reward Components',fontsize =26 )

        ax1 .grid (alpha =0.3 ,lw =1.5 )
        ax1 .legend (loc ='upper left',fontsize =18 )
        ax1 .axhline (0 ,color ='black',lw =2 ,alpha =0.5 )


        ax2 .plot (steps ,global_balance ,label ='Global Balance',color ='blue',lw =4 )


        if len (global_balance )>0 :
            gv_min ,gv_max =np .min (global_balance ),np .max (global_balance )
            if gv_min ==gv_max :
                ax2 .set_ylim (gv_min -1.0 ,gv_max +1.0 )
            else :
                margin =(gv_max -gv_min )*0.1
                ax2 .set_ylim (gv_min -margin ,gv_max +margin )

        ax2 .set_xlabel ('Step',fontsize =26 )
        ax2 .set_ylabel ('Global Reward Components',fontsize =26 )

        ax2 .grid (alpha =0.3 ,lw =1.5 )
        ax2 .legend (loc ='upper left',fontsize =18 )
        ax2 .axhline (0 ,color ='black',lw =2 ,alpha =0.5 )




        fig .tight_layout (rect =[0 ,0.03 ,1 ,0.95 ])


        fname =f"reward_breakdown_episode_{ep_key}.png"
        if title_prefix :
            prefix_clean =title_prefix_en .lower ().replace (' ','_')
            fname =f"{prefix_clean}_{fname}"

        fig .savefig (os .path .join (results_dir ,fname ),dpi =140 ,bbox_inches ='tight')
        plt .close (fig )



        csv_filename =fname .replace ('.png','.csv')
        csv_path =os .path .join (results_dir ,csv_filename )
        with open (csv_path ,'w',newline ='',encoding ='utf-8')as f :
            writer =csv .writer (f )
            writer .writerow ([
            'Step',
            'Global_Balance',
            'Local_Shaping',
            'Local_Departure',
            ])
            for i in range (len (steps )):
                writer .writerow ([
                steps [i ],
                global_balance [i ],
                local_shaping [i ],
                local_departure [i ],
                ])






def snapshot_code_to_archive (model_dir :str ,project_root :str =None )->str :
    """
    Copy source files into `model_dir/code_snapshot` for reproducibility.

    Each training archive stores the code used to produce it, making later
    result review independent of subsequent edits in the working tree.
    """
    root =project_root or os .getcwd ()
    snapshot_dir =os .path .join (model_dir ,"code_snapshot")
    os .makedirs (snapshot_dir ,exist_ok =True )


    dirs_to_copy =["training","environment","tools","data"]
    # `data` holds datasets, not code, and it had been copied whole: 3.5 GB per
    # run against 2 MB of actual source, growing with every command library
    # added. Only its scripts are copied now. Nothing is lost -- which command
    # library a run used is already on the bid by content hash, the bank path
    # and the normalization profile are written to the run's own `input`
    # directory, and the resume state fingerprints all of them.
    source_only ={"data"}
    keep_suffixes =(".py",".txt",".md",".json",".yaml",".yml",".cfg",".toml")
    for d in dirs_to_copy :
        src =os .path .join (root ,d )
        dst =os .path .join (snapshot_dir ,d )
        if os .path .isdir (src ):
            code_only =d in source_only

            def _ignore (dirpath ,names ,code_only =code_only ):
                ignored =set ()
                base =os .path .basename (dirpath )
                if base in {"__pycache__",".git",".idea",".vscode",".venv",".pytest_cache",".mypy_cache"}:
                    ignored .update (names )
                    return ignored
                if code_only :
                    for name in names :
                        path =os .path .join (dirpath ,name )
                        if os .path .isfile (path )and not name .lower ().endswith (keep_suffixes ):
                            ignored .add (name )
                return ignored
            shutil .copytree (src ,dst ,dirs_exist_ok =True ,ignore =_ignore )


    root_files =[
    fn for fn in os .listdir (root )
    if os .path .isfile (os .path .join (root ,fn ))
    ]
    allow_names ={"requirements.txt","README.md","README.MD",".env"}
    for fn in root_files :
        if fn .endswith (".py")or fn in allow_names :
            src =os .path .join (root ,fn )
            dst =os .path .join (snapshot_dir ,fn )
            try :
                shutil .copy2 (src ,dst )
            except Exception :

                pass




    return snapshot_dir


import signal as _signal_module


class InterruptHandler :
    """Handle Ctrl+C and allow the training loop to stop safely."""

    def __init__ (self ):
        self ._interrupted =False

    def setup (self ):
        self ._interrupted =False
        _signal_module .signal (_signal_module .SIGINT ,self ._handler )

    def reset (self ):
        self ._interrupted =False

    def _handler (self ,sig ,frame ):
        self ._interrupted =True
        print ("\n[Info] Ctrl+C: Training will stop after the current episode.",flush =True )

    def is_interrupted (self ):
        return self ._interrupted


def write_train_episode_tb_scalars (
tb_writer ,
training_ep ,
steps_in_ep ,
*,
ep_local_r ,
ep_global_r ,
soc_miss_rate ,
central_soc_miss_rate =0.0 ,
surplus_absorption_rate ,
supply_cooperation_rate ,
raw_actor_tracking_success_rate =0.0 ,
central_tracking_success_rate =0.0 ,
raw_actor_mae_kw =0.0 ,
central_corrected_ev_steps =0 ,
central_corrected_station_steps =0 ,
central_absolute_correction_kwh =0.0 ,
central_max_abs_aggregate_correction_kw =0.0 ,
central_max_corrected_stations_per_step =0 ,
central_max_station_target_error_kw =0.0 ,
system_tracking_success_rate =0.0 ,
pre_bess_mae_kw =0.0 ,
post_bess_mae_kw =0.0 ,
bess_final_soc_pct =0.0 ,
bess_throughput_kwh =0.0 ,
bess_max_abs_power_kw =0.0 ,
bess_power_limit_hits =0 ,
bess_energy_limit_hits =0 ,
ep_local_departure_r =0.0 ,
ep_local_progress_shaping_r =0.0 ,
station_local_reward_sums =None ,
):
    """Write per-episode reward and metric scalars to TensorBoard."""
    from Config import TB_VERBOSE

    if tb_writer is None :
        return

    n =max (steps_in_ep ,1 )

    tb_writer .add_scalar ("Reward/local",ep_local_r /n ,training_ep )
    tb_writer .add_scalar ("Reward/global",ep_global_r /n ,training_ep )
    tb_writer .add_scalar ("Metrics/soc_hit_rate",100 -soc_miss_rate ,training_ep )
    tb_writer .add_scalar ("CentralEV/physical_soc_hit_rate",100 -central_soc_miss_rate ,training_ep )
    tb_writer .add_scalar ("Metrics/surplus_absorption_rate",surplus_absorption_rate ,training_ep )
    tb_writer .add_scalar ("Metrics/supply_cooperation_rate",supply_cooperation_rate ,training_ep )
    # 補正前の学習器そのものの成績.  CentralEV/ と System/ は補正後なので,
    # 制御器どうしを比べるときはこちらを見る.
    tb_writer .add_scalar ("RawActor/tracking_success_rate",raw_actor_tracking_success_rate ,training_ep )
    tb_writer .add_scalar ("CentralEV/tracking_success_rate",central_tracking_success_rate ,training_ep )
    tb_writer .add_scalar ("CentralEV/raw_actor_mae_kw",raw_actor_mae_kw ,training_ep )
    tb_writer .add_scalar ("CentralEV/corrected_ev_steps",central_corrected_ev_steps ,training_ep )
    tb_writer .add_scalar ("CentralEV/corrected_station_steps",central_corrected_station_steps ,training_ep )
    tb_writer .add_scalar ("CentralEV/absolute_correction_kwh",central_absolute_correction_kwh ,training_ep )
    tb_writer .add_scalar ("CentralEV/max_abs_aggregate_correction_kw",central_max_abs_aggregate_correction_kw ,training_ep )
    tb_writer .add_scalar ("CentralEV/max_corrected_stations_per_step",central_max_corrected_stations_per_step ,training_ep )
    tb_writer .add_scalar ("CentralEV/max_station_target_error_kw",central_max_station_target_error_kw ,training_ep )
    tb_writer .add_scalar ("System/tracking_success_rate",system_tracking_success_rate ,training_ep )
    tb_writer .add_scalar ("System/pre_bess_mae_kw",pre_bess_mae_kw ,training_ep )
    tb_writer .add_scalar ("System/post_bess_mae_kw",post_bess_mae_kw ,training_ep )
    tb_writer .add_scalar ("BESS/final_soc_pct",bess_final_soc_pct ,training_ep )
    tb_writer .add_scalar ("BESS/throughput_kwh",bess_throughput_kwh ,training_ep )
    tb_writer .add_scalar ("BESS/max_abs_power_kw",bess_max_abs_power_kw ,training_ep )
    tb_writer .add_scalar ("BESS/power_limit_hits",bess_power_limit_hits ,training_ep )
    tb_writer .add_scalar ("BESS/energy_limit_hits",bess_energy_limit_hits ,training_ep )

    if TB_VERBOSE :
        tb_writer .add_scalar ("Reward/local_departure",ep_local_departure_r /n ,training_ep )
        tb_writer .add_scalar ("Reward/local_shaping",ep_local_progress_shaping_r /n ,training_ep )
        if station_local_reward_sums :
            for st_idx ,sums in enumerate (station_local_reward_sums ):
                st_id =st_idx +1
                tb_writer .add_scalar (f"Reward/local_station{st_id}_total",sums ["total"]/n ,training_ep )
                tb_writer .add_scalar (f"Reward/local_station{st_id}_departure",sums ["departure"]/n ,training_ep )
                tb_writer .add_scalar (f"Reward/local_station{st_id}_shaping",sums ["progress_shaping"]/n ,training_ep )
