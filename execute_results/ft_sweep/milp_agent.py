"""




- MILPAgent(max_evs_per_station, horizon)
"""
import torch
import numpy as np

try :
    import pulp
except Exception as e :
    raise RuntimeError ("PuLP is required for MILPAgent. Please install 'pulp'.")from e 

from Config import (
MILP_W_AG ,
MILP_W_SOC ,
MILP_W_SWITCH ,
MILP_HORIZON ,
MAX_EV_PER_STATION ,
MAX_EV_POWER_KW ,
POWER_TO_ENERGY ,
EV_CAPACITY ,
MILP_SOLVER_TIME_LIMIT ,
MILP_SOLVER_GAP_REL ,
MILP_SOLVER_GAP_ABS ,
MILP_SOLVER_THREADS ,
MILP_SOLVER_PRESOLVE ,
MILP_SOLVER_CUTS ,
MILP_SOLVER_HEURISTIC ,
MILP_SOLVER_STRONG ,
USE_SWITCHING_CONSTRAINTS ,
)


def _quantize_soc_kwh (soc_kwh :float ,step_kwh :float )->float :
    if step_kwh <=0 :
        return float (np .clip (soc_kwh ,0.0 ,EV_CAPACITY ))
    q =round (soc_kwh /step_kwh )*step_kwh 
    return float (np .clip (q ,0.0 ,EV_CAPACITY ))


class MILPAgent :

    def __init__ (self ,max_evs_per_station :int =MAX_EV_PER_STATION ,horizon :int =MILP_HORIZON ):
        self .max_evs =max_evs_per_station 
        self .H =max (1 ,int (horizon ))
        self .test_mode =False 
        
        self .last_solution =None 

        
        self .w_ag =float (MILP_W_AG )
        self .w_soc =float (MILP_W_SOC )
        self .w_switch =float (MILP_W_SWITCH )

    def set_weights (self ,w_ag =None ,w_soc =None ,w_switch =None ):
        if w_ag is not None :self .w_ag =float (w_ag )
        if w_soc is not None :self .w_soc =float (w_soc )
        if w_switch is not None :self .w_switch =float (w_switch )
        print (f"[MILPAgent] Weights updated: AG={self.w_ag:.4f}, SOC={self.w_soc:.4f}, SWITCH={self.w_switch:.4f}")

    def set_test_mode (self ,mode :bool ):
        self .test_mode =bool (mode )

    def episode_start (self ):
        return 

    def episode_end (self ):
        return 

    def update_active_evs (self ,env ):
        return 

    def cache_experience (self ,*args ,**kwargs ):
        return 

    def act (self ,state ,env =None ,noise :bool =False ):
        if env is None :
            raise ValueError ("MILPAgent.act requires env to be provided")

        device =env .soc .device if hasattr (env ,'soc')else 'cpu'
        dtype =torch .float32 

        
        step =int (env .step_count )
        H =self .H 
        ag_series =[]
        for h in range (H ):
        
            idx =step -1 +h 
            if hasattr (env ,'net_demand_series')and env .net_demand_series is not None and idx <len (env .net_demand_series ):
                ag_series .append (float (env .net_demand_series [idx ].item ()))
            else :
                ag_series .append (0.0 )

                
        active_info =[]
        for st in range (env .num_stations ):
            active =torch .nonzero (env .ev_mask [st ],as_tuple =False ).squeeze (-1 )
            if len (active )>0 :
                ordered =env ._sort_active_evs (st ,active )
                ids =env .ev_ids [st ,ordered ].detach ().cpu ().numpy ().astype (int )
                s_true =env .soc [st ,ordered ].detach ().cpu ().numpy ()
                t =env .target [st ,ordered ].detach ().cpu ().numpy ()
                r =(env .depart [st ,ordered ]-env .step_count +1 ).detach ().cpu ().numpy ()

                
                s_obs =np .array ([
                float (np .clip (s_true [j ],0.0 ,EV_CAPACITY ))
                for j in range (len (ordered ))
                ],dtype =float )

                active_info .append ((st ,ordered ,ids ,s_obs ,t ,r ))
            else :
                active_info .append ((st ,torch .empty (0 ,dtype =torch .long ),np .array ([]),np .array ([]),np .array ([]),np .array ([])))

                
        m =pulp .LpProblem ("EV_LP",pulp .LpMinimize )

        
        x ={}
        z_c ={}
        z_d ={}
        S ={}
        sw ={}
        for st ,ordered ,ids ,s_obs ,t ,r in active_info :
        
            last_dirs =env .last_non_zero_state [st ,ordered ].detach ().cpu ().numpy ()
            for j in range (len (ordered )):
                R =int (max (0 ,r [j ]))
                for k in range (min (H ,R )):
                    x [(st ,j ,k )]=pulp .LpVariable (f"x_{st}_{j}_{k}",lowBound =-MAX_EV_POWER_KW ,upBound =MAX_EV_POWER_KW )
                    if USE_SWITCHING_CONSTRAINTS :
                        z_c [(st ,j ,k )]=pulp .LpVariable (f"z_c_{st}_{j}_{k}",cat ='Binary')
                        z_d [(st ,j ,k )]=pulp .LpVariable (f"z_d_{st}_{j}_{k}",cat ='Binary')
                        S [(st ,j ,k )]=pulp .LpVariable (f"S_{st}_{j}_{k}",cat ='Binary')
                        sw [(st ,j ,k )]=pulp .LpVariable (f"sw_{st}_{j}_{k}",lowBound =0 ,upBound =1 )

                        
                if USE_SWITCHING_CONSTRAINTS :
                    for k in range (min (H ,R )):
                        m +=x [(st ,j ,k )]<=MAX_EV_POWER_KW *z_c [(st ,j ,k )]
                        m +=x [(st ,j ,k )]>=-MAX_EV_POWER_KW *z_d [(st ,j ,k )]
                        m +=z_c [(st ,j ,k )]+z_d [(st ,j ,k )]<=1 
                        m +=S [(st ,j ,k )]>=z_c [(st ,j ,k )]
                        m +=S [(st ,j ,k )]<=1 -z_d [(st ,j ,k )]

                        if k ==0 :
                            ld =last_dirs [j ]
                            if ld !=0 :
                                s_prev =1 if ld ==1 else 0 
                                m +=sw [(st ,j ,k )]>=S [(st ,j ,k )]-s_prev 
                                m +=sw [(st ,j ,k )]>=s_prev -S [(st ,j ,k )]
                            else :
                                m +=sw [(st ,j ,k )]==0 
                        else :
                            m +=sw [(st ,j ,k )]>=S [(st ,j ,k )]-S [(st ,j ,k -1 )]
                            m +=sw [(st ,j ,k )]>=S [(st ,j ,k -1 )]-S [(st ,j ,k )]

                            
        e =[pulp .LpVariable (f"e_{k}",lowBound =0.0 )for k in range (H )]

        
        
        episode_steps =env .episode_steps if hasattr (env ,'episode_steps')else 288 
        u ={}
        for st ,ordered ,ids ,s_obs ,t ,r in active_info :
            for j in range (len (ordered )):
                R =int (max (0 ,r [j ]))
                
                if R >0 and (step +R <=episode_steps ):
                    u [(st ,j )]=pulp .LpVariable (f"u_{st}_{j}",lowBound =0.0 )

                    
        for st ,ordered ,ids ,s_obs ,t ,r in active_info :
            for j in range (len (ordered )):
                R =int (max (0 ,r [j ]))
                for k in range (min (H ,R )):
                    cum_energy =pulp .lpSum (x [(st ,j ,kk )]*POWER_TO_ENERGY for kk in range (k +1 )if (st ,j ,kk )in x )
                    m +=s_obs [j ]+cum_energy <=EV_CAPACITY 
                    m +=s_obs [j ]+cum_energy >=0.0 

                    
        for k in range (H ):
            sum_x_k =pulp .lpSum (x [key ]for key in x if key [2 ]==k )
            dev =sum_x_k -ag_series [k ]
            
            m +=e [k ]>=dev 
            m +=e [k ]>=-dev 

            
            
            
        for st ,ordered ,ids ,s_obs ,t ,r in active_info :
            for j in range (len (ordered )):
                R =int (max (0 ,r [j ]))
                
                
                if (st ,j )in u :
                
                    cum_energy =pulp .lpSum (
                    x [(st ,j ,k )]*POWER_TO_ENERGY 
                    for k in range (min (H ,R ))
                    if (st ,j ,k )in x 
                    )
                    
                    soc_at_departure =s_obs [j ]+cum_energy 
                    
                    target_soc =t [j ]
                    
                    m +=soc_at_departure +u [(st ,j )]>=target_soc 

                    
        total_active_evs =sum (len (ordered )for _ ,ordered ,_ ,_ ,_ ,_ in active_info )

        
        
        norm_factor =max (1.0 ,float (total_active_evs ))

        
        
        obj_ag_norm =pulp .lpSum (e [k ]for k in range (H ))/(MAX_EV_POWER_KW *norm_factor *H )

        
        obj_soc_norm =pulp .lpSum (u [v ]for v in u )/(EV_CAPACITY *norm_factor )

        
        obj_switch_norm =pulp .lpSum (sw [v ]for v in sw )/norm_factor 

        
        
        
        
        
        if USE_SWITCHING_CONSTRAINTS :
            m +=self .w_ag *obj_ag_norm +self .w_soc *obj_soc_norm +self .w_switch *obj_switch_norm 
        else :
            m +=self .w_ag *obj_ag_norm +self .w_soc *obj_soc_norm 

            
        num_variables =len (m .variables ())
        num_constraints =len (m .constraints )
        num_stations_with_evs =sum (1 for _ ,ordered ,_ ,_ ,_ ,_ in active_info if len (ordered )>0 )

        
        
        dynamic_time_limit =MILP_SOLVER_TIME_LIMIT 
        if dynamic_time_limit is not None and num_variables >0 :
        
            scale_factor =min (2.0 ,1.0 +(num_variables /1000.0 )*0.5 )
            dynamic_time_limit =dynamic_time_limit *scale_factor 

            
            
            
            
            
        n_vars ,m_constraints =num_variables ,num_constraints 

        
        
        flops_per_iteration =m_constraints *m_constraints *m_constraints 
        
        estimated_iterations =max (m_constraints ,int (n_vars *0.1 ))
        estimated_flops =flops_per_iteration *estimated_iterations 

        
        
        simple_flops_per_iteration =n_vars *m_constraints 
        simple_estimated_flops =simple_flops_per_iteration *estimated_iterations 

        
        if n_vars ==0 or m_constraints ==0 :
            complexity_order ='N/A'
        elif n_vars ==m_constraints :
            complexity_order =f"O({n_vars}^3)"
        elif n_vars <=m_constraints :
            complexity_order =f"O({n_vars}^2*{m_constraints})"
        else :
            complexity_order =f"O({n_vars}*{m_constraints}^2)"

            
        self .last_complexity_info ={
        'num_variables':num_variables ,
        'num_constraints':num_constraints ,
        'total_active_evs':total_active_evs ,
        'num_stations_with_evs':num_stations_with_evs ,
        'horizon':H ,
        'step':step ,
        'estimated_flops':estimated_flops ,
        'simple_estimated_flops':simple_estimated_flops ,
        'estimated_iterations':estimated_iterations ,
        'complexity_order':complexity_order ,
        }

        
        try :
        
            time_limit =None 
            if dynamic_time_limit is not None and dynamic_time_limit >0 :
                time_limit =int (dynamic_time_limit )

                
                
                
            solver_kwargs :dict ={"msg":False }

            if time_limit is not None :
                solver_kwargs ["timeLimit"]=time_limit 

            if MILP_SOLVER_GAP_REL is not None and MILP_SOLVER_GAP_REL >0 :
                solver_kwargs ["gapRel"]=float (MILP_SOLVER_GAP_REL )

            if MILP_SOLVER_GAP_ABS is not None and MILP_SOLVER_GAP_ABS >0 :
                solver_kwargs ["gapAbs"]=float (MILP_SOLVER_GAP_ABS )

            if MILP_SOLVER_THREADS is not None and MILP_SOLVER_THREADS >0 :
                solver_kwargs ["threads"]=int (MILP_SOLVER_THREADS )

                
            solver_options =[]
            if MILP_SOLVER_PRESOLVE =='off':
                solver_options .append ("-presolve off")
            elif MILP_SOLVER_PRESOLVE =='more':
                solver_options .append ("-preprocess more")
            if MILP_SOLVER_CUTS =='off':
                solver_options .append ("-cuts off")
            elif MILP_SOLVER_CUTS =='aggressive':
                solver_options .append ("-cuts aggressive")
            if MILP_SOLVER_HEURISTIC =='off':
                solver_options .append ("-heuristics off")
            if MILP_SOLVER_STRONG is not None and 0 <=MILP_SOLVER_STRONG <=100 :
                solver_options .append (f"-strong {int(MILP_SOLVER_STRONG)}")
            if solver_options :
                solver_kwargs ["options"]=solver_options 

            solver =pulp .PULP_CBC_CMD (**solver_kwargs )
            m .solve (solver )
        except Exception :
        
            try :
                m .solve ()
            except Exception :
            
                pass 

                
        actions =torch .zeros ((env .num_stations ,env .max_ev_per_station ),dtype =dtype ,device =device )
        for st ,ordered ,ids ,s_obs ,t ,r in active_info :
            for j in range (len (ordered )):
                val =0.0 
                key0 =(st ,j ,0 )
                if key0 in x and x [key0 ].value ()is not None :
                    try :
                        val =float (x [key0 ].value ())
                    except Exception :
                        val =0.0 
                actions [st ,j ]=float (max (-1.0 ,min (1.0 ,val /MAX_EV_POWER_KW )))

        return actions 

