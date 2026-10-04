

"""Convert ACN session JSON files into dwell-time and required-SoC tables."""

import json 
from pathlib import Path 

import pandas as pd 


BASE_DIR =Path (__file__ ).resolve ().parent
INPUT_FILES =[BASE_DIR /f"acndata_sessions ({idx}).json"for idx in range (2 ,5 )]
OUTPUT_DIR =BASE_DIR
# Must match EnvConfig.EV_CAPACITY: required_soc_percent is consumed by EVEnv as
# a percentage of the simulated battery, so converting kWhRequested with a
# different denominator silently rescales every EV's energy demand. This was
# 50.0 while EnvConfig.EV_CAPACITY was 100.0, which doubled the effective
# demand (mean 17.71 kWh requested -> 35.43 % -> 35.43 kWh on a 100 kWh pack)
# and clipped 5.57 % of sessions at 100 %.
CAPACITY_KWH =100.0
ROUND_MIN =5


def _to_dt_utc (series ):
    return pd .to_datetime (series ,utc =True ,errors ="coerce")

def latest_user_input (uinputs ):
    """Return the latest user input record attached to one charging session."""
    if not isinstance (uinputs ,list )or len (uinputs )==0 :
        return {}
    try :
        uidf =pd .DataFrame (uinputs )
        if "modifiedAt"in uidf .columns :
            uidf ["modifiedAt"]=pd .to_datetime (uidf ["modifiedAt"],utc =True ,errors ="coerce")
            uidf =uidf .sort_values ("modifiedAt")
            return uidf .iloc [-1 ].to_dict ()
        else :
        
            return uinputs [-1 ]
    except Exception :
        return uinputs [-1 ]

def quantize_minutes (value_min ,step =5 ):
    """Round a duration in minutes to the nearest configured step."""
    if pd .isna (value_min ):
        return None 
    return int (round (float (value_min )/step )*step )

def compute_required_kwh (row ):
    """Infer requested energy from user input, mileage request, or delivered kWh."""
    ui =row .get ("_latest_ui",{})or {}
    
    kwh_req =ui .get ("kWhRequested",None )
    kwh_req =float (kwh_req )if kwh_req is not None else None 

    
    whpm =ui .get ("WhPerMile",None )
    miles =ui .get ("milesRequested",None )
    if whpm is not None and miles is not None :
        try :
            kwh_from_miles =float (whpm )*float (miles )/1000.0 
        except Exception :
            kwh_from_miles =None 
    else :
        kwh_from_miles =None 

        
    kwh_delivered =row .get ("kWhDelivered",None )
    kwh_delivered =float (kwh_delivered )if kwh_delivered is not None else None 

    
    candidates =[kwh_req ,kwh_from_miles ,kwh_delivered ]
    val =next ((v for v in candidates if v is not None and v >=0 ),None )

    
    if val is None and kwh_delivered is not None :
        val =kwh_delivered 

        
    if val is not None and kwh_delivered is not None :
        val =max (val ,kwh_delivered )

    return val 

def _load_json_payload (path :Path )->dict :
    """Load a JSON payload and repair a few common truncated-export endings."""
    text =path .read_text (encoding ="utf-8")
    try :
        return json .loads (text )
    except json .JSONDecodeError as exc :
        trimmed =text .rstrip ()
        
        if trimmed .endswith (","):
            trimmed =trimmed [:-1 ].rstrip ()
            
        for suffix in ("\n  ]\n}\n","\n]\n}\n","\n  ]\n"):
            try_text =trimmed +suffix 
            try :
                return json .loads (try_text )
            except json .JSONDecodeError :
                continue 
                
        last_brace =trimmed .rfind ("}")
        if last_brace !=-1 :
            repaired =trimmed [:last_brace +1 ]+"\n  ]\n}\n"
            try :
                return json .loads (repaired )
            except json .JSONDecodeError :
                pass 
        raise exc 


def process_sessions (input_json :Path )->pd .DataFrame :
    """Create the EV profile rows used by the environment sampler."""
    payload =_load_json_payload (input_json )

    items =payload .get ("_items",payload )
    df =pd .json_normalize (items )

    if df .empty :
        return pd .DataFrame (columns =["ev_id","connection_minutes_5min","required_soc_percent"])

        
        
    df ["connectionTime"]=_to_dt_utc (df .get ("connectionTime"))
    df ["disconnectTime"]=_to_dt_utc (df .get ("disconnectTime"))

    
    df ["dwell_min"]=(df ["disconnectTime"]-df ["connectionTime"]).dt .total_seconds ()/60.0 

    
    latest_list =[]
    for _ ,row in df .iterrows ():
        latest_list .append (latest_user_input (row .get ("userInputs",None )))
    df ["_latest_ui"]=latest_list 

    
    req_kwh =df .apply (compute_required_kwh ,axis =1 )
    df ["required_soc_percent"]=(
    pd .Series (req_kwh ,index =df .index ).astype (float )/CAPACITY_KWH *100.0 
    )
    
    df ["required_soc_percent"]=df ["required_soc_percent"].clip (lower =0 ,upper =100 )

    
    df ["connection_minutes_5"]=df ["dwell_min"].apply (lambda x :quantize_minutes (x ,ROUND_MIN ))

    
    out =pd .DataFrame ({
    "ev_id":range (1 ,len (df )+1 ),
    "connection_minutes_5min":(df ["connection_minutes_5"]//5 ),
    "required_soc_percent":df ["required_soc_percent"].round (4 ),
    })

    
    out =out .dropna (subset =["connection_minutes_5min","required_soc_percent"])
    return out 


def main ():
    OUTPUT_DIR .mkdir (parents =True ,exist_ok =True )

    combined_frames =[]
    for input_path in INPUT_FILES :
        if not input_path .exists ():
            print (f"Skip (not found): {input_path}")
            continue 

        out_df =process_sessions (input_path )
        output_path =OUTPUT_DIR /f"{input_path.stem}_ev_sessions_soc.csv"
        out_df .to_csv (output_path ,index =False ,encoding ="utf-8-sig")
        print (f"Wrote: {output_path}  rows={len(out_df)}")

        if not out_df .empty :
            combined_frames .append (out_df .assign (source_file =input_path .name ))

    if combined_frames :
        combined_df =pd .concat (combined_frames ,ignore_index =True )
        combined_path =OUTPUT_DIR /"ev_sessions_soc_all.csv"
        combined_df .to_csv (combined_path ,index =False ,encoding ="utf-8-sig")
        print (f"Wrote: {combined_path}  rows={len(combined_df)}")
    else :
        print ("No input files processed.")


if __name__ =="__main__":
    main ()
