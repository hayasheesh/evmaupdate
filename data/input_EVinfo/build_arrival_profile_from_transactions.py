

"""Build a 288-step arrival profile from transaction start timestamps."""

from __future__ import annotations 

import argparse 
import csv 
from dataclasses import dataclass 
from datetime import datetime ,timedelta ,timezone 
from pathlib import Path 


EPISODE_STEPS_DEFAULT =288 
STEP_MINUTES_DEFAULT =5 


@dataclass (frozen =True )
class Args :
    input_csv :str 
    output_csv :str 
    episode_steps :int 
    step_minutes :int 
    tz_offset_min :int 


def _parse_args ()->Args :
    p =argparse .ArgumentParser ()
    p .add_argument (
    "--input",
    default =str (Path (__file__ ).resolve ().parent /"transactions.csv"),
    help ="Input transactions CSV with a start_timestamp column.",
    )
    p .add_argument (
    "--output",
    default =str (Path (__file__ ).resolve ().parent /"Arrival_from_transactions.csv"),
    help ="Output arrival profile CSV path.",
    )
    p .add_argument (
    "--episode-steps",
    type =int ,
    default =EPISODE_STEPS_DEFAULT ,
    help ="Number of discrete steps in one episode.",
    )
    p .add_argument (
    "--step-minutes",
    type =int ,
    default =STEP_MINUTES_DEFAULT ,
    help ="Minutes represented by one episode step.",
    )
    p .add_argument (
    "--tz-offset-min",
    type =int ,
    default =0 ,
    help ="Timezone offset in minutes applied before binning timestamps by time of day.",
    )
    ns =p .parse_args ()
    return Args (
    input_csv =ns .input ,
    output_csv =ns .output ,
    episode_steps =ns .episode_steps ,
    step_minutes =ns .step_minutes ,
    tz_offset_min =ns .tz_offset_min ,
    )


def _parse_iso8601_z (ts :str )->datetime |None :
    """Parse an ISO-8601 timestamp, accepting the common trailing-Z UTC form."""
    if not ts :
        return None 
    s =ts .strip ()
    try :
        if s .endswith ("Z"):
        
            s =s [:-1 ]+"+00:00"
        dt =datetime .fromisoformat (s )
        if dt .tzinfo is None :
        
            dt =dt .replace (tzinfo =timezone .utc )
        return dt 
    except ValueError :
        return None 


def build_profile (
input_csv :str ,
episode_steps :int ,
step_minutes :int ,
tz_offset_min :int ,
)->list [tuple [int ,float ]]:

    counts =[0 ]*(episode_steps +1 )

    tz =timezone (timedelta (minutes =tz_offset_min ))

    with open (input_csv ,newline ="",encoding ="utf-8-sig")as f :
        reader =csv .DictReader (f )
        if reader .fieldnames is None :
            raise ValueError ("Error: invalid runtime state.")
        if "start_timestamp"not in reader .fieldnames :
            raise ValueError (
            f"Required column not found in {input_csv}: start_timestamp"
            )

        total =0 
        used =0 
        for row in reader :
            total +=1 
            dt =_parse_iso8601_z (row .get ("start_timestamp",""))
            if dt is None :
                continue 
                
            dt_local =dt .astimezone (tz )
            minute_of_day =dt_local .hour *60 +dt_local .minute 
            step =(minute_of_day //step_minutes )+1 
            if step <1 :
                step =1 
            if step >episode_steps :
                step =episode_steps 
            counts [step ]+=1 
            used +=1 

            
    out =[(i ,float (counts [i ]))for i in range (1 ,episode_steps +1 )]
    return out 


def write_csv (rows :list [tuple [int ,float ]],output_csv :str )->None :
    out_path =Path (output_csv )
    out_path .parent .mkdir (parents =True ,exist_ok =True )
    with out_path .open ("w",newline ="",encoding ="utf-8-sig")as f :
        w =csv .writer (f )
        w .writerow (["arrivaltime","workplace"])
        for step ,weight in rows :
            w .writerow ([int (step ),float (weight )])


def main ()->None :
    args =_parse_args ()
    rows =build_profile (
    input_csv =args .input_csv ,
    episode_steps =args .episode_steps ,
    step_minutes =args .step_minutes ,
    tz_offset_min =args .tz_offset_min ,
    )
    write_csv (rows ,args .output_csv )
    print (f"Wrote: {args.output_csv} (rows={len(rows)})")


if __name__ =="__main__":
    main ()
