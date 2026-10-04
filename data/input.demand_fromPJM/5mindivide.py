
import argparse 
import os 
import sys 

import pandas as pd 


def parse_day (header ):
    d =pd .to_datetime (header ,errors ="coerce")
    if pd .isna (d ):
        return None 
    return d .normalize ()


def ensure_288_steps (day ):
    start =day +pd .Timedelta (minutes =5 )
    end =day +pd .Timedelta (days =1 )
    return pd .date_range (start =start ,end =end ,freq ="5min")


def split_xlsx_to_daily_csv (xlsx_path ,out_dir ,sheet =0 ):
    os .makedirs (out_dir ,exist_ok =True )
    try :
        df =pd .read_excel (xlsx_path ,sheet_name =sheet )
    except FileNotFoundError :
        print (f"[ERROR] File not found: {xlsx_path}",file =sys .stderr )
        return 0 

    if len (df .columns )==0 or len (df .columns )==1 :
        print ("[ERROR] Workbook must have time column + day columns.",file =sys .stderr )
        return 0 

    time_td =pd .to_timedelta (df .iloc [:,0 ].astype (str ),errors ="coerce")
    if time_td .isna ().all ():
        print ("[ERROR] Failed to parse the first column as time.",file =sys .stderr )
        return 0 

    written =0 
    for col in df .columns [1 :]:
        day =parse_day (col )
        if day is None :
            continue 

        vals =pd .to_numeric (df [col ],errors ="coerce")
        ts_index =day +time_td 
        day_df =pd .DataFrame ({"value":vals .values },index =ts_index ).dropna ()
        day_df =day_df .loc [day_df .index .to_series ().between (day ,day +pd .Timedelta (days =1 ),inclusive ="left")]
        if day_df .empty :
            continue 

        rep_5min =day_df .resample ("5min",label ="right",closed ="right").mean ()
        rep_5min =rep_5min .reindex (ensure_288_steps (day ))

        out_df =rep_5min .rename (columns ={"value":"demand_adjustment"}).reset_index (drop =True )
        out_df .insert (0 ,"step",range (len (out_df )))
        out_path =os .path .join (out_dir ,f"day_{day.date().isoformat()}.csv")
        out_df .to_csv (out_path ,index =False ,float_format ="%.4f")
        written +=1 

    if written ==0 :
        print ("[WARN] No day columns were found.",file =sys .stderr )
    else :
        print (f"Done. Wrote {written} daily CSVs to: {out_dir}")
    return written 


def parse_args ():
    base_dir =os .path .abspath (os .path .dirname (__file__ ))if "__file__"in globals ()else os .getcwd ()
    parser =argparse .ArgumentParser (description ="Split monthly workbook into daily 5-minute CSV files")
    parser .add_argument ("--xlsx",default =os .path .join (base_dir ,"12 2024.xlsx"),help ="Source xlsx path")
    parser .add_argument ("--out-dir",default =os .path .join (base_dir ,"output_5min"),help ="Output directory")
    parser .add_argument ("--sheet",default ="0",help ="Sheet index or name")
    return parser .parse_args ()


def main ():
    args =parse_args ()
    xlsx_path =os .path .abspath (args .xlsx )
    out_dir =os .path .abspath (args .out_dir )
    sheet =int (args .sheet )if str (args .sheet ).isdigit ()else args .sheet 
    written =split_xlsx_to_daily_csv (xlsx_path =xlsx_path ,out_dir =out_dir ,sheet =sheet )
    if written ==0 :
        sys .exit (1 )


if __name__ =="__main__":
    main ()
