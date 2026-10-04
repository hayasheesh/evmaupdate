"""
readcsv.py
==========
CSV loading utilities for demand-adjustment (AG request) data.

- `load_multiple_demand_files`: load all CSV files in a directory and split
  them into train/test groups.
- `get_random_demand_episode`: sample one episode-length demand sequence from
  a preloaded data pool.

CSV format:
  Each row is one 5-minute time step of demand adjustment in kW.
  The `demand_adjustment` column is preferred; otherwise the first numeric
  column is used.

Normalization:
  Each file is linearly scaled into the demand target range configured in
  `EnvConfig.DEMAND_TARGET_MIN_KW` / `EnvConfig.DEMAND_TARGET_MAX_KW`.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import List ,Dict
import glob

import numpy as np
import pandas as pd

from EnvConfig import DEMAND_TARGET_MIN_KW ,DEMAND_TARGET_MAX_KW

TARGET_MIN =float (DEMAND_TARGET_MIN_KW )
TARGET_MAX =float (DEMAND_TARGET_MAX_KW )


def load_multiple_demand_files (directory :str =None ,train_split :int =25 )->Dict [str ,List [np .ndarray ]]:
   """
   Load all demand CSV files from a directory and split them into train/test sets.
   """
   from EnvConfig import DEMAND_ADJUSTMENT_DIR

   if directory is None :
       directory =DEMAND_ADJUSTMENT_DIR
   return _load_multiple_demand_files_cached (os .path .abspath (directory ),int (train_split ))


def load_multiple_demand_files_with_labels (directory :str =None ,train_split :int =25 )->Dict [str ,List [dict ]]:
   """
   Load demand CSVs and keep file/date metadata for day-context arrival sampling.
   """
   from EnvConfig import DEMAND_ADJUSTMENT_DIR

   if directory is None :
       directory =DEMAND_ADJUSTMENT_DIR
   return _load_multiple_demand_files_with_labels_cached (os .path .abspath (directory ),int (train_split ))


def _day_label_from_path (file_path :str )->str |None :
   name =os .path .basename (file_path )
   stem ,_ext =os .path .splitext (name )
   if stem .startswith ("day_"):
       return stem [len ("day_"):]
   return stem or None


def _load_and_normalize_demand_file (file_path :str )->np .ndarray :
   df =pd .read_csv (file_path )

   if 'demand_adjustment'in df .columns :
       data =df ['demand_adjustment'].to_numpy (float )
   else :
       numeric_cols =df .select_dtypes (include =['number']).columns
       if len (numeric_cols )==0 :
           raise ValueError (f"No numeric demand column found in {file_path}")
       data =df [numeric_cols [0 ]].to_numpy (float )

   data_288 =data [:288 ]if len (data )>=288 else data
   if len (data_288 )==0 :
       raise ValueError (f"Demand CSV has no rows: {file_path}")

   a =float (data_288 .min ())
   b =float (data_288 .max ())

   if b -a <1e-6 :
       data_288 =np .full_like (data_288 ,(TARGET_MIN +TARGET_MAX )/2.0 )
   else :
       scale =(TARGET_MAX -TARGET_MIN )/(b -a )
       offset =TARGET_MIN /scale -a
       data_288 =scale *(data_288 +offset )
   return data_288


@lru_cache (maxsize =8 )
def _load_multiple_demand_files_cached (directory :str ,train_split :int )->Dict [str ,List [np .ndarray ]]:
   """Cached implementation for repeated train/interim-test demand loading."""
   if not os .path .exists (directory ):
       error_message =f"Demand directory does not exist: {directory}"
       print (error_message )
       raise FileNotFoundError (error_message )

   pattern =os .path .join (directory ,"*.csv")
   files =sorted (glob .glob (pattern ))

   if len (files )==0 :
       error_message =f"No demand CSV files found in directory: {directory}"
       print (error_message )
       raise FileNotFoundError (error_message )

   train_data =[]
   test_data =[]

   for i ,file_path in enumerate (files ,start =1 ):
       data_288 =_load_and_normalize_demand_file (file_path )

       if i <=train_split :
           train_data .append (data_288 )
       else :
           test_data .append (data_288 )

   return {
   'train':train_data ,
   'test':test_data
   }


@lru_cache (maxsize =8 )
def _load_multiple_demand_files_with_labels_cached (directory :str ,train_split :int )->Dict [str ,List [dict ]]:
   if not os .path .exists (directory ):
       error_message =f"Demand directory does not exist: {directory}"
       print (error_message )
       raise FileNotFoundError (error_message )

   pattern =os .path .join (directory ,"*.csv")
   files =sorted (glob .glob (pattern ))

   if len (files )==0 :
       error_message =f"No demand CSV files found in directory: {directory}"
       print (error_message )
       raise FileNotFoundError (error_message )

   train_data =[]
   test_data =[]
   for i ,file_path in enumerate (files ,start =1 ):
       payload ={
       "series":_load_and_normalize_demand_file (file_path ),
       "date":_day_label_from_path (file_path ),
       "path":file_path ,
       }
       if i <=train_split :
           train_data .append (payload )
       else :
           test_data .append (payload )

   return {
   'train':train_data ,
   'test':test_data
   }


def get_random_demand_episode (data_list :List [np .ndarray ],episode_steps :int =288 )->np .ndarray :
   """
   Return one random episode-length demand sequence from a preloaded data list.
   """
   import random

   if not data_list :
       raise ValueError ("data_list is empty; cannot sample a demand episode.")

   data =random .choice (data_list )

   if len (data )>=episode_steps :
       return data [:episode_steps ]
   else :
       padded =np .zeros (episode_steps ,dtype =float )
       padded [:len (data )]=data
       return padded
