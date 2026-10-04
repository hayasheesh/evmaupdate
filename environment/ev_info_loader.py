"""
ev_info_loader.py
=================
Loader for the plug-in SoC distribution.

`soc_arrival_distribution.csv` (built by data/input_EVinfo/Arrivesoc.py) holds
SoC on arrival at the DC fast chargers of the EPFL DESL Level-3 dataset
(github.com/DESL-EPFL/Level-3-EV-charging-dataset, 1,878 sessions, Apr 2022 -
Jul 2023): column 0 is an SoC value in percent, column 2 its share. EVEnv draws
the plug-in SoC of every non-residential session from it by inverse transform.
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Tuple

import numpy as np


def _read_numeric_column (path :str ,value_index :int )->Tuple [np .ndarray ,np .ndarray ]:
    """
    Read a CSV and return column 0 as integer keys and `value_index` as values.

    Rows that cannot be parsed numerically are ignored. This keeps header rows,
    blank lines, and annotation rows from affecting the empirical distributions.
    """
    file_path =Path (path )
    keys =[]
    vals =[]
    with file_path .open (newline ="",encoding ="utf-8-sig")as f :
        reader =csv .reader (f )
        header_skipped =False
        for row in reader :
            if not header_skipped :
                header_skipped =True
                continue
            if len (row )<=value_index :
                continue
            try :
                key =int (row [0 ].strip ())
                val =float (row [value_index ])
            except ValueError :
                continue
            keys .append (key )
            vals .append (val )
    return np .asarray (keys ,dtype =np .int64 ),np .asarray (vals ,dtype =np .float64 )


def load_arrival_soc_cdf (path :str )->Tuple [np .ndarray ,np .ndarray ]:
    """
    Return the plug-in SoC values and their CDF.

    EVEnv draws `u ~ Uniform(0, 1)` and takes `soc_values[searchsorted(cdf, u)]`.
    """
    return _cached_load_arrival_soc_cdf (str (Path (path ).resolve ()))


@lru_cache (maxsize =8 )
def _cached_load_arrival_soc_cdf (path :str )->Tuple [np .ndarray ,np .ndarray ]:
    soc_values ,percents =_read_numeric_column (path ,2 )
    if len (percents )==0 :
        raise ValueError (f"SoC arrival distribution has no usable rows: {path}")
    percents =np .clip (percents ,a_min =0.0 ,a_max =None )
    if float (percents .sum ())<=0 :
        raise ValueError (f"SoC arrival distribution has zero total weight: {path}")
    cdf =np .cumsum (percents )
    cdf /=cdf [-1 ]
    soc_values .setflags (write =False )
    cdf .setflags (write =False )
    return soc_values ,cdf
