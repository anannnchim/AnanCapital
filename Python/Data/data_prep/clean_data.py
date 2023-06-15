#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

-> data_prep: Contain a function for time-series preparation.

"""

import numpy as np
import pandas as pd

def check_invalid_dataframe(data):
    
    # Check NA and inf 
    na_count = data.isna().sum().sum()  
    inf_count = np.isinf(data).sum().sum()
    
    # check number of row
    number_of_rows = len(data)
    
    # check min and max
    
    
    print(f"Length: {number_of_rows}")
    print(f"Number of NA values: {na_count}")
    print(f"Number of inf values: {inf_count}")
    

