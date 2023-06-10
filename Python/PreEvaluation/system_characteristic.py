#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

-> Store System Characteristic Measurement

"""
import pandas as pd

def calc_ewmac_turnover(stock_data, Lfast, Lslow=None):
    """
    Calculate the ewmac trading rule turnover
    
    input:
        1. pd.DataFrame: stock_data
        2. pd.Series: Lfast (EWMA speeds)
        3. pd.Series: Lslow (EWMA speeds)
    """
    # Select the last price 
    stock_data = stock_data.resample("1B").last()
    
    if Lslow is None:
        Lslow = 4 * Lfast
        
            
    turnover = 0
    trades = pd.Series([])
    
    for i in range(1, len(Lslow)):
        if Lfast[i] > Lslow[i]:
            trades.at[i] = 1
        else:
            trades.at[i] = 0
        
        if len(trades) >= 2 and trades.at[i-1] == 0 and trades.at[i] == 1:
            turnover += 1
            
    
    return turnover
            

def calc_holding_period(stock_data, Lfast, Lslow):
    """
    Calculate the ewmac trading rule total holding period
    
    input:
        1. pd.DataFrame: stock_data
        2. pd.Series: Lfast (EWMA speeds)
        3. pd.Series: Lslow (EWMA speeds)
    """
    hold = pd.Series([])
    turnover = 0
    holding_period = 0
    
    for i in range(1, len(Lslow)):
        if Lfast[i] > Lslow[i]:
            hold.at[i] = 1
        else:
            hold.at[i] = 0
    
    if(hold.at[i-1] == 0 and hold.at[i] == 1):
        turnover += 1
        
    
    holding_period = sum(hold[hold == 1]) 
    
    return holding_period
    
    


    