#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

-> Store System Characteristic Measurement

"""
import pandas as pd
import datetime
import yfinance as yf
import sys

# get random data
sys.path.append("/Users/nanthawat/Documents/GitHub/AnanCapital/Python/Data/generate_random_data")
from random_data import arbitrary_timeseries
from random_data import generate_trendy_price


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
    
    


    

# Simulation
def simulate_sma_turnover_with_fdata(window_fast, window_slow, Tlength):
    """
    Simulate a moving average trading strategy given the sizes of the fast and slow windows.
    
    Inputs:
        - window_fast: int, size of the fast moving average window.
        - window_slow: int, size of the slow moving average window.
        - TLength: int, length of directional move 
        
    Outputs:
        - df: pd.DataFrame, dataframe containing calculated metrics of the trading strategy.
    """
    # Create arbitrary time series
    fake_data = arbitrary_timeseries(generate_trendy_price(Nlength= 1000, Tlength=Tlength, Xamplitude=100, Volscale=0.15)) 

    # Add date_time
    fake_data.index = pd.date_range(start = "2000-01-01", periods = len(fake_data))

    # Calculate fast and slow moving averages
    sma_f = fake_data.rolling(window = window_fast).mean()
    sma_s = fake_data.rolling(window = window_slow).mean()

    # Calculate holding series
    hold = pd.Series([])    
    for i in range(1, len(sma_s)):
        if sma_f[i] > sma_s[i]:
            hold.at[i] = 1
        else:
            hold.at[i] = 0

    # Calculate metrics
    avg_holding_period = calc_holding_period(fake_data, sma_f, sma_s) / calc_ewmac_turnover(fake_data, sma_f,sma_s)
    turnover = calc_ewmac_turnover(fake_data, sma_f,sma_s)
    holding_period = calc_holding_period(fake_data, sma_f, sma_s)
    turnover_per_year = turnover / 4

    # Collect data in a dictionary
    data = {
        'Avg. holding period': [avg_holding_period],
        'Turnover per year ': [turnover_per_year]
    }

    # Convert the dictionary to a dataframe
    df = pd.DataFrame(data)

    return df





def simulate_sma_turnover_with_rdata(symbol, window_fast, window_slow):
    """
    Simulate sma strategy with real data and get the turnover
    
    Inputs:
        - stock_data: pd.DataFrame, stock_data that contain "Close"
        - window_fast: int, size of the fast moving average window.
        - window_slow: int, size of the slow moving average window.
        - TLength: int, length of directional move 
        
    Outputs:
        - df: pd.DataFrame, dataframe containing calculated metrics of the trading strategy.
    """
 
    # Get real data 
    start_date = datetime.datetime.now() - datetime.timedelta(days=20*365)

    # Download data from 4 years ago to today
    stock_data = yf.download(symbol, start=start_date)

    # Add date_time
    
    # Calculate fast and slow moving averages
    sma_f = stock_data["Close"].rolling(window = window_fast).mean()
    sma_s = stock_data["Close"].rolling(window = window_slow).mean()

    # Calculate holding series
    hold = pd.Series([])    
    for i in range(1, len(sma_s)):
        if sma_f[i] > sma_s[i]:
            hold.at[i] = 1
        else:
            hold.at[i] = 0

    # Calculate metrics
    avg_holding_period = calc_holding_period(stock_data, sma_f, sma_s) /  calc_ewmac_turnover(stock_data, sma_f,sma_s)
    turnover =  calc_ewmac_turnover(stock_data, sma_f,sma_s)
    holding_period = calc_holding_period(stock_data, sma_f, sma_s)
    trades_per_year = turnover / (len(stock_data) /256)

    # Collect data in a dictionary
    data = {
        'Avg. holding period': [avg_holding_period],
        'Trades/year': [trades_per_year],
        "turnover" :[turnover],
        "holding" :[sum(hold)],
    }

    # Convert the dictionary to a dataframe
    df = pd.DataFrame(data)

    return df