#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 17:24:32 2023

@author: nanthawat
"""
from copy import copy
from typing import Union, List

import numpy as np
import pandas as pd

import datetime
import yfinance as yf
import sys

from syscore.dateutils import BUSINESS_DAYS_IN_YEAR, SECONDS_IN_YEAR
from syscore.pandas.pdutils import uniquets

# get random data
sys.path.append("/Users/nanthawat/Documents/GitHub/AnanCapital/Python/Data/generate_random_data")
from random_data import arbitrary_timeseries
from random_data import generate_trendy_price



def get_daily_return(stock_data):
    '''
    --------------------------
    Calculate the daily return
    --------------------------
    Input: 
        stock_data {pd.Series, pd.DataFrame}   
        
    Output:
        daily_return (same)
    '''
    daily_returns = stock_data.pct_change(1)
    return daily_returns


def get_sharp_ratio(stock_data):
    '''
    --------------------------
    Calculate the Annualized Sharp Ratio 
    - Assume: 
        1) rate: rf = 0, rb = 0, annualized sharp.
        2) number of date = 256 business day per year
        3) sqrt(256) = 16
    --------------------------
    Input: 
        stock_data {pd.Series, pd.DataFrame}   
        
    Output:
        sharp_ratio (float) 
    '''

    
    # Calculate daily returns
    daily_returns = stock_data.pct_change(1)
    
    
    # if return contain inf then remove
    daily_returns = daily_returns[~np.isinf(daily_returns)]
    
    # Calculate average and standard deviation of daily returns
    avg_return = daily_returns.mean()
    std_return = daily_returns.std()
    
    # Calculate sharp ratio
    sharp_ratio = (avg_return * 256) / (std_return * np.sqrt(256))
    
    return sharp_ratio



def drawdown(x: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
    """
    Returns a ts of drawdowns for a time series x

    >>> import datetime
    >>> df = pd.DataFrame(dict(a=[1, 2, 3, 2,1 , 4, 5], b=[2, 2, 1, 2,4 , 6, 5]), index=pd.date_range(datetime.datetime(2000,1,1),periods=7))
    >>> drawdown(df)
                  a    b
    2000-01-01  0.0  0.0
    2000-01-02  0.0  0.0
    2000-01-03  0.0 -1.0
    2000-01-04 -1.0  0.0
    2000-01-05 -2.0  0.0
    2000-01-06  0.0  0.0
    2000-01-07  0.0 -1.0
    >>> s = pd.Series([1, 2, 3, 2,1 , 4, 5], index=pd.date_range(datetime.datetime(2000,1,1),periods=7))
    >>> drawdown(s)
    2000-01-01    0.0
    2000-01-02    0.0
    2000-01-03    0.0
    2000-01-04   -1.0
    2000-01-05   -2.0
    2000-01-06    0.0
    2000-01-07    0.0
    Freq: D, dtype: float64
    """
    maxx = x.expanding(min_periods=1).max()
    return x - maxx





# get sharp Ratio from return 
def compute_sharp_ratio(daily_return):
    '''
    --------------------------
    Calculate the Annualized Sharp Ratio 
    - Assume: 
        1) rate: rf = 0, rb = 0, annualized sharp.
        2) number of date = 256 business day per year
        3) sqrt(256) = 16
    --------------------------
    Input: 
        stock_retrun {pd.Series, pd.DataFrame}   
        
    Output:
        sharp_ratio (float) 
    '''
    
    # if return contain inf then remove
    daily_return = daily_return[~np.isinf(daily_return)]
    
    # Calculate average and standard deviation of daily returns
    avg_return = daily_return.mean()
    std_return = daily_return.std()
    
    # Calculate sharp ratio
    sharp_ratio = (avg_return * 256) / (std_return * np.sqrt(256))
    
    return sharp_ratio




def calculate_sharp_ratio(symbol, window_fast, window_slow):
    """
    Calculate the new sharp ratio for the given stock symbol, using the specified fast and slow window sizes.

    Parameters:
    symbol (str): The stock symbol.
    window_fast (int): The size of the 'fast' window for computing simple moving average.
    window_slow (int): The size of the 'slow' window for computing simple moving average.

    Returns:
    float: The new sharp ratio computed using system returns.
    """

    # Set the start date to be 20 years prior to today's date
    # start_date = datetime.datetime.now() - datetime.timedelta(days=20*365)

    # Download the stock data
    stock_data = yf.download(symbol, start="2000-04-05")
    stock_data = stock_data["Close"]

    # Compute fast and slow simple moving averages (SMAs)
    sma_f = stock_data.rolling(window = window_fast).mean()
    sma_s = stock_data.rolling(window = window_slow).mean()

    # Compute the forecast as the difference between fast and slow SMAs
    forecast = sma_f - sma_s
    
    # If forecast is greater than or equal to 0, hold (buy) the stock
    hold = (forecast >= 0).astype(int)

    # Compute the daily return of the stock
    daily_return = get_daily_return(stock_data)
    
    # Compute the cumulative daily return of the stock
    cumulative_daily_return = (1 + daily_return).cumprod() - 1

    # System return is the product of daily return and holding strategy
    system_return = daily_return * hold.shift(1)
    
    # Compute the cumulative system return
    system_cum_return = (1 + system_return).cumprod() - 1

    # Create a dataframe with all the computed series
    df = stock_data.to_frame()
    df = df.assign(sma_f = sma_f,
                   sma_s = sma_s,
                   forecast = forecast,
                   hold = hold,
                   daily_return = daily_return,
                   system_return = system_return,
                   cumulative_daily_return = cumulative_daily_return,
                   system_cum_return = system_cum_return)

    # Fill NaNs in the dataframe with the first non-NaN value in each column
    df = df.fillna(method='bfill').fillna(method='ffill')

    # Compute the sharp ratios using daily returns and system returns
    original_sharp = compute_sharp_ratio(daily_return)
    new_sharp = compute_sharp_ratio(system_return)
    
    # modify
    annual_return = np.mean(system_return)*256
    annual_risk = np.std(system_return)*16
    
    # Return the computed sharp ratios
    return annual_risk # new_sharp



def calculate_sharp_ratio_fdata(Tlength, Volscale,window_fast, window_slow):
    
    # fake data
    fake_data = arbitrary_timeseries(generate_trendy_price(Nlength= 1000, Tlength=Tlength, Xamplitude=100, Volscale=Volscale)) # fake_data
    stock_data = fake_data.to_frame()
    stock_data.columns = ['Close'] 

    # indicators
    sma_f = stock_data.rolling(window = window_fast).mean()
    sma_s = stock_data.rolling(window = window_slow).mean()

    # forecast
    forecast = sma_f - sma_s

    # holding
    hold = (forecast >= 0).astype(int)

    # daily return
    daily_return = get_daily_return(stock_data) 
    cumulative_daily_return = (1 + daily_return).cumprod() - 1

    # system performance
    system_return = daily_return * hold.shift(1)

    # cum_return 
    system_cum_return = (1 + system_return).cumprod() - 1
    
    # calculate sharp Ratio
    original_sharp = compute_sharp_ratio(daily_return)
    new_sharp = compute_sharp_ratio(system_return)
    #print(f"original: {original_sharp}")
    #print(f"new: {new_sharp}")
    
    # modify
    annual_return = np.mean(system_return)*256
    annual_risk = np.std(system_return)*16
    
    return new_sharp







