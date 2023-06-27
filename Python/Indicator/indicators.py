#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

-> Store all indicators

"""
import sys
import yfinance as yf
import pandas as pd
import numpy as np

sys.path.append('/Users/nanthawat/Desktop/Python/pysystemtrade')
from sysquant.estimators.vol import robust_vol_calc

# Function 

def calc_ewmac_forecast(price, Lfast, Lslow=None):
    """
    Calculate the ewmac trading rule forecast, given a price and EWMA speeds Lfast, Lslow and vol_lookback

    """
    ## price: This is the stitched price series
    ## We can't use the price of the contract we're trading, or the volatility will be jumpy
    ## And we'll miss out on the rolldown. See https://qoppac.blogspot.com/2015/05/systems-building-futures-rolling.html

   # price = price.resample("1B").last()
    if Lslow is None:
        Lslow = 4 * Lfast

    ## We don't need to calculate the decay parameter, just use the span directly

    fast_ewma = price.ewm(span=Lfast).mean()
    slow_ewma = price.ewm(span=Lslow).mean()
    raw_ewmac = fast_ewma - slow_ewma
    
    vol = robust_vol_calc(price.diff())

    return raw_ewmac / vol



def calc_volatility_fdata(price, Lfast, Lslow=None):
    """
    Calculate the ewmac trading rule forecast, given a price and EWMA speeds Lfast, Lslow and vol_lookback

    """
    ## price: This is the stitched price series
    ## We can't use the price of the contract we're trading, or the volatility will be jumpy
    ## And we'll miss out on the rolldown. See https://qoppac.blogspot.com/2015/05/systems-building-futures-rolling.html

   # price = price.resample("1B").last()
    if Lslow is None:
        Lslow = 4 * Lfast

    ## We don't need to calculate the decay parameter, just use the span directly

    fast_ewma = price.ewm(span=Lfast).mean()
    slow_ewma = price.ewm(span=Lslow).mean()
    raw_ewmac = fast_ewma - slow_ewma
    
    volatility = np.sqrt(((price - price.shift(1))**2).ewm(span = 36).mean())
    
    return volatility   #/ vol



def calc_sma_forecast(stock_data, Lfast, Lslow=None):
    """
    Calculate the ewmac trading rule forecast, given a price and EWMA speeds Lfast, Lslow and vol_lookback

    """
    ## price: This is the stitched price series
    ## We can't use the price of the contract we're trading, or the volatility will be jumpy
    ## And we'll miss out on the rolldown. See https://qoppac.blogspot.com/2015/05/systems-building-futures-rolling.html

   #  stock_data = stock_data.resample("1B").last()
    if Lslow is None:
        Lslow = 4 * Lfast

    ## We don't need to calculate the decay parameter, just use the span directly

    sma_f = stock_data.rolling(window = Lfast).mean()
    sma_s = stock_data.rolling(window = Lslow).mean()
    
    raw_sma = sma_f - sma_s
    
    vol = robust_vol_calc(stock_data.diff())
    
    return raw_sma / vol


# need to adjust

def create_stock_df(stock_symbol, start_date, end_date, window_fast, window_slow):
    # Download stock data
    stock_data = yf.download(stock_symbol, start = start_date, end = end_date)["Close"]

    # Calculate SMAs
    sma_f = stock_data.rolling(window = window_fast).mean()
    sma_s = stock_data.rolling(window = window_slow).mean()

    # Calculate forecast ? 
    forecast = calc_sma_forecast(stock_data, window_fast, window_slow)

    # Calculate hold series
    hold = pd.Series(index=forecast.index) 
    for i in forecast.index:
        if forecast[i] > 0:
            hold[i] = 1
        else:
            hold[i] = 0

    # Create DataFrame
    df = stock_data.to_frame()
    df = df.assign(sma_f = sma_f,
                   sma_s = sma_s,
                   forecast = forecast,
                   hold = hold)

    return df














