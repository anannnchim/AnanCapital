#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 10:29:39 2023

-> 01. Develop trading system using: fake data, sma crossover

@author: nanthawat
"""

# normal lib
import matplotlib.pyplot as plt
import sys
import pandas as pd
import yfinance as yf

# get indicator
sys.path.append("/Users/nanthawat/Documents/GitHub/AnanCapital/Python/Indicator")
from indicators import calc_ewmac_forecast # calc_ewmac_forecast

# get random data
sys.path.append("/Users/nanthawat/Documents/GitHub/AnanCapital/Python/Data/generate_random_data")
from random_data import arbitrary_timeseries
from random_data import generate_trendy_price

# get system characteristic
sys.path.append("/Users/nanthawat/Documents/GitHub/AnanCapital/Python/PreEvaluation")
from system_characteristic import calc_ewmac_turnover
from system_characteristic import calc_holding_period


''' 1. Get both fake and real data ''' 

# Real data: pick 4 
stock_data = yf.download("SISB.BK", start = "2019-06-07", end = "2023-06-07") # stock_data

''' 2. Generate plot for Real data: does it make sense'''  

# Prep data
sma_f = stock_data["Close"].rolling(window = 16).mean()
sma_s = stock_data["Close"].rolling(window = 64).mean()
forecast = calc_ewmac_forecast(stock_data["Close"],16,64)

# plot
fig, axs = plt.subplots(2,1,figsize = (8,6))
axs[0].plot(forecast, label = "Forecast")
axs[0].axhline(y = 0, color = 'red', linestyle = "--")

axs[1].plot( stock_data["Close"], label = "Close")
axs[1].plot(sma_f, label = "sma_f")
axs[1].plot(sma_s, label = "sma_s")

axs[0].set_title('Forecast')
axs[1].set_title('Close')

plt.tight_layout()
plt.show()

''' 3. Generate plot for fake data: does it make sense '''

fake_data = arbitrary_timeseries(generate_trendy_price(Nlength= 1000, Tlength=100, Xamplitude=100, Volscale=0.00)) # fake_data

# add date_time
fake_data.index = pd.date_range(start = "2000-01-01", periods = len(fake_data))

sma_f = fake_data.rolling(window = 64).mean()
sma_s = fake_data.rolling(window = 256).mean()

# holding 
hold = pd.Series([])    
for i in range(1, len(sma_s)):
    if sma_f[i] > sma_s[i]:
        hold.at[i] = 1
    else:
        hold.at[i] = 0

fig, axs = plt.subplots(2,1,figsize = (8,6))
axs[0].plot( fake_data, label = "Close")
axs[0].plot(sma_f, label = "sma_f")
axs[0].plot(sma_s, label = "sma_s")
axs[0].legend()  # Display the legend on the first subplot
axs[1].plot(hold, label = "holding")
axs[1].axhline(y = 0, color = 'red', linestyle = "--")
axs[1].legend()  # Display the legend on the second subplot
axs[0].set_title('Holding')
axs[1].set_title('Close')

plt.tight_layout()
plt.show()

avg_holding_period = calc_holding_period(fake_data, sma_f, sma_s) /  calc_ewmac_turnover(fake_data, sma_f,sma_s)
turnover =  calc_ewmac_turnover(fake_data, sma_f,sma_s)
holding_period = calc_holding_period(fake_data, sma_f, sma_s)
trades_per_year = turnover / 4

print(f'Holding period: {holding_period}')
print(f'Turnover: {turnover}')
print(f'Avg.holding period: {avg_holding_period}')
print(f'trade/year: {trades_per_year}')

''' 5. Check turnover for fake_data given a pair of moving average'''

# Simulation
def simulate_ma_strategy(window_fast, window_slow, Tlength):
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
    trades_per_year = turnover / 4

    # Collect data in a dictionary
    data = {
        'Avg. holding period': [avg_holding_period],
        'Trades/year': [trades_per_year]
    }

    # Convert the dictionary to a dataframe
    df = pd.DataFrame(data)

    return df

# Run
Tlength = [5, 21, 64, 128, 256]
pairs = [(2, 8), (4, 16), (8, 32), (16, 64), (32, 128), (64, 256)]
dataframes = []

# iterate over all pairs
for i, pair in enumerate(pairs):
    # iterate over all Tlength values
    for T in Tlength:
        df = simulate_ma_strategy(pair[0], pair[1], Tlength=T)
        df['Pair'] = f'Pair_{i+1}'
        df['Tlength'] = T
        dataframes.append(df)

# Concatenate all dataframes into a single one
final_df = pd.concat(dataframes)
print(final_df)


''' -------------------------------------------------------------------------------- ''' 
# clean code above
# check and verify with step 3 
# check sensitivity with real data
# Prune any window side that are likely to be too expensive

# move to return part

''' -------------------------------------------------------------------------------- ''' 

''' 6. Check Sharp. Ratio'''
# understand correlation structure to work out best window_size 

"""
   Note
   1. Amptitude is not affect turnover, vol.scale affect, when Nlength large enough: avg. holding not change (N:1000to5000)
   - number of avg. holding related to Tlength
   2. need to calculate exact buy and sell and count day. 
   - create a  table window_size x trendlength(rob said not important ) [Turnover]
   
   3. simulate to get a table and check turnover, holding given a pair. 
   4. calculate sharp R. given a pair. 
   - create a function ?
   - create a function to compute sharp R. (boxplot page 90)
   - vdo: (trend length x window size) : [avg. sharp R.]
   
   5. make a function for a plot 
   
"""


''' 6. fit allocation using real data'''

# calc_ewmac_turnover(fake_data,sma_f, sma_s)

# And you want to convert it to a DatetimeIndex starting from 2022-01-01:

# Now fake_data has a DatetimeIndex

# check properties
# how trend length relates to profitability of window_size
# columns NLength month to year
# row: window (half or third of actual trend)
# turnver depend on window size not Tlength


# holding_period = 250 / turnover
#print("Turnover: {} times \nHolding period: {} days".format(turnover, round(holding_period)))

# -------------------------------------#  

''' 5. Understand system '''

'''
create a function to calculate sharp Ratio given forecast 
'''

















