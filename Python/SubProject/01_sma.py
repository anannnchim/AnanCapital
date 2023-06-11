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
import numpy as np
import datetime

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
from system_characteristic import simulate_sma_turnover_with_fdata
from system_characteristic import simulate_sma_turnover_with_rdata

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

fake_data = arbitrary_timeseries(generate_trendy_price(Nlength= 1000, Tlength=256, Xamplitude=100, Volscale=0.15)) # fake_data

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

''' 4. Check turnover for fake_data given a pair of moving average'''

simulate_sma_turnover_with_fdata(64,256,50)

# Run
Tlength = [5, 21, 64, 128, 256]
pairs = [(2, 8), (4, 16), (8, 32), (16, 64), (32, 128), (64, 256)]
# pairs = [(64, 256), (85, 256), (128, 256)]
dataframes = []

# iterate over all pairs
for i, pair in enumerate(pairs):
    # iterate over all Tlength values
    for T in Tlength:
        df = simulate_sma_turnover_with_fdata(pair[0], pair[1], Tlength=T)
        df['Pair'] = f'Pair_{i+1}'
        df['Tlength'] = T
        dataframes.append(df)

# Concatenate all dataframes into a single one
final_df = pd.concat(dataframes)
print(final_df)

'''
Result: fake data
# (2:8)
   Avg. holding period  Trades/year    Pair  Tlength
0             4.930000        25.00  Pair_1        5
0             6.863014        18.25  Pair_1       21 # <<< can capture this trend Length
0             3.406667        37.50  Pair_1       64
0             2.911243        42.25  Pair_1      128
0             2.816092        43.50  Pair_1      256

# (4 : 16) 
0             4.939394        24.75  Pair_2        5
0            18.481481         6.75  Pair_2       21 # <<<
0             9.236364        13.75  Pair_2       64 # <<<
0             5.134021        24.25  Pair_2      128
0             4.420561        26.75  Pair_2      256

# (8 : 32)
0             4.989796        24.50  Pair_3        5
0            20.375000         6.00  Pair_3       21 # <<<
0            40.500000         3.00  Pair_3       64 # <<< 
0            40.250000         3.00  Pair_3      128
0             9.460000        12.50  Pair_3      256


# (16 : 64)
0             5.153846        22.75  Pair_4        5
0            20.608696         5.75  Pair_4       21
0            51.444444         2.25  Pair_4       64 # <<< mathc best
0           116.500000         1.00  Pair_4      128
0           117.000000         1.00  Pair_4      256


# ( 32 : 128)
0             5.920000        18.75  Pair_5        5
0            19.818182         5.50  Pair_5       21
0            62.285714         1.75  Pair_5       64
0           109.000000         1.00  Pair_5      128 # <<< match best
0           217.000000         0.50  Pair_5      256 



# (64 : 256)
0             7.282609        11.50  Pair_6        5
0            20.555556         4.50  Pair_6       21
0            52.000000         1.75  Pair_6       64
0           127.666667         0.75  Pair_6      128 # <<<   
0           116.666667         0.75  Pair_6      256 # <<<


# Result: Real data
Pair           Avg. holding period  Trades/year  turnover  holding
(2, 8)                6.401865    20.431629     389.6   2493.0                                                          
(4, 16)              13.321197    10.069698     192.0   2557.2
(8, 32)              28.625636     4.791888      91.4   2609.4
(16, 64)             53.937053     2.569563      49.0   2634.8
(32, 128)           125.654967     1.111587      21.2   2627.6
(64, 256)           250.340280     0.588288      11.2   2786.6

'''
''' -------------------------------------------------------------------------------- ''' 
''' 6. Check turnover for real data given a pair of moving average'''

# Calculate date 20 years ago
start_date = datetime.datetime.now() - datetime.timedelta(days=20*365)

# Download data from 4 years ago to today
stock_data = yf.download("SCC.BK", start=start_date)

# Prep data
sma_f = stock_data["Close"].rolling(window = 2).mean()
sma_s = stock_data["Close"].rolling(window = 8).mean()
forecast = calc_ewmac_forecast(stock_data["Close"],2,8)

# holding 
hold = pd.Series([])    
for i in range(1, len(sma_s)):
    if sma_f[i] > sma_s[i]:
        hold.at[i] = 1
    else:
        hold.at[i] = 0

# metrix
avg_holding_period = calc_holding_period(stock_data, sma_f, sma_s) /  calc_ewmac_turnover(stock_data, sma_f,sma_s)
turnover =  calc_ewmac_turnover(stock_data, sma_f,sma_s)
holding_period = calc_holding_period(stock_data, sma_f, sma_s)
trades_per_year = turnover / (len(stock_data) /256)

# plot
fig, axs = plt.subplots(3,1,figsize = (8,6))
axs[0].plot(forecast, label = "Forecast")
axs[0].axhline(y = 0, color = 'red', linestyle = "--")

axs[1].plot( stock_data["Close"], label = "Close")
axs[1].plot(sma_f, label = "sma_f")
axs[1].plot(sma_s, label = "sma_s")
axs[2].plot(hold, label = "holding")

axs[0].set_title('Forecast')
axs[1].set_title('Close')

plt.tight_layout()
plt.show()

print(f'Avg.holding period: {avg_holding_period}')
print(f'trade/year: {trades_per_year}')
print(f'Turnover: {turnover}')
print(f'Holding period: {holding_period}')


# Run simulate

symbol_list = ["AOT.BK","PTT.BK","SCC.BK","KBANK.BK", "DELTA.BK"]
# pairs = [(2, 8), (4, 16), (8, 32), (16, 64), (32, 128), (64, 256)]
pairs = [(64, 256), (85, 256), (128, 256)]

dataframes = []

# iterate over all pairs
for i, pair in enumerate(pairs):
    # iterate over all Tlength values
    for symbol in symbol_list:
        df = simulate_sma_turnover_with_rdata(symbol, pair[0], pair[1])
        df['Pair'] = f'{pair[0],pair[1]}'
        df['Symbol'] = symbol
        dataframes.append(df)

# Concatenate all dataframes into a single one
final_df = pd.concat(dataframes)
print(final_df.groupby("Pair").mean())
print(final_df)

'''
Pair           Avg. holding period  Trades/year  turnover  holding
(2, 8)                6.401865    20.431629     389.6   2493.0                                                          
(4, 16)              13.321197    10.069698     192.0   2557.2
(8, 32)              28.625636     4.791888      91.4   2609.4

(16, 64)             53.937053     2.569563      49.0   2634.8 <<< capture Medium
(32, 128)           125.654967     1.111587      21.2   2627.6 <<< capture Long
(64, 256)           250.340280     0.588288      11.2   2786.6 <<< capture Long


# Check variation of pair 16:64 >> same
          Avg. holding period  Trades/year  turnover  holding
Pair                                                         
(16, 64)            53.937053     2.569563      49.0   2634.8 
(21, 64)            59.010313     2.350177      44.8   2638.0
(32, 64)            60.052206     2.318957      44.2   2642.6


# Check variation of pair 64:256 >> same

            Avg. holding period  Trades/year  turnover  holding
Pair                                                           
(64, 256)            250.340280     0.588288      11.2   2786.6
(85, 256)            285.935556     0.515017       9.8   2784.8
(128, 256)           258.671779     0.566625      10.8   2752.4

* each stock has a consistent result. 

Summary:
    1. choice of {1/4, 1/3, 1/4} give the same expected holding and turnover
    2. pair (16,64) capture 3 month trend or  avg.holding at 53 <<< System S1  
    -  pair (32,128) capture 6 month trend or avg.holding at 125 
    -  pair (64,256) capture 1 year trend or avg.holding at 250
    3. fake data give similar properties as real data 
'''




''' -------------------------------------------------------------------------------- ''' 
# move to return part
# * Prune any window side that are likely to be too expensive
''' 6. Check Sharp. Ratio'''
# calculate sharp to fake data 
# calculate sharp for real data 
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

















