#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 10:29:39 2023

-> 01. Develop trading system using: fake data, sma crossover

@author: nanthawat
"""

# plot 
import matplotlib.pyplot as plt

''' 1. Get both fake and real data ''' 

# Real data
import yfinance as yf

stock_data = yf.download("SISB.BK", start = "2022-06-07", end = "2023-06-07") # stock_data

# Fake data
import os
os.chdir('/Users/nanthawat/Documents/GitHub/AnanCapital/Python/Data/generate_random_data') 
os.getcwd()
from random_data import arbitrary_timeseries
from random_data import generate_trendy_price


''' 2. Import Indicator '''  # DONE
 
# get indicator
os.chdir('/Users/nanthawat/Documents/GitHub/AnanCapital/Python/Indicator') 
from indicators import calc_ewmac_forecast # calc_ewmac_forecast


''' 3. Generate plot for Real data: does it make sense'''  # DONE 

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



''' 4. Generate plot for fake data: does it make sense '''
# Prep data
import pandas as pd
fake_data = arbitrary_timeseries(generate_trendy_price(Nlength=400, Tlength=50, Xamplitude=40.0, Volscale=0.00)) # fake_data

# add date_time
fake_data.index = pd.date_range(start = "2000-01-01", periods = len(fake_data))

sma_f = fake_data.rolling(window = 16).mean()
sma_s = fake_data.rolling(window = 64).mean()

# plot
plt.figure(figsize=(10,6))

# Plot the original data
plt.plot(fake_data, label='Original Data')

# Plot the simple moving averages

plt.plot(sma_f, label='SMA Fast')
plt.plot(sma_s, label='SMA Slow')

# Add a legend
plt.legend()

# Show the plot
plt.show()


''' 5. Check turnover for fake_data given a pair of moving average'''


"""
    stock data vs. fake
    - pandas.core.frame.DataFrame
    - pandas.core.series.Series
    * convert series to dataframe
    
"""
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

# - measure profit by sharp Ratio
'''
- make a function for plot 
- create a function: Count how many time trade per year 
- create a function to compute sharp R. (boxplot page 90    q)
- 

'''

# Get idea of how fast different window size will trade
# 2. Turn over


# Prune any window side that are likely to be too expensive

# understand correlation structure to work out best window_size 










