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

stock_data = yf.download("SISB.BK", start = "2021-06-07", end = "2023-06-07") # stock_data

# Fake data
import os
os.getcwd()
from random_data import arbitrary_timeseries
from random_data import generate_trendy_price
os.chdir('/Users/nanthawat/Documents/GitHub/AnanCapital/Python/Data/generate_random_data') 

fake_data = arbitrary_timeseries(generate_trendy_price(Nlength=400, Tlength=60, Xamplitude=100.0, Volscale=0.10)) # fake_data


''' 2. Import Indicator ''' 

# get indicator
os.chdir('/Users/nanthawat/Documents/GitHub/AnanCapital/Python/Indicator') 
from indicators import calc_ewmac_forecast # calc_ewmac_forecast


''' 3. Generate plot for Real data ''' 

# Prep data
sma_f = stock_data["Close"].rolling(window = 0).mean()
sma_s = stock_data["Close"].rolling(window = 200).mean()

forecast = calc_ewmac_forecast(stock_data["Close"],1,200)


# plot
fig, axs = plt.subplots(2,1,figsize = (13,8))
axs[0].plot(forecast, label = "Forecast")
axs[0].axhline(y = 0, color = 'red', linestyle = "--")

axs[1].plot( stock_data["Close"], label = "Close")
axs[1].plot(sma_f, label = "sma_f")
axs[1].plot(sma_s, label = "sma_s")

axs[0].set_title('Forecast')
axs[1].set_title('Close')

plt.tight_layout()
plt.show()



''' 4. Generate plot for fake data  << Pending '''

# Prep data
import pandas as pd

# #  #

# Pending: got NA 
fake_data.index = pd.to_datetime(stock_data.iloc[:len(fake_data), 0])
forecast = calc_ewmac_forecast(fake_data, 0, 40)

# # # 

sma_f = fake_data.rolling(window = 0).mean()
sma_s = fake_data.rolling(window = 40).mean()

# plot
fig, axs = plt.subplots(2,1,figsize = (13,8))
axs[0].plot(forecast, label = "Forecast")
axs[0].axhline(y = 0, color = 'red', linestyle = "--")

axs[1].plot(fake_data, label = "Close")
axs[1].plot(sma_f, label = "sma_f")
axs[1].plot(sma_s, label = "sma_s")

axs[0].set_title('Forecast')
axs[1].set_title('Close')

plt.tight_layout()
plt.show()




#

fake_data = arbitrary_timeseries(generate_trendy_price(Nlength=500, Tlength=60, Xamplitude=10.0, Volscale=0.10)) # fake_data
sma_f = fake_data.rolling(window = 0).mean()
sma_s = fake_data.rolling(window = 40).mean()

fake_data.plot()
plt.plot(sma_f)
plt.plot(sma_s)



''' 5. Understand system '''
# how trend length relates to profitability of window_size
# - measure profit by sharp Ratio

cross = [forecast[i] for i in range(1,len(forecast)) if( forecast[i-1] < 0 and forecast[i] >= 0) or
         (forecast[i-1] >= 0 and forecast[i] < 0 )]


# Get idea of how fast different window size will trade
# 2. Turn over




# Prune any window side that are likely to be too expensive
# understand correlation structure to work out best window_size 

