#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 10:29:39 2023

-> 01. Develop trading system using: fake data, sma crossover

@author: nanthawat
"""

from generate_random_data import arbitrary_timeseries
from random_data import generate_trendy_price

''' 1. Data ''' 

# Real data
import yfinance as yf
import matplotlib.pyplot as plt
stock_data = yf.download("SPY", start = "2022-01-01", end = "2022-01-07").plot()
print(stock_data.head())



# Fake data
ans=arbitrary_timeseries(generate_trendy_price(Nlength=180, Tlength=30, Xamplitude=40.0, Volscale=0.15)).plot()


''' 2. indicator ''' 



''' 3. FIND THE PARAMETER FOR WINDOW_SIZE '''

