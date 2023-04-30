#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 10:29:39 2023

-> Create a Regression model for predict stock price

@author: nanthawat
"""


''' 1. IMPORT DATA '''

'''
Input  
    1. start_date  
    2. end_date 3. symbol 
Output  
    1. stock_data 

Function
    1. yf.download
'''

import yfinance as yf
import matplotlib.pyplot as plt
stock_data = yf.download("SPY", start = "2022-01-01", end = "2022-01-07")
print(stock_data.head())

''' 2. CREATE A ROLLING REGRESSION MODEL ''' 

'''
Input  
    1. start_date  
    2. end_date 3. symbol 
Output  
    1. stock_data 

Function
    1. yf.download
'''

''' 3. FIND THE PARAMETER FOR WINDOW_SIZE '''

