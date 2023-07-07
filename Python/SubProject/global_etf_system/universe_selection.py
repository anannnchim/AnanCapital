#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 14:20:31 2023

ETFs Trading system: Universe Selection

"""

import pandas as pd
import sys
import datetime
import yfinance as yf
import numpy as np

sys.path.append('/Users/nanthawat/Desktop/Python/pysystemtrade')
sys.path.append("/Users/nanthawat/Documents/GitHub/AnanCapital/Python/PreEvaluation")
sys.path.append("/Users/nanthawat/Documents/GitHub/AnanCapital/Python/Indicator")
sys.path.insert(0, '/Users/nanthawat/Documents/GitHub/AnanCapital/Python/Indicator')

from system_performance import get_annual_risk

 
''' -------- 1) Get all tradable assets ------------------------ ''' 

# 1. Import symbol from googlesheet
df = pd.read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vQC6-xU1dmxued0x9A711j1_VmbFhaV7IsgXncJ5FB7AvzlwDK9FzLk6RBvJ9OPhe-aQcWzJYgU1SX_/pub?output=csv")

# modify data
df["Ticker"]   = df["Ticker"].str.replace(" US",'').replace(" LN",'')
df["Expense Ratio"] = df["Expense Ratio"].str.replace("%",'').astype(float)/100

# RUN
print(f"Average Expense Ratio: {round(np.mean(df['Expense Ratio'])*100,2)}%")

# 2. Calculate risk of each pair and cost
annual_risk = pd.DataFrame()
for ticker in df["Ticker"]:
    start_date = datetime.datetime.now() - datetime.timedelta(days=1*365)
    stock_data = yf.download(ticker, start=start_date)
    stock_data = stock_data["Close"]
    annual_risk[ticker]= get_annual_risk(stock_data)
    
    
# 3. Calculate risk of each instrument
annual_risk = pd.DataFrame()  # Initialize an empty dataframe
for ticker in df["Ticker"]:
    start_date = datetime.datetime.now() - datetime.timedelta(days=20*365)
    stock_data = yf.download(ticker, start=start_date)
    stock_data = stock_data["Close"]
    annual_risk.loc[ticker, 'Annual Risk'] = get_annual_risk(stock_data)  # Store the result in a new column 'AnnualRisk'

# RUN
print(f"Average {round(annual_risk.drop(annual_risk.index[2]).mean()*100,2)}%")

# export to csv
annual_risk.to_csv("/Users/nanthawat/Desktop/annual_risk.csv")

