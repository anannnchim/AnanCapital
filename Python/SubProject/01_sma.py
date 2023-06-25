#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 10:29:39 2023

-> 01. Develop trading system using: fake data, sma crossover
"""

''' ---------------------------------0. import library --------------------------------- ''' 

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
# from indicators import calc_ewmac_forecast_fake # calc_ewmac_forecast_fake: not working

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

# get system_performance
from system_performance import get_daily_return
from system_performance import get_sharp_ratio
from system_performance import compute_sharp_ratio
from system_performance import calculate_sharp_ratio
from system_performance import calculate_sharp_ratio_fdata

# clean data
sys.path.append("/Users/nanthawat/Documents/GitHub/AnanCapital/Python/Data/data_prep")
from clean_data import check_invalid_dataframe

 
''' -------------3. Generate plot for fake data: does it make sense ------------------------ ''' 


# fake_data
fake_data = arbitrary_timeseries(generate_trendy_price(Nlength= 1024, Tlength=50, Xamplitude=100, Volscale=0.0)) # fake_data

# add date_time
fake_data.index = pd.date_range(start = "2000-01-01", periods = len(fake_data))

# add indicator
sma_f = fake_data.rolling(window = 32).mean()
sma_s = fake_data.rolling(window = 128).mean()

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
turnover_per_year = turnover / 4

print(f'Holding period: {holding_period}')
print(f'Turnover: {turnover}')
print(f'Avg.holding period: {avg_holding_period}')
print(f'turnover per year: {turnover_per_year}')

''' 4. Check turnover for fake_data given a pair of moving average'''

simulate_sma_turnover_with_fdata(32,128,50)

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
stock_data = yf.download("DELTA.BK", start="2018-06-23", end = "2023-06-23")

# Prep data
sma_f = stock_data["Close"].rolling(window = 32).mean()
sma_s = stock_data["Close"].rolling(window = 128).mean()
forecast = calc_ewmac_forecast(stock_data["Close"],32,128)

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

# stocks
symbol_list = ["AOT.BK","PTT.BK","SCC.BK","KBANK.BK", "DELTA.BK"]

# ETFs
symbol_list = ["XLF","EWJ","ARKG","EMB", "TIP","VNQ", "GLD"]
# pairs = [(2, 8), (4, 16), (8, 32), (16, 64), (32, 128), (64, 256)]
pairs = [(64, 256), (85, 256), (128, 256)]
pairs = [ (32, 128), (64, 128), (50, 150)]
 
''' Thai stock 
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


'' ETFs: same 
Pair            Avg. holding period  Trades/year  turnover      holding
                                                              
(64, 256)            259.699206     0.566351  9.714286  2430.000000
(85, 256)            293.906502     0.502315  8.714286  2436.428571
(128, 256)           329.023535     0.468272  8.285714  2430.571429


* each stock has a consistent result. 

Summary:
    1. choice of {1/4, 1/3, 1/4} give the same expected holding and turnover
    2. pair (16,64) capture 3 month trend or  avg.holding at 53 <<< System S1  
    -  pair (32,128) capture 6 month trend or avg.holding at 125 
    -  pair (64,256) capture 1 year trend or avg.holding at 250
    3. fake data give similar properties as real data 
'''

''' -------------------------------------------------------------------------------- ''' 

''' 6. Compute Sharp Ratio '''

# stock data
start_date = datetime.datetime.now() - datetime.timedelta(days=20*365)
stock_data = yf.download("AOT.BK", start=start_date)
stock_data = stock_data["Close"]


''' Compute sharp for fake data '''  

# fake data
fake_data = arbitrary_timeseries(generate_trendy_price(Nlength= 1000, Tlength=125, Xamplitude=100, Volscale=0.0)) # fake_data
stock_data = fake_data.to_frame()
stock_data.columns = ['Close'] 

# indicators
stock_data
window_fast = 32
window_slow = 128
sma_f = stock_data.rolling(window = window_fast).mean()
sma_s = stock_data.rolling(window = window_slow).mean()

# forecast  <<< adjust
forecast = sma_f - sma_s

# holding  <<< adjust
hold = (forecast >= 0).astype(int)

# daily return
daily_return = get_daily_return(stock_data) 
cumulative_daily_return = (1 + daily_return).cumprod() - 1

# system performance
system_return = daily_return * hold.shift(1)

# cum_return 
system_cum_return = (1 + system_return).cumprod() - 1

# Convert Series to DataFrame
# df = stock_data.to_frame()
df = pd.DataFrame()
df = df.assign(sma_f = sma_f,
               sma_s = sma_s,
               forecast = forecast,
               hold = hold,
               daily_return = daily_return,
               system_return = system_return,
               cumulative_daily_return = cumulative_daily_return,
               system_cum_return = system_cum_return) 

# rename heading
df.rename(columns = {df.columns[0]: 'Close'}, inplace = True)

# fill all NaN with first element
df = df.fillna(method='bfill').fillna(method='ffill')

# calculate sharp Ratio
original_sharp = compute_sharp_ratio(daily_return)
new_sharp = compute_sharp_ratio(system_return)
print(f"original: {original_sharp}")
print(f"new: {new_sharp}")

# plot
fig, axs = plt.subplots(5,1,figsize = (8,6))

axs[0].set_title('Close')
axs[0].plot( df["Close"], label = "Close")
axs[0].plot(df["sma_f"], label = "sma_f")
axs[0].plot(df["sma_s"], label = "sma_s")


axs[1].set_title('Forecast')
axs[1].plot(df["forecast"], label = "Forecast")
axs[1].axhline(y = 0, color = 'red', linestyle = "--")

axs[2].set_title('Hold')
axs[2].plot(df["hold"], label = "sma_s")

axs[3].set_title('cumulative_daily_return')
axs[3].plot(df["cumulative_daily_return"] , label = "cumulative_daily_return")

axs[4].set_title('system_cum_return')
axs[4].plot(df['system_cum_return'] , label = "system_cum_return")

plt.tight_layout()
plt.show()


# compute sharp Ratio using function
window_fast = 32
window_slow = 128
Tlength = 125
Volscale = 0.15

calculate_sharp_ratio_fdata(Tlength,Volscale, window_fast,window_slow)

pairs = [(2, 8), (4, 16), (8, 32), (16, 64), (32, 128), (64, 256)]
trend_length = [5, 21, 64, 128, 256]

sharp_ratios_by_pair = {str(pair): [calculate_sharp_ratio_fdata(TLength,  0.15, pair[0], pair[1]) for TLength in trend_length] for pair in pairs}

# simulate sharp of fake data
pairs = [(2, 8), (4, 16), (8, 32), (16, 64), (32, 128), (64, 256)]

pairs = [ (32, 128)]
trend_length = [5, 21, 64, 128, 256]

Volscale = 0


sharp_ratios_by_pair = {}
# Using dict and list comprehensions
sharp_ratios_by_pair = {str(pair): [calculate_sharp_ratio_fdata(TLength,  0, pair[0], pair[1]) for TLength in trend_length] for pair in pairs}
df = pd.DataFrame(sharp_ratios_by_pair)

# Transpose the DataFrame if you want each row to correspond to a pair
df = df.T

# Reset the index if you want the pairs to be a column instead of an index
df.reset_index(inplace=True)

# mean value 
mean_values = {key: np.round(np.mean(val), 2) for key, val in sharp_ratios_by_pair.items()}
mean_values

# Create the bar plot
plt.bar(mean_values.keys(), mean_values.values())
plt.xlabel('Pairs')
plt.ylabel('Mean Value')
plt.title('Mean Value for each Pair')
plt.xticks(rotation=90) # This rotates the x-axis labels so they don't overlap
plt.show()

''' Simulate sharp ratio with real data. '''  
# Compute Sharp Ratio for single real data
calculate_sharp_ratio("XLF",i,128)


# simulate variation of pair
main = 128
pair = [26, 38, 51, 64, 77, 90, 102, 115]
for i in pair:
    print( calculate_sharp_ratio("DELTA.BK", i, main))
    

# Simulate the result
pairs = [(2, 8), (4, 16), (8, 32), (16, 64), (32, 128), (64, 256)]
pairs = [ (32, 128), (64, 128), (50, 150)]

# variation of (64,256)
pairs = [(64, 256), (85, 256), (128, 256)]

pairs = [(50, 200), (100, 200), (150, 200)]

# RUN
symbol_list = ["AOT.BK","PTT.BK","SCC.BK","KBANK.BK", "DELTA.BK",
               "ADVANC.BK", "BANPU.BK", "BBL.BK", "BEM.BK", "BH.BK",
               "CENTEL.BK", "EA.BK", "GPSC.BK", "TRUE.BK", "TTB.BK"] # SET

symbol_list = ["AOT.BK"]
 
# mean value 
mean_values = {key: np.round(np.mean(val), 2) for key, val in sharp_ratios_by_pair.items()}
mean_values

median_values = {key: np.round(np.median(val), 2) for key, val in sharp_ratios_by_pair.items()}
median_values

# ETFs Universe (35)
''' *Apply sma algo
* Orignial sharp = 0.27

{'(2, 8)': 0.09,
 '(4, 16)': 0.20,
 '(8, 32)': 0.27,
 '(16, 64)': 0.26,
 '(32, 128)': 0.30,
 '(64, 256)': 0.32}
'''
# variation of (64,256): similar 
'''
{'(64, 256)': 0.32,
 '(85, 256)': 0.31,
 '(128, 256)': 0.32}
'''

# variation of (150,200
'''
{'(50, 200)': 0.31, 
'(100, 200)': 0.33, 
'(150, 200)': 0.32})
'''

# Thai stocks high cap. 
'''
{'(2, 8)': 0.45
 '(4, 16)': 0.50
 '(8, 32)': 0.49
 '(16, 64)': 0.49
 '(32, 128)': 0.50
 '(64, 256)': 0.47
'''

# ETFs Universe (35) 
''' Annual return
{'(2, 8)': 0.03,
 '(4, 16)': 0.04,
 '(8, 32)': 0.05,
 '(16, 64)': 0.05,
 '(32, 128)': 0.05,
 '(64, 256)': 0.06}

 Annual Risk 
{'(2, 8)': 0.15,
 '(4, 16)': 0.14,
 '(8, 32)': 0.14,
 '(16, 64)': 0.14,
 '(32, 128)': 0.14,
 '(64, 256)': 0.15}
'''

# 1. apply cost limit 
'''
# Sharp Ratio for real data:  
    * Universe : 35 ETFs
    * with Annual risk = 14% and return = 5%
    * turnover/year 
    * transaction cost = 0.01% (according to US ETFs from Rob excel)
    * holding cost = 1.05% (management fee by ETF and by SA)
    
            Sharp R.   | Turnover | 1/3 Sharp.(limit)  || 
    {'(2, 8)': 0.09,   | 20 | 0.03  || (20*0.01%) + (1.05%) = 0.09*** exceed 
     '(4, 16)': 0.20,  | 10 |  0.07 || 0.08*** exceed
     '(8, 32)': 0.27,  |  5 | 0.09  || 0.08 [Pass]
     '(16, 64)': 0.26, |  3 | 0.09  || 0.08 [Pass]
     '(32, 128)': 0.30, | 1 | 0.1   || 0.08 [Pass]
     '(64, 256)': 0.32} | 1 | 0.11  || 0.08 [Pass]
    
    *assume no exchange fee, custodian fee and no inside trading fund fee.
    
'''


''' -------------------------------------------------------------------------------- ''' 

"""
  Summary:
      
      Turnover:
          
          # Result: Real data
          Pair           Avg. holding period  Trades/year  turnover  holding
          (2, 8)                6.401865    20.431629     389.6   2493.0                                                          
          (4, 16)              13.321197    10.069698     192.0   2557.2
          (16, 64)             53.937053     2.569563      49.0   2634.8
          (32, 128)           125.654967     1.111587      21.2   2627.6
          (64, 256)           250.340280     0.588288      11.2   2786.6
          
          
          1. it is consistant among real {ETFs and SET} and fake data
          2. choice of pair variation of {1/4, 1/3, 1/4} give the same expected holding and turnover
          3. pair (16,64) capture 3 month trend or  avg.holding at 53 <<< System S1  
          4. pair (32,128) capture 6 month trend or avg.holding at 125 
          5. pair (64,256) capture 1 year trend or avg.holding at 250 
          
          
          
     Sharp Ratio:
         
         * Universe : 35 ETFs
         * ETFs: Orignial sharp = 0.27
         * with Annual risk = 14% and return = 5%
         
         # Pre-cost sharp.
         {'(2, 8)': 0.09,
          '(4, 16)': 0.20,
          '(8, 32)': 0.27,
          '(16, 64)': 0.26,
          '(32, 128)': 0.30,
          '(64, 256)': 0.32}
         '''
         
         * turnover/year  
         * transaction cost = 0.01% (according to US ETFs from Rob excel)
         * holding cost = 1.05% (management fee by ETF and by SA)
         * assume no exchange fee, custodian fee and no inside trading fund fee.
         
         
         # Post-cost sharp.
          Pair | Sharp R.   | Turnover | 1/3 Sharp.(limit)  ||  ->>  sharp.after cost
          
         {'(2, 8)': 0.09,   | 20 | 0.03  || (20*0.01%) + (1.05%) = 0.09 [Exceed]  ->> 0
          '(4, 16)': 0.20,  | 10 |  0.07 || 0.08 [Exceed] ->> 0.08
          
          '(8, 32)': 0.27,  |  5 | 0.09  || 0.08 [Pass] ->> 0.19
          '(16, 64)': 0.26, |  3 | 0.09  || 0.08 [Pass] ->> 0.18
          '(32, 128)': 0.30, | 1 | 0.1   || 0.08 [Pass] ->> 0.22
          '(64, 256)': 0.32} | 1 | 0.11  || 0.08 [Pass] ->> 0.24
        
         
        
         
         1. all sharp.after cost < original sharp. 
         2. choice of pair variation of {1/4, 1/3, 1/4} give the same sharp ratio
         3. Pair (2,8) and (4,16) are too expensive to trade.
         4. from pair (8,32) to (64,256) give similar cost in term of sharp. - 0.08
         
         
         Example sma: (64,256)
         
             - turnover: 1 / year
             - sharp.before cost = 0.32
             - sharp.after cost = 0.24
             - cost in sharp = 0.08
             
             Expected return = 3.36%
             Expected risk = 14% 
  
  
     Further study:
         - fit allocation using real data (how sharp ratio affect position sizing)
         - try different fake data 
      
"""




''' 7. Compute forecast vs binary Sharp.ratio '''

''' page 284
    1. raw forecast 
    2. risk adjusted raw forecast (standardized forecast) (volatility adjusted)
    3. scaled forecast = raw forecast x forcast scalar 
    4. capped forecast
    * forcast scalar = 10 / natual average value 
    * ETFs: forecast scalar of EWMA(32,128) = 3.30
'''


# Download data from 4 years ago to today
start_date = datetime.datetime.now() - datetime.timedelta(days=20*365)

# manual
stock_list = ["AOT.BK","PTT.BK","JMART.BK"]
stock_list = ["SKY.BK"]
stock_list = ["AOT.BK","PTT.BK","SCC.BK","KBANK.BK", "DELTA.BK",
               "ADVANC.BK", "BANPU.BK", "BBL.BK", "BEM.BK", "BH.BK",
               "CENTEL.BK", "EA.BK", "GPSC.BK", "TTB.BK"] # SET

# ETFs 
file_path = '/Users/nanthawat/Desktop/universe.csv'
df = pd.read_csv(file_path)
stock_list = df["STOCK"].tolist()
stock_list = symbol_list[1:]



# Initialize two empty dictionaries to store the forecasts and raw data
forecasts = {}
raw_data = {}  # Rename this variable to avoid confusion with the 'raw' inside the loop

natural_f = -1
sum_natural_f = 0
f_scalar = -1 

for stock in stock_list:
    
    stock_data = yf.download(stock, start=start_date)

    # Prep data
    sma_f = stock_data["Close"].rolling(window = 32).mean()
    sma_s = stock_data["Close"].rolling(window = 128).mean()

    # Calculate forecast and raw data
    forecast = calc_ewmac_forecast(stock_data["Close"],32,128)
    raw = sma_f - sma_s
    

    # Save forecast and raw data in the dictionaries
    forecasts[stock + "_forecast"] = forecast
    raw_data[stock + "_raw"] = raw  # Save raw data into the 'raw_data' dictionary
    
    sum_natural_f = sum_natural_f + forecast.abs().mean()
    natural_f = sum_natural_f / len(stock_list)
    f_scalar = 10 / natural_f
    


# check raw ( how many time it cross vs forecast)
# plot to see the different or pattern
# continue in google sheet


# 1) simulate raw forecast across stocks
plt.figure(figsize=(14,7))
for stock in stock_list:
    raw_data[stock + "_raw"].plot(label = stock)
plt.axhline(0, color = 'red')
plt.title("Raw data given different instrument")
plt.xlabel("Date")
plt.ylabel("raw data")
plt.legend()
plt.show()

zero_crossings = {}
for stock in stock_list:
    # Subtract shifted series from original series and check sign (True if positive)
    sign_changes = np.sign(raw_data[stock + "_raw"]).diff().abs() > 0 

    # Count sign changes, indicating a zero crossing
    zero_crossings[stock] = sign_changes.sum()

print(zero_crossings)
sum(zero_crossings.values()) / len(zero_crossings.values())

plt.bar(zero_crossings.keys(), zero_crossings.values())
plt.xticks(rotation=90) # to make the x-axis labels vertical for better visibility
plt.show()



# 2) simulate Vol.adj crossover across stocks
plt.figure(figsize=(14,7))
for stock in stock_list:
    forecasts[stock + "_forecast"].plot(label = stock)
plt.axhline(0, color = 'red')
plt.title("Vol.adj crossover given different instrument")
plt.xlabel("Date")
plt.ylabel("Vol.adj crossover")
plt.legend()
plt.show()


zero_crossings_n = {}
for stock in stock_list:
    # Subtract shifted series from original series and check sign (True if positive)
    sign_changes = np.sign(forecasts[stock + "_forecast"]).diff().abs() > 0 

    # Count sign changes, indicating a zero crossing
    zero_crossings_n[stock] = sign_changes.sum()

print(zero_crossings_n)
sum(zero_crossings_n.values()) / len(zero_crossings_n.values())


plt.bar(zero_crossings_n.keys(), zero_crossings_n.values())
plt.xticks(rotation=90) # to make the x-axis labels vertical for better visibility
plt.show()



# 3) simulate finalized forecast across stocks
plt.figure(figsize=(14,7))
for stock in stock_list:
    (forecasts[stock + "_forecast"]*f_scalar).plot(label = stock)
plt.axhline(0, color = 'red')
plt.title("finialized forecast crossover given different instrument")
plt.xlabel("Date")
plt.ylabel("fianalized f")
plt.legend()
plt.show()



'''
? how to add or reduce position according to forecast ?
= 25% different in weight
? how to determine whether it it help vs binary ?
= sharp r. 
'''





