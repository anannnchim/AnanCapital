"""
Created on Sat May  6 22:12:20 2023

-> 02 Create a random price data from Rob Carver
: Generate large amounts of random data with the desired characteristics you want, and then analyse that. 
- ref: https://qoppac.blogspot.com/2015/11/using-random-data.html
- Func for random data: https://github.com/robcarver17/systematictradingexamples/blob/master/commonrandom.py#L56
"""




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define function to generate random price data
def arbitrary_timeseries(prices):
    return pd.Series(prices)

def generate_trendy_price(Nlength, Tlength, Xamplitude, Volscale):
    """ Generate a random price series with trends """
    noise = np.random.normal(scale=Volscale, size=Nlength+Tlength)
    prices = [Xamplitude]
    for i in range(1, Nlength+Tlength):
        if i % Tlength == 0:
            prices.append(prices[-1] + Xamplitude)
        else:
            prices.append(prices[-1] + noise[i])
    return prices[Tlength:]

# Generate random price data with trends
price_data = generate_trendy_price(Nlength=180, Tlength=30, Xamplitude=10.0, Volscale=0.2)

# Plot the price data
plt.plot(price_data)
plt.title("Random Price Data")
plt.xlabel("Time")
plt.ylabel("Price")
plt.show()




def generate_trendy_price(Nlength, Tlength , Xamplitude, Volscale, sines=False):
    """
    Generates a trend of length N amplitude X, plus gaussian noise mean zero std. dev (vol scale * amplitude)
    
    If sines=True then generates as a sine wave, otherwise straight line
    
    returns a vector of numbers
    """
    
    stdev=Volscale*Xamplitude
    noise=generate_noise(Nlength, stdev)

    ## Can use a different process here if desired
    if sines:
        process=generate_siney_trends(Nlength, Tlength , Xamplitude) 
    else:
        process=generate_trends(Nlength, Tlength , Xamplitude)    
    
    combined_price=[noise_item+process_item for (noise_item, process_item) in zip(noise, process)]
    
    return combined_price


import numpy as np
import pandas as pd
import scipy.signal as sg
from datetime import datetime, timedelta
from random import gauss
from typing import List

def generate_siney_trends(Nlength: int, Tlength: int , Xamplitude: float) -> List[float]:
    halfAmplitude = Xamplitude/2.0
    cycles = Nlength/Tlength
    cycles_as_pi = cycles*np.pi
    increment = cycles_as_pi/Nlength

    alltrends = [np.sin(x)*halfAmplitude for x in np.arange(0.0, cycles_as_pi, increment)]
    alltrends = alltrends[:Nlength]

    return alltrends

def generate_trends(Nlength: int, Tlength: int , Xamplitude: float) -> List[float]:
    halfAmplitude = Xamplitude/2.0
    trend_step = Xamplitude/Tlength
    cycles = int(np.ceil(Nlength/Tlength))

    trendup = list(np.arange(start=-halfAmplitude, stop=halfAmplitude, step=trend_step))
    trenddown = list(np.arange(start=halfAmplitude, stop=-halfAmplitude, step=-trend_step))
    alltrends = [trendup + trenddown] * int(np.ceil(cycles))
    alltrends = sum(alltrends, [])
    alltrends = alltrends[:Nlength]

    return alltrends

def generate_trendy_price(Nlength: int, Tlength: int, Xamplitude: float, Volscale: float, sines: bool = False) -> List[float]:
    stdev = Volscale*Xamplitude
    noise = generate_noise(Nlength, stdev)

    if sines:
        process = generate_siney_trends(Nlength, Tlength, Xamplitude) 
    else:
        process = generate_trends(Nlength, Tlength, Xamplitude)    

    combined_price = [noise_item + process_item for (noise_item, process_item) in zip(noise, process)]

    return combined_price

def generate_noise(Nlength: int, stdev: float) -> List[float]:
    return [gauss(0.0, stdev) for Unused in range(Nlength)]

def threeassetportfolio(plength: int = 5000, SRlist: List[float] = [1.0, 1.0, 1.0], annual_vol: float = .15, clist: List[float] = [.0,.0,.0], index_start: datetime = datetime(2000,1,1)) -> pd.DataFrame:
    (c1, c2, c3) = clist
    dindex = pd.date_range(start=index_start, periods=plength, freq='D')
    daily_vol = annual_vol/16.0
    means = [x*annual_vol/250.0 for x in SRlist]
    stds = np.diagflat([daily_vol]*3)
    corr = np.array([[1.0, c1, c2], [c1, 1.0, c3], [c2, c3, 1.0]])

    covs = np.dot(stds, np.dot(corr, stds))
    m = np.random.multivariate_normal(means, covs, plength).T
    portreturns = pd.DataFrame(dict(one=m[0], two=m[1], three=m[2]), dindex)
    portreturns = portreturns[['one', 'two', 'three']]
    return portreturns

ans=arbitrary_timeseries(generate_trendy_price(Nlength=30, Tlength=15, Xamplitude=40.0, Volscale=0.0)).plot()



