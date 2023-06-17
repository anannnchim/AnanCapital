"""

-> 02 Create a random price data from Rob Carver

: Generate large amounts of random data with the desired characteristics you want, and then analyse that. 
- ref: https://qoppac.blogspot.com/2015/11/using-random-data.html
- Func for random data: https://github.com/robcarver17/systematictradingexamples/blob/master/commonrandom.py#L56

"""

import pandas as pd
from typing import List
from random import gauss
import numpy as np

# Define function to generate random price data (convert to pd.series)

def arbitrary_timeseries(prices):
    # modify: add time_date
    #pd.Series(prices).index = pd.date_range(start = '2020-01-01', periods= len(pd.Series(prices)))
    return pd.Series(prices)

def generate_noise(Nlength: int, stdev: float) -> List[float]:
    return [gauss(0.0, stdev) for Unused in range(Nlength)]


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
    
    ## adjust: shift all up
    #combined_price = [value - min(combined_price) for value in combined_price]
    combined_price = [value + Xamplitude for value in combined_price]


    return combined_price



