import datetime

import pandas as pd
import pandas_datareader.data
from pandas import Series, DataFrame

STOCK = pandas_datareader.data.get_data_yahoo('^NDX', 
                                 start=datetime.datetime(2000, 1, 1), 
                                 end=datetime.datetime(2015, 1, 1))

print list(STOCK)

STOCK = STOCK[['Close']]

print STOCK.head()

STOCK.to_csv('NASDAQ100_20000101_20150101.csv')