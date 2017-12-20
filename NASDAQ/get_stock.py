import datetime

import pandas as pd
import pandas_datareader.data
from pandas import Series, DataFrame

def getStock():

    ADBE = pandas_datareader.data.get_data_yahoo('ADBE', 
                                    start=datetime.datetime(2000, 1, 1), 
                                    end=datetime.datetime(2015, 1, 1))

    ADBE = ADBE[['Close']]


    AAPL = pandas_datareader.data.get_data_yahoo('AAPL', 
                                    start=datetime.datetime(2000, 1, 1), 
                                    end=datetime.datetime(2015, 1, 1))

    AAPL = AAPL[['Close']]


    STOCK = pd.concat([ADBE, AAPL], axis=1)

    with open('NASDAQ100.txt', 'r') as f:
        companies = f.readlines()
        companies = [x.strip() for x in companies] 
        for company in companies:
            print company
            price = pandas_datareader.data.get_data_yahoo(str(company), 
                                    start=datetime.datetime(2000, 1, 1), 
                                    end=datetime.datetime(2015, 1, 1))
            price = price[['Close']]
            STOCK = pd.concat([STOCK, price], axis=1)


    # print STOCK.head()

    STOCK.to_csv('NASDAQ100_company_20000101_20150101.csv')

def cleanData():

    

if __name__ == '__main__':
    cleanData()