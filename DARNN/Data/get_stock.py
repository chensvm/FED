import datetime
import csv
import pandas as pd
import pandas_datareader.data
from pandas import Series, DataFrame
from pandas_datareader._utils import RemoteDataError

def getStock():
    try:

        ADBE = pandas_datareader.data.get_data_yahoo('ADBE', 
                                        start=datetime.datetime(2017, 8, 20), 
                                        end=datetime.datetime(2017, 12, 20))

        ADBE = ADBE[['Close']]
        
    except RemoteDataError:
        ADBE = pandas_datareader.data.get_data_yahoo('ADBE', 
                                        start=datetime.datetime(2017, 8, 20), 
                                        end=datetime.datetime(2017, 12, 20))

        ADBE = ADBE[['Close']]

    try:

        AAPL = pandas_datareader.data.get_data_yahoo('AAPL', 
                                        start=datetime.datetime(2017, 8, 20), 
                                        end=datetime.datetime(2017, 12, 20))

        AAPL = AAPL[['Close']]

    except RemoteDataError:

        AAPL = pandas_datareader.data.get_data_yahoo('AAPL', 
                                        start=datetime.datetime(2017, 8, 20), 
                                        end=datetime.datetime(2017, 12, 20))

        AAPL = AAPL[['Close']]      


    STOCK = pd.concat([ADBE, AAPL], axis=1)

    with open('NASDAQ100.txt', 'r') as f:
        companies = f.readlines()
        companies = [x.strip() for x in companies] 
        for company in companies:
            print company
            try:
                price = pandas_datareader.data.get_data_yahoo(str(company), 
                                        start=datetime.datetime(2017, 8, 20), 
                                        end=datetime.datetime(2017, 12, 20))
            except RemoteDataError:
                price = pandas_datareader.data.get_data_yahoo(str(company), 
                                        start=datetime.datetime(2017, 8, 20), 
                                        end=datetime.datetime(2017, 12, 20))
            price = price[['Close']]
            STOCK = pd.concat([STOCK, price], axis=1)


    # print STOCK.head()

    STOCK.to_csv('company.csv')

def cleanData():
    with open ('company.csv', 'r') as fin, open ('DARNN_stock.csv', 'w') as fout:
        # fin.next()
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        for row in reader:
            writer.writerow(row[1:])

def deleteEmbedding():
    with open ('DARNN_stock.csv', 'r') as fin, open ('DARNN_stock_num.csv', 'w') as fout:
        # fin.next()
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        for row in reader:
            writer.writerow(row[150:])

    

if __name__ == '__main__':
    # getStock()
    # cleanData()
    deleteEmbedding()