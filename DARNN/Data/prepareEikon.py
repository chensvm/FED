import csv
import os
import datetime
from datetime import timedelta, date
import pandas as pd
import numpy as np

def getDirectory():
    with open ('ind_name.txt', 'w') as f:

        files = os.listdir('../pyeikon/dataset/ind')

        for item in files:
            f.write(str(item)+'\n')

# with open (../pyeikon/dataset/ind)

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

def getFirstIndex():
    start_date = date(2017, 8, 19)
    end_date = date(2017, 12, 21)
    
    with open ('eikon_data.csv', 'w') as fout, open ('../pyeikon/dataset/ind/.AXJO.csv') as fin:
        writer = csv.writer(fout)
        reader = csv.reader(fin)
        reader.next()
        writer.writerow(['Date', '.AXJO'])
        for row in reader:
            year, mon, day = row[0].split('-')
            row_date = date(int(year), int(mon), int(day))

            if row_date in daterange(start_date, end_date):

                li = [row[0], row[1]]
                writer.writerow(li)
 

def combineIndex():

    with open ('ind_name.txt', 'r') as f:
        inds = f.read().split('\n')

    for item in inds:

        with open ('../pyeikon/dataset/ind/'+str(item), 'r') as fin:
            reader = csv.reader(fin)
            reader.next()
            value = []
            df = pd.read_csv('eikon_data.csv')
            da = df['Date']
            
            for row in reader:
                for row_date in da:  
                    if row_date == row[0]:
                        value.append(row[2])
            
            

            ind_name = str(item).replace('.csv', '')
            df2 = pd.DataFrame({ind_name: value})
            print df2
            df[ind_name] = df2
            df.to_csv('eikon_data.csv', index=False)

def cleanData():
    with open ('eikon_data.csv', 'r') as fin, open ('DARNN_eikon.csv', 'w') as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        for row in reader:
            writer.writerow(row[1:])             

if __name__ == "__main__":

    # getFirstIndex()
    # combineIndex()
    cleanData()