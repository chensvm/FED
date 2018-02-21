import csv
import os
import datetime
from datetime import timedelta, date
import pandas as pd
import numpy as np

def getDirectory():
    with open ('ind_name.txt', 'w') as f:

        files = os.listdir('../../pyeikon/dataset/ind')

        for item in files:
            f.write(str(item)+'\n')

# with open (../pyeikon/dataset/ind)

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

def getFirstIndex():
    start_date = date(2010, 1, 1)
    end_date = date(2017, 12, 21)
    
    with open ('eikon_data.csv', 'w') as fout, open ('../../pyeikon/dataset/ind/.AXJO.csv') as fin:
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

        with open ('../../pyeikon/dataset/ind/'+str(item), 'r') as fin:
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

def upDown():
    with open ('../../pyeikon/dataset/ind/US10YT=RR.csv', 'r') as fin, open ('US10YT_eikon.csv', 'w') as fout:
        df = pd.read_csv('../../pyeikon/dataset/ind/US10YT=RR.csv')
        da = list(df['CLOSE'])
        value = ['']
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        print da[len(da)-1]
        for i in range(1, len(da)-1):
            if da[i] > da[i-1]:
                value.append(1)
            elif da[i] < da[i-1]:
                value.append(-1)
            else:
                value.append(0)

    df1 = pd.DataFrame()      
    df2 = pd.DataFrame({'y': value})
    df1['Date'] = df['Date']
    df1['y'] = df2
    df1.to_csv('US10YT_eikon.csv', index=False)           

if __name__ == "__main__":

    getFirstIndex()
    combineIndex()
    cleanData()
    # upDown()