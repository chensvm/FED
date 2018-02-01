# -*- coding: utf-8 -*-

from gensim.models import Doc2Vec
import gensim
import datetime
import numpy as np
import collections
import datetime
from datetime import timedelta, date
import csv
import os
import re
import pandas as pd
import pandas_datareader.data
from pandas import Series, DataFrame
from pandas_datareader._utils import RemoteDataError

start_date = date(2005, 1, 1)
end_date = date(2005, 1, 3)

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

# documents per day for training embedding model
documents = []
doc_label = []

for single_date in daterange(start_date, end_date):

    if os.path.isfile('../../../../tmp/finance/nytimes/business_news_and_title/' + str(single_date.strftime("%Y")) + '/' + str(single_date.strftime("%Y%m%d")) + ".npy"):

        with open('../../../../tmp/finance/nytimes/business_news_and_title/' + str(single_date.strftime("%Y")) + '/' + str(single_date.strftime("%Y%m%d")) + ".npy", 'r') as myfile:
            listOfToken = []
            document = []
            article_day = np.load(myfile)
            for news in article_day:
                news = str(news).replace('\\n', '')
                for ch in ['\\','"','_','{','}','[',']','(',')','>','#','+','-','.','!','$','\'', ',']:
                    if ch in news:
                        news = news.replace(ch,'')
                listOfToken = news.split(' ')
                document += listOfToken
        # documents.append(document)
        # doc_label.append(single_date.strftime("%Y%m%d"))
        documents.append(gensim.models.doc2vec.TaggedDocument(document, single_date.strftime("%Y%m%d")))

model = Doc2Vec(documents, size=150, min_count=3, workers=4, iter = 3)

with open ('company.csv', 'r') as fin, open ('DARNN_stock.csv', 'w') as fout:
        # fin.next()
        reader = csv.reader(fin)
        reader.next()
        writer = csv.writer(fout)
        for row in reader:
            year, mon, day = row[0].split('-')
            date = row[0].replace('-', '')
            if os.path.isfile('../../../../tmp/finance/nytimes/business_news_and_title/' + str(year) + '/' + str(date) + ".npy"):
                with open('../../../../tmp/finance/nytimes/business_news_and_title/' + str(year) + '/' + str(date) + ".npy", 'r') as myfile:
                    listOfToken = []
                    document = []
                    article_day = np.load(myfile)
                    for news in article_day:
                        news = str(news).replace('\\n', '')
                        for ch in ['\\','"','_','{','}','[',']','(',')','>','#','+','-','.','!','$','\'', ',']:
                            if ch in news:
                                news = news.replace(ch,'')
                        listOfToken = news.split(' ')
                        document += listOfToken
                    
                    infervec = model.infer_vector(document)
                    infervec = list(infervec)
                    content = infervec + row[1:]
                    writer.writerow(content)
            
            else:
                infervec = np.zeros((150,),dtype="float32")
                infervec = list(infervec)
                content = infervec + row[1:]
                writer.writerow(content)


            # row.append()
            


        # writer = csv.writer(fout)
        # for row in reader:
        #     writer.writerow(row[1:])
