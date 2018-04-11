# -*- coding: utf-8 -*-
import matplotlib 
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use('Agg')
from gensim.models import Doc2Vec
import gensim
import numpy as np
import collections
import datetime
from datetime import timedelta, date
import csv
import os
import re
import json
import pandas as pd
import pandas_datareader.data
from pandas import Series, DataFrame
from pandas_datareader._utils import RemoteDataError
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

start_date = date(2017, 1, 1)
end_date = date(2017, 12, 31)

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

def previewNews():
    titles = []

    for single_date in daterange(start_date, end_date):

        if os.path.isfile('../../../../tmp2/finance_threedays/us10yy/' + str(single_date.strftime("%Y")) + '/' + str(single_date.strftime("%Y%m%d")) + ".npy"):

            with open('../../../../tmp2/finance_threedays/us10yy/' + str(single_date.strftime("%Y")) + '/' + str(single_date.strftime("%Y%m%d")) + ".npy", 'r') as myfile:
                
                article_day = np.load(myfile)
            
                for news in article_day:

                    title = news[0].encode('utf-8')
                    article = news[1].encode('utf-8')
                
                    for ch in ['\\','"','_','{','}','[',']','(',')','>','#','+','-','.','!','$','\'', ',']:
                        if ch in title:
                            title = title.replace(ch,'')

                    titles.append(title)

    df = pd.DataFrame(titles, columns=["title"])
    df.to_csv('titles.csv', index=False) 

def concatenatingThreeDayNews():

    for single_date in daterange(start_date, end_date):
        three_day_title = []
        for day in daterange(single_date - timedelta(3) , single_date ):
            
            if os.path.isfile('../../../../tmp2/finance_threedays/us10yy/' + str(single_date.strftime("%Y")) + '/' + str(single_date.strftime("%Y%m%d")) + ".npy"):
                if os.path.isfile('../../../../tmp2/finance_threedays/us10yy/' + str(day.strftime("%Y")) + '/' + str(day.strftime("%Y%m%d")) + ".npy"):
                    with open('../../../../tmp2/finance_threedays/us10yy/' + str(day.strftime("%Y")) + '/' + str(day.strftime("%Y%m%d")) + ".npy") as fin:
                        article_day = np.load(fin)

                        for news in article_day:

                            title = news[0].encode('utf-8')
                            article = news[1].encode('utf-8')
                            full_text = title + article
                            for ch in ['\\','"','_','{','}','[',']','(',')','>','#','+','-','.','!','$','\'', ',']:
                                if ch in title:
                                    title = title.replace(ch,'')
                                    # full_text = full_text.replace(ch,'')

                            # three_day_title.append(full_text)
                            three_day_title.append(title)
        
        with open ("../DARNN/Data/US10YT_eikon.csv", 'r') as fread:
            eikonreader = csv.reader(fread)
            eikonreader.next()
            for row in eikonreader:
                
                if str(single_date.strftime("%Y")) + "-" + str(single_date.strftime("%m")) + "-" + str(single_date.strftime("%d")) == str(row[0]):
                    
                    if str(row[1]) == "-1.0":
                        
                        df = pd.DataFrame(three_day_title, columns = None)
                        df.to_csv('../../../../tmp2/finance_threedays/negative_key/'  + str(single_date.strftime("%Y%m%d")) +".csv", index = False )
                        print "-1:  " + str(single_date.strftime("%Y%m%d"))
                    
                    elif str(row[1]) == "1.0":
                        df = pd.DataFrame(three_day_title, columns = None)
                        df.to_csv('../../../../tmp2/finance_threedays/positive_key/'  + str(single_date.strftime("%Y%m%d")) +".csv", index = False )
                        print "+1:  " + str(single_date.strftime("%Y%m%d"))
                    
                    else:
                        pass

def findSimilarity():
    documents = []
    ind = 0
    lables = []
   
    for filename in os.listdir('../../../../tmp2/finance_threedays/negative_key/'):
        try:

            listOfToken = []

            df = pd.read_csv('../../../../tmp2/finance_threedays/negative_key/' + str(filename))
            docreader = pd.DataFrame(df, columns = None, index = None)
      
            for index, row in docreader.iterrows():
        
                news = str(row['0'])
                for ch in ['\\','"','_','{','}','[',']','(',')','>','#','+','-','.','!','$','\'', ',', '\\n']:
                    if ch in news:
                        news = news.replace(ch,'')
                listOfToken += news.split(' ')

            tag = [ind] 
            documents.append(gensim.models.doc2vec.TaggedDocument(listOfToken, tag))
            lables = lables + [0]
            ind = ind + 1
        except:
            continue


    print "the first index of positive:"
    print ind
    print "====== positive ======"
       

    for filename in os.listdir('../../../../tmp2/finance_threedays/positive_key/'):
        try:

            listOfToken = []

            df = pd.read_csv('../../../../tmp2/finance_threedays/positive_key/' + str(filename))
            docreader = pd.DataFrame(df, columns = None, index = None)
      
            for index, row in docreader.iterrows():
        
                news = str(row['0'])
                for ch in ['\\','"','_','{','}','[',']','(',')','>','#','+','-','.','!','$','\'', ',', '\\n']:
                    if ch in news:
                        news = news.replace(ch,'')
                listOfToken += news.split(' ')
            
            tag = [ind]
            lables = lables + [1]
         
            documents.append(gensim.models.doc2vec.TaggedDocument(listOfToken, tag))
          
            ind = ind + 1
        except:
            continue

    print "num of docs"


    d2v_model = Doc2Vec(documents, size=200, min_count=0) 
    d2v_model.train(documents, total_examples=len(documents), epochs=50)
    print len(d2v_model.docvecs)
  

    # kmeans_model = KMeans(n_clusters=2, init='k-means++', max_iter=100)  
    # X = kmeans_model.fit(d2v_model.docvecs.doctag_syn0)
    # labels=kmeans_model.labels_.tolist()
    # l = kmeans_model.fit_predict(d2v_model.docvecs.doctag_syn0)

    pca = PCA(n_components = 2).fit([vec for vec in d2v_model.docvecs])
    datapoint = pca.transform([vec for vec in d2v_model.docvecs])

    
    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
   
    label1 = ["#FA8072", "#0066cc"]
    # red for negative, blue for positive
    color = [label1[i] for i in lables]
    plt.scatter(datapoint[:, 0], datapoint[:, 1], c=color)  
    # ax.scatter(datapoint[:, 0], datapoint[:, 1],datapoint[:, 2], c=color)
    plt.savefig("cluster.png")


if __name__ == "__main__":
    concatenatingThreeDayNews()
    findSimilarity()
    
