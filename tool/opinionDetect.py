#-*- coding: utf-8 -*-
import numpy as np
import re
import csv
import math
from nltk.corpus import stopwords
from datetime import timedelta, date
import pandas as pd
import nltk
import os

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)


def detectOpinion():

    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    with open('../fed_rates/fed_date_rate_training.csv', 'r') as c1:
        reader = csv.reader(c1)
        next(reader, None)  # skip the headers
        #for training
        prevRow = "2004-11-01"

        #for testing
        # prevRow = "1998-01-01"

        for row in reader:

            cur_year, cur_month, cur_day = str(row[0]).split("-")
            pre_year, pre_month, pre_day = prevRow.split("-")

            start_date = date(int(pre_year), int(pre_month), int(pre_day))
            end_date = date(int(cur_year), int(cur_month), int(cur_day))

            for single_date in daterange(start_date, end_date):


                with open('../../../../tmp2/finance_data/filtered_articles/nytimes/' +str(single_date.strftime("%Y"))+"/"+ str(single_date.strftime("%Y%m%d")) + ".npy", 'r') as myfile:

                    print str(single_date.strftime("%Y-%m-%d"))
                    data = np.load(myfile)
                    if data.size == 0:
                        pass

                    else:
                        for news in data:
                            article = []
                            for item in tokenizer.tokenize(news):
                                if "Alan" in item or "Bernanke" in item:
                                    print item
                                    article.append(item)
                                else:
                                    pass
                        np.save("../../../../tmp2/finance_data/filtered_opinion/" + str(single_date.strftime("%Y%m%d")) + ".npy", article)



                myfile.close()

            

            else:
                print "No such directory "+ row[0]
                pass

            prevRow = row[0]

def readFile():
    with open('../fed_rates/fed_date_rate_training.csv', 'r') as c1:
        reader = csv.reader(c1)
        next(reader, None)  # skip the headers
        prevRow = "2004-11-01"

        for row in reader:

            cur_year, cur_month, cur_day = str(row[0]).split("-")
            pre_year, pre_month, pre_day = prevRow.split("-")

            start_date = date(int(pre_year), int(pre_month), int(pre_day))
            end_date = date(int(cur_year), int(cur_month), int(cur_day))

            for single_date in daterange(start_date, end_date):

                if os.path.isfile("../../../../tmp2/finance_data/filtered_opinion/" + str(single_date.strftime("%Y%m%d")) + ".npy"):

                    with open("../../../../tmp2/finance_data/filtered_opinion/" + str(single_date.strftime("%Y%m%d")) + ".npy", 'r') as myfile:

                        print str(single_date.strftime("%Y-%m-%d"))
                        data = np.load(myfile)
                        if data.size == 0:
                            pass

                        else:
                            for news in data:
                                print news


            

                else:
                    print "pass posting list error: "+ row[0]
                    pass

                prevRow = row[0]




if __name__ == '__main__':
    detectOpinion()
    #readFile()
    