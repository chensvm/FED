#-*- coding: utf-8 -*-
import nltk
from nltk.stem.lancaster import LancasterStemmer
import json
import datetime
import numpy as np
import re
from datetime import timedelta, date
import csv
from nltk.corpus import stopwords
import os

stemmer = LancasterStemmer()
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)



# with open('../fed_rates/fed_date_rate_testing.csv', 'r') as c1:
#     reader = csv.reader(c1)
#     next(reader, None)  # skip the headers
#     prevRow = "1998-01-01"
#
#     for row in reader:
#
#         cur_year, cur_month, cur_day = str(row[0]).split("-")
#         pre_year, pre_month, pre_day = prevRow.split("-")
#
#         start_date = date(int(pre_year), int(pre_month), int(pre_day))
#         end_date = date(int(cur_year), int(cur_month), int(cur_day))
#         collection = []



            #if os.path.isfile('../../../../tmp2/finance_data/filtered_articles_remove_past/' + str(single_date.strftime("%Y%m%d")) + ".npy"):

with open('../../../../tmp2/finance_data/filtered_articles_remove_past/' + "19991231.npy",
          'r') as myfile:



    data = np.load(myfile)

    if data.size == 0:
        pass

    else:
        for news in data:

            print news
            print "%%%%%%%%%%%%"

            # for item in tokenizer.tokenize(news):
            #      print cur_year + '-' + cur_month + '-' + cur_day + ',' + str(
            #         classify(item)) + "\n"
            #      quit()




print "######## finish one section ########"





