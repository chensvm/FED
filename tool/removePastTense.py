#-*- coding: utf-8 -*-

import numpy as np
import math
from nltk.corpus import stopwords
from datetime import timedelta, date
import nltk.data
import os




def removePastTense():

    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    start_date = date(1997, 1, 1)
    end_date = date(2010, 12, 31)

    #for single_date in daterange(start_date, end_date):

        #with open('../filtered_articles/nytimes/' +str(single_date.strftime("%Y"))+"/"+ str(single_date.strftime("%Y%m%d")) + ".npy", 'r') as myfile:
    with open('../filtered_articles/nytimes/2010/20101231.npy', 'r') as myfile:
        #print str(single_date.strftime("%Y-%m-%d"))
        data = np.load(myfile)


        if data.size == 0:
            pass

        else:
            for news in data:

                article = []
                # exact sentence from paragraph
                for item in tokenizer.tokenize(news):

                    text = nltk.word_tokenize(item)
                    tagged = nltk.pos_tag(text)

                    tense = {}
                    tense["future"] = len([word for word in tagged if word[1] == "MD"])
                    # MD: modal	could, will
                    tense["present"] = len([word for word in tagged if word[1] in ["VBP", "VBZ", "VBG"]])
                    # VBP: sing. present, VBZ: 3rd person sing. present, VBG: gerund/present participle
                    tense["past"] = len([word for word in tagged if word[1] in ["VBD"]])
                    # VBD: past tense, VBN: past participle

                    if tense["past"] > tense["present"] and tense["past"] > tense["future"]:
                        #print "**future tense"
                        pass

                    else:
                        #print "**unknown tense"
                        article.append(item)

            np.save("20101231.npy", article)

                #np.save('../../../tmp2/finance_data/filtered_articles_remove_past/nytimes/' +str(single_date.strftime("%Y"))+'/'+ str(single_date.strftime("%Y%m%d")) + ".npy", article)

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

if __name__ == '__main__':

    removePastTense()
