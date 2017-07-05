# LSTM for sequence classification in the IMDB dataset
import numpy as np
import sys
import csv
import copy
from collections import Counter
from datetime import date, timedelta,datetime
import os.path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Dropout
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import pickle
import json
print(round(12.5656, 1))

#IMDB word_index 88584
#length of word index
MAX_NB_WORDS = 88584 # 100

# load the dataset but only keep the top n words, zero the rest
top_words = 20000 # 5000 20000

#model.add(Embedding
#embedding_words = 88584

# truncate and pad input sequences
max_review_length = 500 # 500

#number of epoch
num_epoch = 20

#article_path = '/tmp2/finance/nytimes/'# for hp machine
article_train_path = '/tmp2/finance2/nytimes/training_data/' # for hp, cuda3 machine 2005~2008
#article_train_path = '/tmp2/finance2/nytimes/temp/'# 2005 
#article_train_path = '/tmp2/finance2/nytimes/temp2/'# 2005~2007
#article_train_path = '/tmp2/finance2/nytimes/temp_2008/'#
article_test_path = '/tmp2/finance2/nytimes/testing1998_2004/'# 1998~2004
#article_test_path = '/tmp2/finance2/nytimes/2000_2001/'# 2000~2001
#article_test_path = '/tmp2/finance2/nytimes/temp_2004/'# 2004
#article_test_path = '/tmp2/finance2/nytimes/2001/'# 2001
article_all_path = '/tmp2/finance2/nytimes/1998_2008/'
token_path = './token_index/' 
#token_file = token_path + 'token2004_2005_index.pkl'#2004~2005
token_file = token_path + 'token1998_2008_index.pkl'#1998~2008
#token_file = token_path + 'token2000_2001_index.pkl'#2000~2001
#token_file = token_path + 'token2008_index.pkl'#2008
src_path = './training_src/'
src_file = src_path + 'src1998_2008.pkl'

rates_path = '../fed_rates/'
rates_train_file = rates_path + 'fed_date_rate_training.csv'
rates_test_file = rates_path + 'fed_date_rate_testing.csv'
rates_all_file = rates_path + 'fed_date_rate_all.csv'
#outputfilename = 'training_predictions_train_2008_test_2008_wi=88584_tw=5000_mrl=500_nb_epoch=10_Dropout=0.2.txt'
#outputfilename = 'testing_predictions_train_2005_2008_test_2000_2001_wi=88584_tw=5000_mrl=500_nb_epoch=10_Dropout=0.2.txt'
outputfilename = 'testing_predictions_train_2005_2008_test_1998_2004_wi=88584_tw=5000_mrl=500_nb_epoch=20_Dropout=0.2.txt'
#outputfilename = 'testing_predictions_train_2005_test_2004_wi=88584_tw=5000_mrl=500_nb_epoch=3_Dropout=0.2.txt'
# fix random seed for reproducibility
np.random.seed(7)

def choose_top_words(sequences, top_words_number):
    #print('sequences')
    #print(sequences)
    #print('sequences length')
   # print(len(sequences))
    #print('sequences[0] length')
    #print(len(sequences[0]))
    all_top_element = []
    for i in range(len(sequences)):
        top_element = []
        temp_sequence = copy.deepcopy(sequences[i])
        count_temp_co = Counter(temp_sequence)
        count_temp = count_temp_co.most_common(top_words_number)

        for k in range(len(count_temp)):
            top_element.append(count_temp[k][0])
        all_top_element.append(top_element)
        for j in range(len(sequences[i])):
            temp_word = sequences[i][j]
            if(temp_word not in top_element):
                sequences[i][j] = 0
    return sequences

def del_higher_top_words(tokenizer, top_words_number):
    for key, value in tokenizer.word_index.items():
        #print(str('key:{}').format(key))
        #print(str('value:{}').format(value))
        #print('\n')
        if int(value)>=top_words_number:
            tokenizer.word_index[key] = 0
    return tokenizer

    

def load_data(article_path, rates_file, top_words, tokenizer):
    start_end_rate = []
    prev_date = ''
    rates = {}
    f = open(rates_file, 'r')

    for row in csv.DictReader(f):
        rates[row['date']] = row['rate']
        t = []
        t.append(prev_date)
        t.append(row['date'])
        t.append(int(row['rate']))
        start_end_rate.append(t)
        prev_date = row['date']
    del start_end_rate[0]


    ## constructing array with all npy filenames for start_end_rate[]
    article_file = []
    for ser in start_end_rate:
        start_date = datetime.strptime(ser[0],'%Y-%m-%d')
        start_date = start_date + timedelta(days=1)
        end_date = datetime.strptime(ser[1],'%Y-%m-%d')
        t = []
        count_date = start_date
        while count_date < end_date:
            t.append(str(count_date.year)+'/'+datetime.strftime(count_date,'%Y%m%d')+'.npy')
            count_date = count_date + timedelta(days=1)
        article_file.append(t)
    #print('start_end_rate[0]')
    #print('article_file[0]')

    ## getting all sentences and rate label for meeting_date
    ## data_for_date[date of meeting] = [array of sentences]
    ## X_data is composed of article
    data_for_date = {}
    rate_for_date = {}
    date_of_meeting_list = []
    #X_data = np.empty(0)#max_review_length
    X_data = np.arange(max_review_length).reshape(1, max_review_length)
    y_data = []
    #
    actual_article = []
    actual_article_length = []
    actual_document_length = []
    #tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    test_i = 0.0
    test_count = 0
    for ind,ser in enumerate(start_end_rate):
        date_of_meeting = datetime.strptime(ser[1],'%Y-%m-%d')
        date_of_meeting = datetime.strftime(date_of_meeting,'%Y%m%d')
        date_of_meeting_list.append(date_of_meeting)
        data_for_date[date_of_meeting] = []
        rate_for_date[date_of_meeting] = ser[2]
        t = []
        #record actual_document_length
        length_day = 0
        for f in article_file[ind]:
            if os.path.isfile(article_path+f):
                day = np.load(article_path+f)
                if(len(day) == 0):
                    continue
                #tokenizer.fit_on_texts(day)
                length_day = length_day + len(day)
                t.append(f)
                counttt = 0
                sequences = tokenizer.texts_to_sequences(day)
                #sequences = choose_top_words(sequences, top_words) # unfinished , hope it can get frequency order of words and give it ID
                #sequences =del_higher_top_words(sequences, top_words) # only delete id that higher than top_words
                data = sequence.pad_sequences(sequences, maxlen=max_review_length)
                X_data = np.concatenate((X_data, data), axis = 0)
                for i in range(len(day)):
                    y_data.append(rate_for_date[date_of_meeting])
                test_i = test_i +1
                if(test_i %10 == 0):
                    print(str('process:{}').format(test_i/len(article_file)))

        actual_article.append(t)
        actual_article_length.append(len(t))
        actual_document_length.append(length_day)
    X_data = X_data[1:]
    y_data = np.asarray(y_data)
    return X_data, y_data, article_file, actual_article_length, actual_document_length



#tokenizer = Tokenizer(nb_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', 
   # lower=True, split=" ", char_level=False)

#load tokenizer from pickle file made from tokenize_document.py (1998~2008)
with open(token_file, 'rb') as input:
    tokenizer = pickle.load(input)
tokenizer = del_higher_top_words(tokenizer, top_words)

X_data, y_data, article_file, article_length, actual_document_length = load_data(article_all_path, rates_all_file, top_words, tokenizer)
encoder = LabelEncoder()
# encode class values as integers
encoder.fit(y_data)
encoded_Ydata = encoder.transform(y_data)
# convert integers to dummy variables (i.e. one hot encoded)
y_onehot = np_utils.to_categorical(encoded_Ydata)
print('y_onehot')
print(y_onehot)
print('print(encoder.classes_)')
print(encoder.classes_)
print('print(encoder.classes_[0])')
print(encoder.classes_[0])





print('article_file[0]')
for article in article_file[0]:
    print(article)
print('article_length[0]')
print(article_length[0])

print('write')
with open(src_file, 'wb') as store:
    #pickle.dump((X_data, y_onehot, article_file, article_length), store, pickle.HIGHEST_PROTOCOL)
    pickle.dump((X_data, y_onehot, article_file, article_length, actual_document_length, encoder), store)
