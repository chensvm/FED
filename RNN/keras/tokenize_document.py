import numpy as np
import sys
import csv
import copy
from collections import Counter
from datetime import date, timedelta,datetime
import os.path
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
import pickle

#article_path = '/tmp2/finance/nytimes/'# for hp machine
#article_train_path = '/tmp2/finance2/nytimes/training_data/' # for hp, cuda3 machine
#article_train_path = '/tmp2/finance2/nytimes/temp/'# 2005
#article_train_path = '/tmp2/finance2/nytimes/temp2/'# 2005~2007
#article_train_path = '/tmp2/finance2/nytimes/temp_2008/'#2008 data has problem
#article_test_path = '/tmp2/finance2/nytimes/testing1998_2004/'# 1998~2004
#article_test_path = '/tmp2/finance2/nytimes/temp_2004/'# 2004
#article_all_path = '/tmp2/finance2/nytimes/1998_2007/'#1998~2007
article_all_path = '/tmp2/finance2/nytimes/1998_2008/'#1998~2008
token_file = 'token1998_2008_index.pkl'#1998~2008
rates_path = '../fed_rates/'
rates_train_file = rates_path + 'fed_date_rate_training.csv'
rates_test_file = rates_path + 'fed_date_rate_testing.csv'
rates_all_file = rates_path + 'fed_date_rate_all.csv' 


#IMDB word_index 88584
#length of word index
MAX_NB_WORDS = 88584 # 100

# truncate and pad input sequences
max_review_length = 500 # 500


# load the dataset but only keep the top n words, zero the rest
top_words = 20000 # 5000 20000

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

def del_higher_top_words(sequences, top_words_number):
    for i in range(len(sequences)):
        temp_sequence = sequences[i]
        for j in range(len(temp_sequence)):
            if(temp_sequence[j]>=top_words_number):# maybe have bug
                temp_sequence[j] = 0
    return sequences



def load_data(article_path, rates_file, tokenizer):
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
    X_data = []
    y_data = []
    #tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    test_i = 0.0
    test_count = 0
    for ind,ser in enumerate(start_end_rate):
        #date_of_meeting = datetime.strptime(ser[1],'%Y-%m-%d')
        #date_of_meeting = datetime.strftime(date_of_meeting,'%Y%m%d')
        #date_of_meeting_list.append(date_of_meeting)
        #data_for_date[date_of_meeting] = []
        #rate_for_date[date_of_meeting] = ser[2]
        for f in article_file[ind]:
            if os.path.isfile(article_path+f):
                day = np.load(article_path+f)
                if(len(day) == 0):
                    continue
                tokenizer.fit_on_texts(day)
                #counttt = 0
                #sequences = tokenizer.texts_to_sequences(day)
                #sequences = choose_top_words(sequences, top_words) # unfinished , hope it can get frequency order of words and give it ID
                #sequences =del_higher_top_words(sequences, top_words) # only delete id that higher than top_words
                #X_data = X_data + test
                #for i in range(len(day)):
                    #y_data.append(rate_for_date[date_of_meeting])
                test_i = test_i +1
                if(test_i %10 == 0):
                    print(str('process:{}').format(test_i/len(article_file)))
    #X_data = X_data[1:]
    #y_data = np.asarray(y_data)
    return tokenizer # X_data, y_data, tokenize



tokenizer = Tokenizer(nb_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', 
    lower=True, split=" ", char_level=False)
#X_train , y_train, train_tokenizer = load_data(article_train_path, rates_train_file, top_words, tokenizer)
#X_test , y_test, test_tokenizer = load_data(article_test_path, rates_test_file, top_words, tokenizer)
tokenizer = load_data(article_all_path, rates_all_file, tokenizer)
print('tokenizer.word_index')
print(tokenizer.word_index)
print('tokenizer.word_index[good]')
print(tokenizer.word_index['good'])

with open(token_file, 'wb') as output:
    pickle.dump(tokenizer, output, pickle.HIGHEST_PROTOCOL)
with open(token_file, 'rb') as input:
    token_test = pickle.load(input)
    print('token_test.word_index[good]')
    print(token_test.word_index['good'])