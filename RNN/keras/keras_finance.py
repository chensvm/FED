# LSTM for sequence classification in the IMDB dataset
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
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
print(round(12.5656, 1))

#IMDB word_index 88584
#length of word index
MAX_NB_WORDS = 88584 # 100

# load the dataset but only keep the top n words, zero the rest
top_words = 20000 # 5000

#model.add(Embedding
#embedding_words = 88584

# truncate and pad input sequences
max_review_length = 500 # 500

#article_path = '/tmp2/finance/nytimes/'# for hp machine
article_train_path = '/tmp2/finance2/nytimes/training_data/' # for hp, cuda3 machine
#article_train_path = '/tmp2/finance2/nytimes/temp/'
#article_train_path = '/tmp2/finance2/nytimes/temp2/'
rates_path = '../fed_rates/'
rates_train_file = rates_path + 'fed_date_rate_training.csv'
outputfilename = 'training_predictions_training_wi=88584_tw=5000_mrl=500_nb_epoch=10.txt'

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

def del_higher_top_words(sequences, top_words_number):
    for i in range(len(sequences)):
        temp_sequence = sequences[i]
        for j in range(len(temp_sequence)):
            if(temp_sequence[j]>=top_words_number):# maybe have bug
                temp_sequence[j] = 0
    return sequences

    

def load_data(article_path, rates_file, top_words):
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
    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    test_i = 0.0
    test_count = 0
    for ind,ser in enumerate(start_end_rate):
        date_of_meeting = datetime.strptime(ser[1],'%Y-%m-%d')
        date_of_meeting = datetime.strftime(date_of_meeting,'%Y%m%d')
        date_of_meeting_list.append(date_of_meeting)
        data_for_date[date_of_meeting] = []
        rate_for_date[date_of_meeting] = ser[2]
        for f in article_file[ind]:
            if os.path.isfile(article_path+f):
                day = np.load(article_path+f)
                if(len(day) == 0):
                    continue
                tokenizer.fit_on_texts(day)
                counttt = 0
                sequences = tokenizer.texts_to_sequences(day)
                #sequences = choose_top_words(sequences, top_words) # unfinished , hope it can get frequency order of words and give it ID
                sequences =del_higher_top_words(sequences, top_words) # only delete id that higher than top_words
                data = sequence.pad_sequences(sequences, maxlen=max_review_length)
                X_data = np.concatenate((X_data, data), axis = 0)
                for i in range(len(day)):
                    y_data.append(rate_for_date[date_of_meeting])
                test_i = test_i +1
                if(test_i %10 == 0):
                    print(str('process:{}').format(test_i/len(article_file)))
    X_data = X_data[1:]
    y_data = np.asarray(y_data)
    return X_data, y_data




"""
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
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
test_i = 0.0
test_count = 0
for ind,ser in enumerate(start_end_rate):
	date_of_meeting = datetime.strptime(ser[1],'%Y-%m-%d')
	date_of_meeting = datetime.strftime(date_of_meeting,'%Y%m%d')
	date_of_meeting_list.append(date_of_meeting)
	data_for_date[date_of_meeting] = []
	rate_for_date[date_of_meeting] = ser[2]
	for f in article_file[ind]:
		if os.path.isfile(article_path+f):
			day = np.load(article_path+f)
			if(len(day) == 0):
				continue
			tokenizer.fit_on_texts(day)
			counttt = 0
			sequences = tokenizer.texts_to_sequences(day)
			data = sequence.pad_sequences(sequences, maxlen=max_review_length)
			X_data = np.concatenate((X_data, data), axis = 0)
			for i in range(len(day)):
				y_data.append(rate_for_date[date_of_meeting])
			print(str('process:{}').format(test_i/len(article_file)))
y_data = np.asarray(y_data)
X_data = X_data[1:]
#print(str('start_end_rate:{}').format(start_end_rate))
"""
X_train , y_train = load_data(article_train_path, rates_train_file, top_words)
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length)) #top_words embedding_words
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, nb_epoch=10, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_train, y_train, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
ACC = print(round(scores[1]*100, 2))
with open(outputfilename,'w') as fout:
    fout.write(str('Accuracy:{}').format( scores[1]*100 ))
    fout.close()
#with open(outputfilename, "w") as text_file:
   # text_file.write("Accuracy: %.2f%%", % ACC)

"""
temp_train = []
temp_pad_train = []
texts = []
texts.append('I am so handsome')
texts.append('I am a pig')
sec_texts = []
sec_texts.append('you are I am')
print(str('texts[]:{}').format(texts))
print(str('sec_texts[]:{}').format(sec_texts))
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
tokenizer.fit_on_texts(sec_texts)
sequences = tokenizer.texts_to_sequences(texts)
sec_sequences = tokenizer.texts_to_sequences(sec_texts)
print('first sequences')
print(sequences)
print('sec_sequences')
print(sec_sequences)
temp_train = sequences + sec_sequences
sequences = sequence.pad_sequences(sequences, maxlen=max_review_length)
sec_sequences = sequence.pad_sequences(sec_sequences, maxlen=max_review_length)
print('after padding first sequences')
print(sequences)
print('after padding sec_sequences')
print(sec_sequences)
word_index = tokenizer.word_index
print('sec word_index')
print(word_index)
temp_pad_train = sequences + sec_sequences
print('temp_train')
print(temp_train)
print('temp_pad_train')
print(temp_pad_train)
"""




"""
# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
#keras_path = './keras.json'
top_words = 5000
(X_data, y_data), (X_test, y_test) = imdb.load_data(nb_words=top_words)

#word_index = imdb.get_word_index(path=keras_path)
#test = word_index["good"]

# truncate and pad input sequences
max_review_length = 500
X_data = sequence.pad_sequences(X_data, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)


#print('X_data[0] pad_sequences')
#print(X_data[0])


# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_data, y_data, nb_epoch=3, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
"""