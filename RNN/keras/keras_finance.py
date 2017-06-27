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
from keras.layers import Dropout
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pickle
import json
print(round(12.5656, 1))

#IMDB word_index 88584
#length of word index
MAX_NB_WORDS = 88584 # 100

# load the dataset but only keep the top n words, zero the rest
top_words = 5000 # 5000 20000

#model.add(Embedding
#embedding_words = 88584

# truncate and pad input sequences
max_review_length = 500 # 500

#number of epoch
num_epoch = 15

#article_path = '/tmp2/finance/nytimes/'# for hp machine
article_train_path = '/tmp2/finance2/nytimes/training_data/' # for hp, cuda3 machine 2005~2008
#article_train_path = '/tmp2/finance2/nytimes/temp/'# 2005 
#article_train_path = '/tmp2/finance2/nytimes/temp2/'# 2005~2007
#article_train_path = '/tmp2/finance2/nytimes/temp_2008/'#
article_test_path = '/tmp2/finance2/nytimes/testing1998_2004/'# 1998~2004
#article_test_path = '/tmp2/finance2/nytimes/2000_2001/'# 2000~2001
#article_test_path = '/tmp2/finance2/nytimes/temp_2004/'# 2004
#article_test_path = '/tmp2/finance2/nytimes/2001/'# 2001
token_path = './token_index/' 
#token_file = token_path + 'token2004_2005_index.pkl'#2004~2005
token_file = token_path + 'token1998_2008_index.pkl'#1998~2008
#token_file = token_path + 'token2000_2001_index.pkl'#2000~2001
#token_file = token_path + 'token2008_index.pkl'#2008

rates_path = '../fed_rates/'
rates_train_file = rates_path + 'fed_date_rate_training.csv'
rates_test_file = rates_path + 'fed_date_rate_testing.csv'
#outputfilename = 'training_predictions_train_2008_test_2008_wi=88584_tw=5000_mrl=500_nb_epoch=10_Dropout=0.2.txt'
#outputfilename = 'testing_predictions_train_2005_2008_test_2000_2001_wi=88584_tw=5000_mrl=500_nb_epoch=10_Dropout=0.2.txt'
outputfilename = 'testing_predictions_train_2005_2008_test_1998_2004_wi=88584_tw=5000_mrl=500_nb_epoch=15_Dropout=0.2.txt'
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
    #tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
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
                #tokenizer.fit_on_texts(day)
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
    X_data = X_data[1:]
    y_data = np.asarray(y_data)
    return X_data, y_data, tokenizer




#tokenizer = Tokenizer(nb_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', 
   # lower=True, split=" ", char_level=False)
with open(token_file, 'rb') as input:
    tokenizer = pickle.load(input)
    #print('tokenizer.word_index[good]')
    #print(tokenizer.word_index['good'])
tokenizer = del_higher_top_words(tokenizer, top_words)
#print('tokenizer.word_index')
#print(tokenizer.word_index)
#print('tokenizer.word_index[good]')
#print(tokenizer.word_index['good'])
encoder = LabelEncoder()
X_train , y_train, train_tokenizer = load_data(article_train_path, rates_train_file, top_words, tokenizer)

# encode class values as integers
encoder.fit(y_train)
encoded_Ytrain = encoder.transform(y_train)
# convert integers to dummy variables (i.e. one hot encoded)
y_train_onehot = np_utils.to_categorical(encoded_Ytrain)
print('y_train_onehot')
print(y_train_onehot)
print('print(encoder.classes_)')
print(encoder.classes_)
print('print(encoder.classes_[0])')
print(encoder.classes_[0])

X_test , y_test, test_tokenizer = load_data(article_test_path, rates_test_file, top_words, train_tokenizer)

encoded_Ytest = encoder.transform(y_test)
# convert integers to dummy variables (i.e. one hot encoded)
y_test_onehot = np_utils.to_categorical(encoded_Ytest)


###
#exit()
#
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length)) #top_words embedding_words
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(y_train_onehot.shape[1], activation='sigmoid')) #tanh sigmoid
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) #categorical_crossentropy  binary_crossentropy
print(model.summary())
history = model.fit(X_train, y_train_onehot, nb_epoch=num_epoch, batch_size=64) 
# Final evaluation of the model
train_scores = model.evaluate(X_train, y_train_onehot, verbose=0)
result = model.predict(X_train)
print(result)


print(str('len(result):{}').format(len(result)))
test_scores = model.evaluate(X_test, y_test_onehot, verbose=0)
print("Train Accuracy: %.2f%%" % (train_scores[1]*100))
#print("Test Accuracy: %.2f%%" % (test_scores[1]*100))
#ACC = print(round(test_scores[1]*100, 2))

with open(outputfilename,'w') as fout:
    fout.write(str('train Accuracy:{}\n').format( train_scores[1]*100 ))
    fout.write(str('test Accuracy:{}').format( test_scores[1]*100 ))
    fout.close()
#with open(outputfilename, "w") as text_file:
   # text_file.write("Accuracy: %.2f%%", % ACC)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
 #plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left') #plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')#plt.legend(['train', 'test'], loc='upper left')
plt.show()