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
from keras import optimizers
from keras import regularizers
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
#prediction column = target + 3  # target 6, 54
target = 13

###parameter
sample_weight_para = 0.5
class_weight_para = 0.5

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
outputfilename = 'testing_predictions_train_2005_2008_test_1998_2004_wi=88584_tw=5000_mrl=500_nb_epoch=10_Dropout=0.2.txt'
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
    #tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    test_i = 0.0
    test_count = 0
    all_sectionX = []
    all_sectionY = []
    for ind,ser in enumerate(start_end_rate):
        date_of_meeting = datetime.strptime(ser[1],'%Y-%m-%d')
        date_of_meeting = datetime.strftime(date_of_meeting,'%Y%m%d')
        date_of_meeting_list.append(date_of_meeting)
        data_for_date[date_of_meeting] = []
        rate_for_date[date_of_meeting] = ser[2]
        t = []
        sectionX = []
        sectionY = []
        for f in article_file[ind]:
            if os.path.isfile(article_path+f):
                day = np.load(article_path+f)
                if(len(day) == 0):
                    continue
                #tokenizer.fit_on_texts(day)
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

    X_data = X_data[1:]
    y_data = np.asarray(y_data)
    return X_data, y_data, article_file, actual_article_length
def data_formation(sectionX, sectionY, target):
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for i in range(target):
        temp_sectionX = copy.deepcopy(sectionX[i])
        temp_sectionY = copy.deepcopy(sectionY[i])
        for j in range(len(temp_sectionX)):
            X_train.append(temp_sectionX[i])
            y_train.append(temp_sectionY[i])
    for i in range(len(sectionX[target])):
        X_test.append(sectionX[target][i])
        y_test.append(sectionY[target][i])
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)
    return X_train, y_train, X_test, y_test
def training_sample_weight_formation(X_train, y_train, target, actual_document_length):
    sample_weight = half_life(X_train, y_train, target, actual_document_length)
    return sample_weight

def half_life(X_train, y_train, target, actual_document_length):
    sample_weight = []
    #value = 100.0
    value = 1.0
    weight_classify = []
    for i in range(target):
        weight_classify.append(value)
        value = value/2
    #a.sort 從小排到大
    weight_classify.sort()
    #print('weight_classify')
    #print(weight_classify)
    classify_index = 0
    doc_count = 0
    weight = weight_classify[0]
    for i in range(len(X_train)):
        if(doc_count >= actual_document_length[classify_index]):
            classify_index = classify_index + 1
            weight = weight_classify[classify_index]
            doc_count = 0
        sample_weight.append(weight)
        doc_count = doc_count + 1
    sample_weight = np.asarray(sample_weight)
    return sample_weight
def training_class_weight_formation(X_train, y_train, target, actual_document_length, record_sectionY):
    class_weight = reciprocal(X_train, y_train, target, actual_document_length, record_sectionY)
    return class_weight
def reciprocal(X_train, y_train, target, actual_document_length, record_sectionY):
    class_weight = []
    ratio_label = []
    count_label = []
    for i in range(y_train.shape[1]):
        count_label.append(0)
    for i in range(y_train.shape[1]):
        ratio_label.append(0.0)
    #print('temp_label')
    for i in range(target):
        temp_label = record_sectionY[i]
        temp_length = actual_document_length[i]
        #print(temp_label)
        ratio_label[temp_label] = ratio_label[temp_label] + float(temp_length)
        count_label[temp_label] = count_label[temp_label] + float(temp_length)
    for i in range(len(ratio_label)):
        if(ratio_label[i] == 0):
            ratio_label[i] = 1
        else:
            ratio_label[i] = 1/(ratio_label[i]/float(len(y_train)))
    #print('count_label')
    #print(count_label)
    #print('ratio_label')
    #print(ratio_label)
    classify_index = 0
    doc_count = 0
    weight = 0.0
    temp = y_train[i].tolist()
    index = temp.index(max(temp))
    weight = ratio_label[index]

    for i in range(len(y_train)):
        if(doc_count >= actual_document_length[classify_index]):
            classify_index = classify_index + 1
            temp = y_train[i].tolist()
            index = temp.index(max(temp))
            weight = ratio_label[index]
            #print('weight')
            #print(weight)
            doc_count = 0
        class_weight.append(weight)
        doc_count = doc_count + 1
    class_weight = np.asarray(class_weight)
    #print('class_weight[0]')
    #print(class_weight[0])
    return class_weight
"""
#tokenizer = Tokenizer(nb_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', 
   # lower=True, split=" ", char_level=False)

#load tokenizer from pickle file made from tokenize_document.py (1998~2008)
with open(token_file, 'rb') as input:
    tokenizer = pickle.load(input)
tokenizer = del_higher_top_words(tokenizer, top_words)

X_data, y_data, article_file, article_length = load_data(article_all_path, rates_all_file, top_words, tokenizer)
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
    pickle.dump((X_data, y_onehot, article_file, article_length), store)
"""
X_data = []
y_data = []
article_file = []
article_length = []
actual_document_length = []
sectionX = []
sectionY = []
encoder = LabelEncoder()
print('load')
with open(src_file, 'rb') as input:
    X_data, y_onehot, article_file, article_length, actual_document_length, encoder = pickle.load(input)
    """
    print('X_data[0]')
    print(X_data[0])
    print('y_onehot[0]')
    print(y_onehot[0])
    print('article_file[0]')
    print(article_file[0])
    print('article_length[0]')
    print(article_length[0])
    print('X_data type')
    print(type(X_data))
    print('y_onehot type')
    print(type(y_onehot))
    print('length of X_data')
    print(len(X_data))
    print('length of y_onehot')
    print(len(y_onehot))
    print('length of article_length')
    print(len(article_length))
    """
count = 0
for i in range(len(actual_document_length)):
    count = count + actual_document_length[i]

num_arfile = 0
section_id = 0
tempX = []
tempY = []
temp = []
for i in range(len(y_onehot)):
    tempX.append(X_data[i])
    tempY.append(y_onehot[i])
    num_arfile = num_arfile + 1
    if(num_arfile>=actual_document_length[section_id]):
        deepcopy_tempX = copy.deepcopy(tempX)
        deepcopy_tempX = np.asarray(deepcopy_tempX)
        sectionX.append(deepcopy_tempX)
        tempX = []
        deepcopy_tempY = copy.deepcopy(tempY)
        deepcopy_tempY = np.asarray(deepcopy_tempY)
        sectionY.append(deepcopy_tempY)
        tempY = []
        num_arfile = 0
        section_id = section_id + 1
sectionX = np.asarray(sectionX)
sectionY = np.asarray(sectionY)
"""
print('sectionX')
print(sectionX)
print('sectionY')
print(sectionY)
print('sectionX[0]')
print(sectionX[0])
print('sectionY[0]')
print(sectionY[0])
print('sectionX[0][0]')
print(sectionX[0][0])
print('sectionY[0][0]')
print(sectionY[0][0])

print('type of sectionX')
print(type(sectionX))
print('type of sectionX[0]')
print(type(sectionX[0]))
print('type of sectionX[0][0]')
print(type(sectionX[0][0]))
print('type of sectionY')
print(type(sectionY))
print('type of sectionY[0]')
print(type(sectionY[0]))
print('type of sectionY[0][0]')
print(type(sectionY[0][0]))
print('length of sectionX')
print(len(sectionX))
print('length of sectionY')
print(len(sectionY))
print('length of sectionX[91]')
print(len(sectionX[91]))
print('length of sectionY[91]')
print(len(sectionY[91]))
print('length of X_data')
print(len(X_data))
print('length of y_onehot')
print(len(y_onehot))
print('count')
print(count)
"""

#print('sectionY[0][0]')
#print(sectionY[0][0])
#print('type of sectionY[0][0]')
#print(type(sectionY[0][0]))
#print('len(sectionY)')
#print(len(sectionY))
record_sectionY = []
count = 0
for i in range(len(sectionY)):
    temp = sectionY[i][0].tolist()
    index = temp.index(max(temp))
    record_sectionY.append(index)
    #index = np.where(sectionY[i][0] == 1)
    #print(sectionY[k][0])
    #count = count + 1
    #print(count)
#print('record_sectionY')
for i in range(len(record_sectionY)):
    if(len(sectionY[i])== 0):
        print("sgregergegedgedrdgge")
        print(i)
    #print(record_sectionY[i])

"""
print('actual_document_length[58]')
print(actual_document_length[58])
print('len(sectionX)')
print(len(sectionX))
print('len(sectionY)')
print(len(sectionY))
print('len(actual_document_length)')
print(len(actual_document_length))
print('len(article_file)')
print(len(article_file))
print('article_file[len(article_file)-1]')
print(article_file[len(article_file)-1])
"""
sectionY_count = 0
sectionX_count = 0
for i in range(len(sectionY)):
    sectionY_count = sectionY_count + len(sectionY[i])
for i in range(len(sectionX)):
    sectionX_count = sectionX_count + len(sectionX[i])
"""
print('len(y_onehot)')
print(len(y_onehot))
print('sectionX_count')
print(sectionX_count)
print('sectionY_count')
print(sectionY_count)
print('actual_document_length[91]')
print(actual_document_length[91])

print('record_sectionY')
for i in range(len(record_sectionY)):
    print(record_sectionY[i])
"""
#print(article_file[target])
X_train, y_train, X_test, y_test = data_formation(sectionX, sectionY, target)
"""
print('X_test')
print(X_test)
print('X_test[0]')
print(X_test[0])
print('y_test')
print(y_test)
print('y_test[0]')
print(y_test[0])
print('encoder.classes_')
print(encoder.classes_)

print('encoder.classes_(y_test[0])')
"""
temp = y_test[0].tolist()
index = temp.index(max(temp))
#print('index')
#print(index)
print(encoder.classes_[index])

"""
print('len(X_train)')
print(len(X_train))
print('len(y_train)')
print(len(y_train))
print('len(X_test)')
print(len(X_test))
print('len(y_test)')
print(len(y_test))
"""

training_length = 0
for i in range(target):
    training_length = training_length + actual_document_length[i]

testing_length = actual_document_length[target]



sample_weight = training_sample_weight_formation(X_train, y_train, target, actual_document_length)
"""
print('sample_weight')
print(sample_weight)
print('len(sample_weight)')
print(len(sample_weight))
print('type(sample_weight)')
print(type(sample_weight))
"""
class_weight = training_class_weight_formation(X_train, y_train, target, actual_document_length, record_sectionY)
"""
print('class_weight')
print(class_weight)
print('len(class_weight)')
print(len(class_weight))
print('type(class_weight)')
print(type(class_weight))
"""
weight_function = []
for i in range(len(class_weight)):
    #value = sample_weight[i] * class_weight[i]
    value = (sample_weight_para * sample_weight[i]) + (class_weight_para * class_weight[i])
    weight_function.append(value)
weight_function = np.asarray(weight_function)
###

#
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length)) #top_words embedding_words
model.add(Dense(64, input_dim=64, kernel_regularizer=regularizers.l2(0.001)))
#model.add(Dense(64, input_dim=64, kernel_regularizer=regularizers.l2(0.001), activity_regularizer=regularizers.l1(0.001)))
model.add(Dropout(0.2))
model.add(LSTM(20))
model.add(Dropout(0.2)) 
model.add(Dense(y_train.shape[1], activation='sigmoid')) #tanh sigmoid
#sgd = keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=0.0001, nesterov=True)
adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) #optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy']) #categorical_crossentropy  binary_crossentropy
print(model.summary())
history = model.fit(X_train, y_train, validation_data=(X_test, y_test) , sample_weight=weight_function, nb_epoch=num_epoch, batch_size=64) 
#history = model.fit(X_train, y_train, validation_data=(X_test, y_test) , sample_weight=None, nb_epoch=num_epoch, batch_size=64) 
# Final evaluation of the model
train_scores = model.evaluate(X_train, y_train, verbose=0)


#print(str('len(result):{}').format(len(result)))
test_scores = model.evaluate(X_test, y_test, verbose=0)
print("Train Accuracy: %.2f%%" % (train_scores[1]*100))
print("Test Accuracy: %.2f%%" % (test_scores[1]*100))
#ACC = print(round(test_scores[1]*100, 2))
# list all data in history
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left') #plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('acc_0.0_1.0.png') ;
plt.clf()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')#plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('loss_0.0_1.0.png') ;

result = model.predict(X_test)
print(result)

result_value = []
for round in result:
    temp_large = max(list(round))
    idx = list(round).index(temp_large)
    value = encoder.classes_[idx]
    result_value.append(value)
print(str('len(y_test):{}').format(len(y_test)))
print(str('len(result_value):{}').format(len(result_value)))

count = 0.0
for i in range(len(y_test)):
    temp = y_test[i].tolist()
    index = temp.index(max(temp))
    if(encoder.classes_[index] == result_value[i]):
        count = count + 1
ratio = count/len(y_test)

with open('test result.txt','w') as fout:
    fout.write(str('y_test     '))
    fout.write(str('result_value '))
    fout.write('\n')
    for i in range(len(y_test)):
        temp = y_test[i].tolist()
        index = temp.index(max(temp))
        fout.write(str('{} ').format( encoder.classes_[index] ))
        fout.write(str('{}    ').format( result_value[i]))
        fout.write('\n')
    fout.close()

with open(outputfilename,'w') as fout:
    fout.write(str('train Accuracy:{}\n').format( train_scores[1]*100 ))
    fout.write(str('test Accuracy:{}\n').format( test_scores[1]*100 ))
    fout.write(str('test ratio:{}').format( ratio*100 ))
    fout.close()
#with open(outputfilename, "w") as text_file:
   # text_file.write("Accuracy: %.2f%%", % ACC)

