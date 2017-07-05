import numpy

# LSTM for sequence classification in the IMDB dataset
import sys
import csv
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


# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
keras_path = './keras.json'
top_words = 5000
(X_data, y_data), (X_test, y_test) = imdb.load_data(nb_words=top_words)
(X_data2, y_data2), (X_test2, y_test2) = imdb.load_data()

word_index = imdb.get_word_index(path=keras_path)
print(str('X_data2:{}').format(X_data2[0]))
print('\n')
print(str('X_data:{}').format(X_data[0]))
bigger  = []
bigger_element = []
for i in range(len(X_data2[0])):
    if(X_data2[0][i]>5000):
        bigger.append(i)
        bigger_element.append(X_data2[0][i])
for i in range(len(bigger)):
    index = bigger[i]
    print(str('bigger:{}').format(bigger[i]))
    print(str('X_data2:{}').format(X_data2[0][index]))
    print(str('X_data:{}').format(X_data[0][index]))
print('X_data2 length')
print(len(X_data2))
print('X_data length')
print(len(X_data))

df = word_index["good"]
print(str('df:{}').format(df))
test = word_index.get(1)#2
#test = word_index[2]
print(str('test:{}').format(test))


# truncate and pad input sequences
max_review_length = 500
X_data = sequence.pad_sequences(X_data, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

#X_data2 = sequence.pad_sequences(X_data2, maxlen=max_review_length)
#X_test2 = sequence.pad_sequences(X_test2, maxlen=max_review_length)


# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_data, y_data, nb_epoch=2, batch_size=64)
result = model.predict(X_data)
print('X_data prediction')
print(result)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
