import numpy as np
from collections import Counter
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
"""
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
"""
import pickle

#IMDB word_index 88584
#length of word index
MAX_NB_WORDS = 88584 # 100

# load the dataset but only keep the top n words, zero the rest
top_words = 88584 # 5000 20000

x = np.arange(0, 5, 0.1);
y = np.sin(x)
plt.plot(x, y)
plt.show()
plt.savefig('test.png') ;

"""
max_index = []
a = np.arange(4).reshape((2,2))
print('a')
print(a)
for i in a:
    temp_large = max(list(i))
    idx = list(i).index(temp_large)
    max_index.append(idx)
print('max_index')
print(max_index)
"""

"""
y_train = [1,0,-1,1,0,-1,1,0,-1]
y_test = [-1,0,1,1,1,-1,0,1]

encoder = LabelEncoder()
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

encoded_Ytest = encoder.transform(y_test)
# convert integers to dummy variables (i.e. one hot encoded)
y_test_onehot = np_utils.to_categorical(encoded_Ytest)

print('y_test_onehot')
print(y_test_onehot)
print('print(encoder.classes_)')
print(encoder.classes_)
print('print(encoder.classes_[0])')
print(encoder.classes_[0])
"""
