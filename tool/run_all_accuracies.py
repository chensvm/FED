import re
import nltk
import pandas as pd
import numpy as np

from datetime import date, timedelta,datetime
import string

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
english_stemmer=nltk.stem.SnowballStemmer('english')

from sklearn.feature_selection.univariate_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.linear_model import SGDClassifier, SGDRegressor,LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import random
import itertools

import sys
import os
import argparse
from sklearn.pipeline import Pipeline
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
import six
from abc import ABCMeta
from scipy import sparse
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y, check_array
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.preprocessing import normalize, binarize, LabelBinarizer
from sklearn.svm import LinearSVC

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Lambda
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.preprocessing.text import Tokenizer
from collections import defaultdict
from keras.layers.convolutional import Convolution1D
from keras import backend as K
# import seaborn as sns
# import matplotlib.pyplot as plt
# from matplotlib import cm
# %matplotlib inline
# plt.style.use('ggplot')

### TODO: ignore deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

########################################################################################
########################## NAIVE BAYES SVM IMPLEMENTATION ##############################
class NBSVM(six.with_metaclass(ABCMeta, BaseEstimator, ClassifierMixin)):
    def __init__(self, alpha=1.0, C=1.0, max_iter=10000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.C = C
        self.svm_ = [] # fuggly
    def fit(self, X, y):
        X, y = check_X_y(X, y, 'csr')
        _, n_features = X.shape

        labelbin = LabelBinarizer()
        Y = labelbin.fit_transform(y)
        self.classes_ = labelbin.classes_
        if Y.shape[1] == 1:
            Y = np.concatenate((1 - Y, Y), axis=1)
        # LabelBinarizer().fit_transform() returns arrays with dtype=np.int64.
        # so we don't have to cast X to floating point
        Y = Y.astype(np.float64)
        # Count raw events from data
        n_effective_classes = Y.shape[1]
        self.class_count_ = np.zeros(n_effective_classes, dtype=np.float64)
        self.ratios_ = np.full((n_effective_classes, n_features), self.alpha,
                                 dtype=np.float64)
        self._compute_ratios(X, Y)
        # flugglyness
        for i in range(n_effective_classes):
            X_i = X.multiply(self.ratios_[i])
            svm = LinearSVC(C=self.C, max_iter=self.max_iter)
            Y_i = Y[:,i]
            svm.fit(X_i, Y_i)
            self.svm_.append(svm) 
        return self
    def predict(self, X):
        n_effective_classes = self.class_count_.shape[0]
        n_examples = X.shape[0]

        D = np.zeros((n_effective_classes, n_examples))

        for i in range(n_effective_classes):
            X_i = X.multiply(self.ratios_[i])
            D[i] = self.svm_[i].decision_function(X_i)        
        return self.classes_[np.argmax(D, axis=0)]        
    def _compute_ratios(self, X, Y):
        """Count feature occurrences and compute ratios."""
        if np.any((X.data if issparse(X) else X) < 0):
            raise ValueError("Input X must be non-negative")
        self.ratios_ += safe_sparse_dot(Y.T, X)  # ratio + feature_occurrance_c
        normalize(self.ratios_, norm='l1', axis=1, copy=False)
        row_calc = lambda r: np.log(np.divide(r, (1 - r)))
        self.ratios_ = np.apply_along_axis(row_calc, axis=1, arr=self.ratios_)
        check_array(self.ratios_)
        self.ratios_ = sparse.csr_matrix(self.ratios_)
        #p_c /= np.linalg.norm(p_c, ord=1)
        #ratios[c] = np.log(p_c / (1 - p_c))
########################## NAIVE BAYES SVM IMPLEMENTATION END ##########################
########################################################################################

pd.options.display.max_colwidth = 100

####################################################################################START
# NLTK
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

stop = set(stopwords.words('english'))
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    string = ''.join(re.findall("[a-zA-Z ]+",normalized))
    return string.strip().lower()

def clean_str(string):
    string = re.sub(r"\\", "", string)    
    string = re.sub(r"\'", "", string)    
    string = re.sub(r"\"", "", string) 
    string = ''.join(re.findall("[a-zA-Z ]+",string))   
    return string.strip().lower()
####################################################################################END

# Get data
article_path = '/tmp3/finance/nytimes/business_news_and_title/'
rates_path = '/tmp3/finance/data/'
rates_file = rates_path + 'label_NASDAQ100_19850101_20171231.csv'

## getting date and rate
rates = []
with open(rates_file, 'r') as fin:
    for ind,row in enumerate(fin):
        if ind > 0:
            r = row.split(',')
            rates.append([r[0], int(r[1])])

### TODO: shrink dataset
rates = rates[:7900] # last 100 days has too little news
# rates = rates[:500]

## constructing array with all npy filenames for rates[]
### TODO: change number of days to get news
NUMBER_OF_DAYS = 7
article_file = []
for r in rates:
    date = datetime.strptime(r[0],'%Y-%m-%d')
    date = date - timedelta(days=1)
    t = []
    for i in range(NUMBER_OF_DAYS):
    	t.append(str(date.year)+'/'+datetime.strftime(date,'%Y%m%d')+'.npy')
    	date = date - timedelta(days=1)
    article_file.append(t)

## getting all articles and labels
labels = []
days = []
for ind,r in enumerate(rates):
	print('>> stock price {} of {}: {}, label:{}'.format(ind,len(rates),r[0],r[1]))
	date_of_meeting = datetime.strptime(r[0],'%Y-%m-%d')
	date_of_meeting = datetime.strftime(date_of_meeting,'%Y%m%d')
	news = []
	for f in article_file[ind]:
		if os.path.isfile(article_path+f):
			day = np.load(article_path+f)
			for data in day:
				title = data[0]
				article = data[1]
				news.append(title.encode('utf-8'))
	days.append(news)
	labels.append(r[1])
	print('>>> {} news for {}'.format(len(news),r[0]))

print ''

headlines = []
for i in range(0,len(days)):
    headlines.append(' '.join(str(x) for x in days[i]))

# print len(days)
# print len(headlines)
# print labels
# print days
# print headlines

# print len(labels)
# print len(days)
# print len(headlines)

labels = np.array(labels)
days = np.array(days)
headlines = np.array(headlines)

## TODO: RANDOM SHUFFLE
NUMBER_OF_SHUFFLES = 10
scores = []
for shuffle_iter in range(NUMBER_OF_SHUFFLES):
	print '======================================================='
	print 'SHUFFLE {}'.format(shuffle_iter+1)
	# shuffle indices
	indices = np.arange(labels.shape[0])
	np.random.shuffle(indices)
	labels = labels[indices]
	days = days[indices]
	headlines = headlines[indices]

	VALIDATION_SPLIT = 0.2
	nb_validation_samples = int(VALIDATION_SPLIT * len(labels))

	scores_in_iter = []

	####################################################################################START
	# Logistic Regression 1
	basicvectorizer = CountVectorizer()
	newdata = basicvectorizer.fit_transform(headlines)

	train_data = newdata[:-nb_validation_samples]
	train_label = labels[:-nb_validation_samples]
	test_data = newdata[-nb_validation_samples:]
	test_label = labels[-nb_validation_samples:]
	# print '>>>>>>> train_data:{}, train_label:{}, test_data:{}, test_label:{}'.format(train_data.shape,train_label.shape,test_data.shape,test_label.shape)

	basicmodel = LogisticRegression()
	basicmodel = basicmodel.fit(train_data, train_label)

	preds = basicmodel.predict(test_data)
	acc=accuracy_score(test_label, preds)
	print 'Logistic Regression 1 accuracy:{}'.format(acc)
	scores_in_iter.append(acc)
	####################################################################################END

	####################################################################################START
	# Logistic Regression 2
	advancedvectorizer = TfidfVectorizer(min_df=0.03, max_df=0.97, max_features = 200000, ngram_range = (2, 2))
	newdata = advancedvectorizer.fit_transform(headlines)

	train_data = newdata[:-nb_validation_samples]
	train_label = labels[:-nb_validation_samples]
	test_data = newdata[-nb_validation_samples:]
	test_label = labels[-nb_validation_samples:]
	# print '>>>>>>> train_data:{}, train_label:{}, test_data:{}, test_label:{}'.format(train_data.shape,train_label.shape,test_data.shape,test_label.shape)

	advancedmodel = LogisticRegression()
	advancedmodel = advancedmodel.fit(train_data, train_label)

	preds = advancedmodel.predict(test_data)
	acc=accuracy_score(test_label, preds)
	print 'Logistic Regression 2 accuracy:{}'.format(acc)
	scores_in_iter.append(acc)
	####################################################################################END

	####################################################################################START
	# Logistic Regression 3
	advancedvectorizer = TfidfVectorizer(min_df=0.0039, max_df=0.1, max_features = 200000, ngram_range = (3, 3))
	newdata = advancedvectorizer.fit_transform(headlines)

	train_data = newdata[:-nb_validation_samples]
	train_label = labels[:-nb_validation_samples]
	test_data = newdata[-nb_validation_samples:]
	test_label = labels[-nb_validation_samples:]
	# print '>>>>>>> train_data:{}, train_label:{}, test_data:{}, test_label:{}'.format(train_data.shape,train_label.shape,test_data.shape,test_label.shape)

	advancedmodel = LogisticRegression()
	advancedmodel = advancedmodel.fit(train_data, train_label)

	preds = advancedmodel.predict(test_data)
	acc=accuracy_score(test_label, preds)
	print 'Logistic Regression 3 accuracy:{}'.format(acc)
	scores_in_iter.append(acc)
	####################################################################################END

	####################################################################################START
	# Naive Bayes 1
	advancedvectorizer = TfidfVectorizer(min_df=0.1, max_df=0.7, max_features = 200000, ngram_range = (1, 1))
	newdata = advancedvectorizer.fit_transform(headlines)

	train_data = newdata[:-nb_validation_samples]
	train_label = labels[:-nb_validation_samples]
	test_data = newdata[-nb_validation_samples:]
	test_label = labels[-nb_validation_samples:]
	# print '>>>>>>> train_data:{}, train_label:{}, test_data:{}, test_label:{}'.format(train_data.shape,train_label.shape,test_data.shape,test_label.shape)

	advancedmodel = MultinomialNB(alpha=0.01)
	advancedmodel = advancedmodel.fit(train_data, train_label)

	preds = advancedmodel.predict(test_data)
	acc=accuracy_score(test_label, preds)
	print 'Naive Bayes 1 accuracy:{}'.format(acc)
	scores_in_iter.append(acc)
	####################################################################################END

	####################################################################################START
	# Naive Bayes 2
	advancedvectorizer = TfidfVectorizer(min_df=0.03, max_df=0.2, max_features = 200000, ngram_range = (2, 2))
	newdata = advancedvectorizer.fit_transform(headlines)

	train_data = newdata[:-nb_validation_samples]
	train_label = labels[:-nb_validation_samples]
	test_data = newdata[-nb_validation_samples:]
	test_label = labels[-nb_validation_samples:]
	# print '>>>>>>> train_data:{}, train_label:{}, test_data:{}, test_label:{}'.format(train_data.shape,train_label.shape,test_data.shape,test_label.shape)

	advancedmodel = MultinomialNB(alpha=0.0001)
	advancedmodel = advancedmodel.fit(train_data, train_label)

	preds = advancedmodel.predict(test_data)
	acc=accuracy_score(test_label, preds)
	print 'Naive Bayes 2 accuracy:{}'.format(acc)
	scores_in_iter.append(acc)
	####################################################################################END

	####################################################################################START
	# Random Forest 1
	advancedvectorizer = TfidfVectorizer(min_df=0.01, max_df=0.99, max_features = 200000, ngram_range = (1, 1))
	newdata = advancedvectorizer.fit_transform(headlines)

	train_data = newdata[:-nb_validation_samples]
	train_label = labels[:-nb_validation_samples]
	test_data = newdata[-nb_validation_samples:]
	test_label = labels[-nb_validation_samples:]
	# print '>>>>>>> train_data:{}, train_label:{}, test_data:{}, test_label:{}'.format(train_data.shape,train_label.shape,test_data.shape,test_label.shape)

	advancedmodel = RandomForestClassifier()
	advancedmodel = advancedmodel.fit(train_data, train_label)

	preds = advancedmodel.predict(test_data)
	acc=accuracy_score(test_label, preds)
	print 'Random Forest 1 accuracy:{}'.format(acc)
	scores_in_iter.append(acc)
	####################################################################################END

	####################################################################################START
	# Random Forest 2
	advancedvectorizer = TfidfVectorizer(min_df=0.03, max_df=0.2, max_features = 200000, ngram_range = (2, 2))
	newdata = advancedvectorizer.fit_transform(headlines)

	train_data = newdata[:-nb_validation_samples]
	train_label = labels[:-nb_validation_samples]
	test_data = newdata[-nb_validation_samples:]
	test_label = labels[-nb_validation_samples:]
	# print '>>>>>>> train_data:{}, train_label:{}, test_data:{}, test_label:{}'.format(train_data.shape,train_label.shape,test_data.shape,test_label.shape)

	advancedmodel = RandomForestClassifier()
	advancedmodel = advancedmodel.fit(train_data, train_label)

	preds = advancedmodel.predict(test_data)
	acc=accuracy_score(test_label, preds)
	print 'Random Forest 2 accuracy:{}'.format(acc)
	scores_in_iter.append(acc)
	####################################################################################END

	####################################################################################START
	# Gradient Boosting Machine 1
	advancedvectorizer = TfidfVectorizer(min_df=0.1, max_df=0.9, max_features = 200000, ngram_range = (1, 1))
	newdata = advancedvectorizer.fit_transform(headlines)

	train_data = newdata[:-nb_validation_samples]
	train_label = labels[:-nb_validation_samples]
	test_data = newdata[-nb_validation_samples:]
	test_label = labels[-nb_validation_samples:]
	# print '>>>>>>> train_data:{}, train_label:{}, test_data:{}, test_label:{}'.format(train_data.shape,train_label.shape,test_data.shape,test_label.shape)

	advancedmodel = GradientBoostingClassifier()
	advancedmodel = advancedmodel.fit(train_data, train_label)

	preds = advancedmodel.predict(test_data)
	acc=accuracy_score(test_label, preds)
	print 'Gradient Boosting Machine 1 accuracy:{}'.format(acc)
	scores_in_iter.append(acc)
	####################################################################################END

	####################################################################################START
	# Gradient Boosting Machine 2
	advancedvectorizer = TfidfVectorizer(min_df=0.02, max_df=0.175, max_features = 200000, ngram_range = (2, 2))
	newdata = advancedvectorizer.fit_transform(headlines)

	train_data = newdata[:-nb_validation_samples]
	train_label = labels[:-nb_validation_samples]
	test_data = newdata[-nb_validation_samples:]
	test_label = labels[-nb_validation_samples:]
	# print '>>>>>>> train_data:{}, train_label:{}, test_data:{}, test_label:{}'.format(train_data.shape,train_label.shape,test_data.shape,test_label.shape)

	advancedmodel = GradientBoostingClassifier()
	advancedmodel = advancedmodel.fit(train_data, train_label)

	preds = advancedmodel.predict(test_data)
	acc=accuracy_score(test_label, preds)
	print 'Gradient Boosting Machine 2 accuracy:{}'.format(acc)
	scores_in_iter.append(acc)
	####################################################################################END

	####################################################################################START
	# Stochastic Gradient Descent 1
	advancedvectorizer = TfidfVectorizer(min_df=0.2, max_df=0.8, max_features = 200000, ngram_range = (1, 1))
	newdata = advancedvectorizer.fit_transform(headlines)

	train_data = newdata[:-nb_validation_samples]
	train_label = labels[:-nb_validation_samples]
	test_data = newdata[-nb_validation_samples:]
	test_label = labels[-nb_validation_samples:]
	# print '>>>>>>> train_data:{}, train_label:{}, test_data:{}, test_label:{}'.format(train_data.shape,train_label.shape,test_data.shape,test_label.shape)

	advancedmodel = SGDClassifier(loss='modified_huber', n_iter=5, random_state=0, shuffle=True)
	advancedmodel = advancedmodel.fit(train_data, train_label)

	preds = advancedmodel.predict(test_data)
	acc=accuracy_score(test_label, preds)
	print 'Stochastic Gradient Descent 1 accuracy:{}'.format(acc)
	scores_in_iter.append(acc)
	####################################################################################END

	####################################################################################START
	# Stochastic Gradient Descent 2
	advancedvectorizer = TfidfVectorizer(min_df=0.03, max_df=0.2, max_features = 200000, ngram_range = (2, 2))
	newdata = advancedvectorizer.fit_transform(headlines)

	train_data = newdata[:-nb_validation_samples]
	train_label = labels[:-nb_validation_samples]
	test_data = newdata[-nb_validation_samples:]
	test_label = labels[-nb_validation_samples:]
	# print '>>>>>>> train_data:{}, train_label:{}, test_data:{}, test_label:{}'.format(train_data.shape,train_label.shape,test_data.shape,test_label.shape)

	advancedmodel = SGDClassifier(loss='modified_huber', n_iter=5, random_state=0, shuffle=True)
	advancedmodel = advancedmodel.fit(train_data, train_label)

	preds = advancedmodel.predict(test_data)
	acc=accuracy_score(test_label, preds)
	print 'Stochastic Gradient Descent 2 accuracy:{}'.format(acc)
	scores_in_iter.append(acc)
	####################################################################################END

	####################################################################################START
	# Naive Bayes SVM 1
	advancedvectorizer = TfidfVectorizer(min_df=0.1, max_df=0.8, max_features = 200000, ngram_range = (1, 1))
	newdata = advancedvectorizer.fit_transform(headlines)

	train_data = newdata[:-nb_validation_samples]
	train_label = labels[:-nb_validation_samples]
	test_data = newdata[-nb_validation_samples:]
	test_label = labels[-nb_validation_samples:]
	# print '>>>>>>> train_data:{}, train_label:{}, test_data:{}, test_label:{}'.format(train_data.shape,train_label.shape,test_data.shape,test_label.shape)

	advancedmodel = NBSVM(C=0.01)
	advancedmodel = advancedmodel.fit(train_data, train_label)

	preds = advancedmodel.predict(test_data)
	acc=accuracy_score(test_label, preds)
	print 'Naive Bayes SVM 1 accuracy:{}'.format(acc)
	scores_in_iter.append(acc)
	####################################################################################END

	####################################################################################START
	# Naive Bayes SVM 2
	advancedvectorizer = TfidfVectorizer(min_df=0.031, max_df=0.2, max_features = 200000, ngram_range = (2, 2))
	newdata = advancedvectorizer.fit_transform(headlines)

	train_data = newdata[:-nb_validation_samples]
	train_label = labels[:-nb_validation_samples]
	test_data = newdata[-nb_validation_samples:]
	test_label = labels[-nb_validation_samples:]
	# print '>>>>>>> train_data:{}, train_label:{}, test_data:{}, test_label:{}'.format(train_data.shape,train_label.shape,test_data.shape,test_label.shape)

	advancedmodel = NBSVM(C=0.01)
	advancedmodel = advancedmodel.fit(train_data, train_label)

	preds = advancedmodel.predict(test_data)
	acc=accuracy_score(test_label, preds)
	print 'Naive Bayes SVM 2 accuracy:{}'.format(acc)
	scores_in_iter.append(acc)
	####################################################################################END

	####################################################################################START
	####################################################################################END

	scores.append(scores_in_iter)
	print '=======================================================\n'

print np.mean(scores, axis=0)
print np.std(scores, axis=0)