# http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

import numpy as np
import sys
from datetime import date, timedelta,datetime
import os.path

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier

## get articles from HP
# article_path = '/tmp3/finance/filtered_articles_remove_past/nytimes/'
article_path = '/tmp3/finance/filtered_articles/nytimes/'
rates_path = '../fed_rates/'
rates_file = rates_path + 'fed_date_rate_training.csv'
testing_rates_file = rates_path + 'fed_date_rate_testing.csv'
# rates_file = rates_path + 'fed_date_rate_testing.csv'
# output_path = 'classified_future_training_data/'

## getting meeting date and rate
rates = []
with open(rates_file, 'r') as fin:
	for ind,row in enumerate(fin):
		if ind > 0:
			r = row.split(',')
			# rates changed to 0,1,2 for fall,same,rise
			rates.append([r[0], int(r[1])+1])
## testing
testing_rates = []
with open(testing_rates_file, 'r') as fin:
	for ind,row in enumerate(fin):
		if ind > 0:
			r = row.split(',')
			# rates changed to 0,1,2 for fall,same,rise
			testing_rates.append([r[0], int(r[1])+1])

## constructing array with format: [previous_meeting_date, current_meeting_date, rate]
prev_date = '';
start_end_rate = [];
for date in rates:
	t = []
	t.append(prev_date)
	t.append(date[0])
	t.append(date[1])
	start_end_rate.append(t)
	prev_date = date[0]
del start_end_rate[0]
## testing
prev_date = '';
testing_start_end_rate = [];
for date in testing_rates:
	t = []
	t.append(prev_date)
	t.append(date[0])
	t.append(date[1])
	testing_start_end_rate.append(t)
	prev_date = date[0]
del testing_start_end_rate[0]

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
## testing
testing_article_file = []
for ser in testing_start_end_rate:
	start_date = datetime.strptime(ser[0],'%Y-%m-%d')
	start_date = start_date + timedelta(days=1)
	end_date = datetime.strptime(ser[1],'%Y-%m-%d')
	t = []
	count_date = start_date
	while count_date < end_date:
		t.append(str(count_date.year)+'/'+datetime.strftime(count_date,'%Y%m%d')+'.npy')
		count_date = count_date + timedelta(days=1)
	testing_article_file.append(t)

## split all articles into sentences?
article_to_sentences = False

## getting all articles and labels
data = []
target = []
for ind,ser in enumerate(start_end_rate):
	print('> meeting {} of {}: {} ~ {}'.format(ind,len(start_end_rate),ser[0],ser[1]))
	date_of_meeting = datetime.strptime(ser[1],'%Y-%m-%d')
	date_of_meeting = datetime.strftime(date_of_meeting,'%Y%m%d')
	for f in article_file[ind]:
		#print('>> getting articles: {}'.format(f))
		if os.path.isfile(article_path+f):
			day = np.load(article_path+f)
			if article_to_sentences:
				for article in day:
					for sentence in article.split('\n'):
						if len(sentence) > 1:
							data.append(sentence)
							target.append(ser[2])
			else:
				for article in day:
					data.append(article)
					target.append(ser[2])
## testing
testing_data = []
testing_target = []
for ind,ser in enumerate(testing_start_end_rate):
	print('> meeting {} of {}: {} ~ {}'.format(ind,len(testing_start_end_rate),ser[0],ser[1]))
	date_of_meeting = datetime.strptime(ser[1],'%Y-%m-%d')
	date_of_meeting = datetime.strftime(date_of_meeting,'%Y%m%d')
	for f in testing_article_file[ind]:
		#print('>> getting articles: {}'.format(f))
		if os.path.isfile(article_path+f):
			day = np.load(article_path+f)
			if article_to_sentences:
				for article in day:
					for sentence in article.split('\n'):
						if len(sentence) > 1:
							testing_data.append(sentence)
							testing_target.append(ser[2])
			else:
				for article in day:
					testing_data.append(article)
					testing_target.append(ser[2])

print('')
print('===============================')
print('len(data)')
print(str(len(data)))
print('len(target)')
print(str(len(target)))
print('len(testing_data)')
print(str(len(testing_data)))
print('len(testing_target)')
print(str(len(testing_target)))
print('===============================')
print('')

for iter_var in [2000]: #[10,50,100,200,500]:

	text_clf = Pipeline([('vect', CountVectorizer()),
	                     ('tfidf', TfidfTransformer()),
	                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
	                                           alpha=1e-3, random_state=42,
	                                           max_iter=iter_var, tol=None)),
	])

	text_clf.fit(data, target)  

	predicted = text_clf.predict(data)
	results = np.mean(predicted == target)


	print('===============================')
	print('{} iterations'.format(iter_var))
	# print('')

	print('training accuracy:')
	print(str(results))

	testing_predicted = text_clf.predict(testing_data)
	testing_results = np.mean(testing_predicted == testing_target)

	print('testing accuracy:')
	print(str(testing_results))
	print('===============================')
	print('')
