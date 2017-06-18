import numpy as np
import sys
from datetime import date, timedelta,datetime
from nltk_classify import determine_input
import os.path

## get articles from HP
article_path = '/tmp2/finance/nytimes/'
rates_path = '../fed_rates/'
rates_file = rates_path + 'fed_date_rate_testing.csv'
# rates_file = rates_path + 'fed_date_rate_testing.csv'
output_path = 'classified_future_testing_data/'

## getting meeting date and rate
rates = {}
with open(rates_file, 'r') as fin:
	for ind,row in enumerate(fin):
		if ind > 0:
			r = row.split(',')
			rates[r[0]] = int(r[1])

## constructing array with format: [previous_meeting_date, current_meeting_date, rate]
prev_date = '';
start_end_rate = [];
for date in rates:
	t = []
	t.append(prev_date)
	t.append(date)
	t.append(rates[date])
	start_end_rate.append(t)
	prev_date = date
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

## getting all sentences for meeting_date
## data_for_date[date of meeting] = [array of sentences]
data_for_date = {}
for ind,ser in enumerate(start_end_rate):
	date_of_meeting = datetime.strptime(ser[1],'%Y-%m-%d')
	date_of_meeting = datetime.strftime(date_of_meeting,'%Y%m%d')
	data_for_date[date_of_meeting] = []
	for f in article_file[ind]:
		if os.path.isfile(article_path+f):
			day = np.load(article_path+f)
			for article in day:
				for sentence in article.split('\n'):
					if len(sentence) > 1:
						data_for_date[date_of_meeting].append(sentence)

## getting positive and negative sentences for each day and saving
tense_threshold = 0
polarity_threshold = 0.05
for d in data_for_date:
	## getting sentences
	positive_sentences = []
	negative_sentences = []
	for ind,sentence in enumerate(data_for_date[d]):
		out = determine_input(sentence)
		if out['tense'] > tense_threshold:
			if out['polarity'] > polarity_threshold:
				positive_sentences.append(sentence)
			elif out['polarity'] < -polarity_threshold:
				negative_sentences.append(sentence)
		# print(str('> doing {} of {} in date({})').format(ind,len(data_for_date[d]),d))
	## saving
	print(str('>> ({}) pos:{}, neg{}').format(d,len(positive_sentences),len(negative_sentences)))
	pos_outputfilename = output_path + 'positive_' + d + '.npy'
	neg_outputfilename = output_path + 'negative_' + d + '.npy'
	np.save(pos_outputfilename, np.array(positive_sentences))
	np.save(neg_outputfilename, np.array(negative_sentences))

