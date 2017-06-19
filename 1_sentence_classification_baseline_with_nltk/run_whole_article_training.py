import numpy as np
import sys
from datetime import date, timedelta,datetime
from nltk_classify import determine_input
import os.path

## get articles from HP
article_path = '/tmp2/finance/nytimes/'
rates_path = '../fed_rates/'
rates_file = rates_path + 'fed_date_rate_training.csv'
outputfilename = 'training_predictions_whole_article.txt'

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

## getting all sentences and rate label for meeting_date
## data_for_date[date of meeting] = [array of sentences]
data_for_date = {}
rate_for_date = {}
for ind,ser in enumerate(start_end_rate):
	date_of_meeting = datetime.strptime(ser[1],'%Y-%m-%d')
	date_of_meeting = datetime.strftime(date_of_meeting,'%Y%m%d')
	data_for_date[date_of_meeting] = []
	rate_for_date[date_of_meeting] = ser[2]
	for f in article_file[ind]:
		if os.path.isfile(article_path+f):
			day = np.load(article_path+f)
			for article in day:
				data_for_date[date_of_meeting].append(article)
				# for sentence in article.split('\n'):
				# 	if len(sentence) > 1:
				# 		data_for_date[date_of_meeting].append(sentence)

## getting positive and negative sentences for each day and predict
tense_threshold = 0.
polarity_threshold = 0.05
rate_threshold = 2
corrects = 0
with open(outputfilename,'w') as fout:
	for d in data_for_date:
		## getting sentences
		number_of_positive_sentences = 0
		number_of_negative_sentences = 0
		for ind,sentence in enumerate(data_for_date[d]):
			out = determine_input(sentence)
			if out['tense'] > tense_threshold:
				if out['polarity'] > polarity_threshold:
					number_of_positive_sentences += 1
				elif out['polarity'] < -polarity_threshold:
					number_of_negative_sentences += 1
			# print(str('> doing {} of {} in date({})').format(ind,len(data_for_date[d]),d))
		## prediction
		pred = 0
		if number_of_positive_sentences - number_of_negative_sentences > rate_threshold:
			pred = 1
		elif number_of_negative_sentences - number_of_positive_sentences > rate_threshold:
			pred = -1
		if pred == rate_for_date[d]:
			corrects += 1
		print(str('>> ({}) pred:{}, actual:{}, {}/{} ').format(d,pred,rate_for_date[d],number_of_positive_sentences,number_of_negative_sentences))
		fout.write(str('({}) pred:{}, actual:{}, {}/{}\n').format(d,pred,rate_for_date[d],number_of_positive_sentences,number_of_negative_sentences))
	print(str('>>> Accuracy:{}, {}/{}').format(corrects/len(data_for_date),corrects,len(data_for_date)))
	fout.write(str('> Accuracy:{}, {}/{}').format(corrects/len(data_for_date),corrects,len(data_for_date)))
	

