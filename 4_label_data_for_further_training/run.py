import numpy as np
import sys
from datetime import date, timedelta,datetime
from nltk_classify import determine_input

# getyear_start = int(sys.argv[1])
# getmonth_start = int(sys.argv[2])
# getday_start = int(sys.argv[3])

# getyear_mid = int(sys.argv[4])
# getmonth_mid = int(sys.argv[5])
# getday_mid = int(sys.argv[6])

# getyear_end = int(sys.argv[7])
# getmonth_end = int(sys.argv[8])
# getday_end = int(sys.argv[9])

# df = date(getyear_start,getmonth_start,getday_start)

article_path = '../training_articles/'
rates_path = '../fed_rates/'
rates_file = rates_path + 'fed_date_rate_training.csv'
# rates_file = rates_path + 'fed_date_rate_testing.csv'

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
		t.append(datetime.strftime(count_date,'%Y%m%d')+'.npy')
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
		day = np.load(article_path+f)
		for article in day:
			for sentence in article.split('\n'):
				if len(sentence) > 1:
					data_for_date[date_of_meeting].append(sentence)

for d in data_for_date:
	print(d)
	print(len(data_for_date[d]))
	break









# ## getting all dates
# datestr_for_future = []
# datestr_for_past = []
# # get dates before FED
# while True:
# 	if df.year==getyear_mid and df.month==getmonth_mid and df.day==getday_mid:
# 		datestr_for_future.append(df.strftime('%Y%m%d'))	
# 		break
# 	datestr_for_future.append(df.strftime('%Y%m%d'))
# 	df = df + timedelta(days=1)
# # get dates after FED
# while True:
# 	if df.year==getyear_end and df.month==getmonth_end and df.day==getday_end:
# 		datestr_for_past.append(df.strftime('%Y%m%d'))	
# 		break
# 	datestr_for_past.append(df.strftime('%Y%m%d'))
# 	df = df + timedelta(days=1)

# ## constructing file path from dates
# file_path = '/tmp2/finance/nytimes/'
# filenamestr_for_future = []
# filenamestr_for_past = []
# # path before FED
# for d in datestr_for_future:
# 	filenamestr_for_future.append(file_path+str(d[:4])+str('/')+d+str('.npy'))
# # path after FED
# for d in datestr_for_past:
# 	filenamestr_for_past.append(file_path+str(d[:4])+str('/')+d+str('.npy'))	

# ## loading news
# news_for_future = []
# news_for_past = []
# # loading news before FED
# for fns in filenamestr_for_future:
# 	news_for_future.append(np.load(fns))
# # loading news after FED
# for fns in filenamestr_for_past:
# 	news_for_past.append(np.load(fns))

# # threshold for past or future classification
# tense_threshold = 0

# # getting polarity for news before FED
# p1 = 0.0
# for day in news_for_future:
# 	for article in day:
# 		for sentence in article.split('\n'):
# 			if len(sentence) > 1:
# 				out = determine_input(sentence)
# 				print(str('sentence:{}').format(sentence))
# 				# print('============================================================')
# 				print(str('<before FED> tense:{}, polarity:{}').format(out['tense'], out['polarity']))
# 				if out['tense'] > tense_threshold:
# 					p1 += out['polarity']

# # getting polarity for news after FED
# p2 = 0.0
# for day in news_for_past:
# 	for article in day:
# 		for sentence in article.split('\n'):
# 			if len(sentence) > 1:
# 				out = determine_input(sentence)
# 				print(str('sentence:{}').format(sentence))
# 				# print('============================================================')
# 				print(str('<after FED> tense:{}, polarity:{}').format(out['tense'], out['polarity']))
# 				if out['tense'] < -tense_threshold:
# 					p2 += out['polarity']

# # threshold for polarity classification
# polarity_threshold = 0.05
# print(str('p1:{}, p2:{}').format(p1,p2))
# p = p1 # + p2
# if p > polarity_threshold:
# 	# rise
# 	pass
# elif p < -polarity_threshold:
# 	# fall
# 	pass
# else:
# 	# neutral
# 	pass




