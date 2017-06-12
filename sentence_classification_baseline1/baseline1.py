import numpy as np
import sys
from datetime import date, timedelta
from nltk_classify import determine_input

getyear_start = int(sys.argv[1])
getmonth_start = int(sys.argv[2])
getday_start = int(sys.argv[3])

getyear_mid = int(sys.argv[4])
getmonth_mid = int(sys.argv[5])
getday_mid = int(sys.argv[6])

getyear_end = int(sys.argv[7])
getmonth_end = int(sys.argv[8])
getday_end = int(sys.argv[9])

df = date(getyear_start,getmonth_start,getday_start)

## getting all dates
datestr_for_future = []
datestr_for_past = []
# get dates before FED
while True:
	if df.year==getyear_mid and df.month==getmonth_mid and df.day==getday_mid:
		datestr_for_future.append(df.strftime('%Y%m%d'))	
		break
	datestr_for_future.append(df.strftime('%Y%m%d'))
	df = df + timedelta(days=1)
# get dates after FED
while True:
	if df.year==getyear_end and df.month==getmonth_end and df.day==getday_end:
		datestr_for_past.append(df.strftime('%Y%m%d'))	
		break
	datestr_for_past.append(df.strftime('%Y%m%d'))
	df = df + timedelta(days=1)

## constructing file path from dates
file_path = '/tmp2/finance/nytimes/'
filenamestr_for_future = []
filenamestr_for_past = []
# path before FED
for d in datestr_for_future:
	filenamestr_for_future.append(file_path+str(d[:4])+str('/')+d+str('.npy'))
# path after FED
for d in datestr_for_past:
	filenamestr_for_past.append(file_path+str(d[:4])+str('/')+d+str('.npy'))	

## loading news
news_for_future = []
news_for_past = []
# loading news before FED
for fns in filenamestr_for_future:
	news_for_future.append(np.load(fns))
# loading news after FED
for fns in filenamestr_for_past:
	news_for_past.append(np.load(fns))

# threshold for past or future classification
tense_threshold = 0

# getting polarity for news before FED
p1 = 0.0
for day in news_for_future:
	for article in day:
		for sentence in article.split('\n'):
			if len(sentence) > 1:
				out = determine_input(sentence)
				print(str('sentence:{}').format(sentence))
				# print('============================================================')
				print(str('<before FED> tense:{}, polarity:{}').format(out['tense'], out['polarity']))
				if out['tense'] > tense_threshold:
					p1 += out['polarity']

# getting polarity for news after FED
p2 = 0.0
for day in news_for_past:
	for article in day:
		for sentence in article.split('\n'):
			if len(sentence) > 1:
				out = determine_input(sentence)
				print(str('sentence:{}').format(sentence))
				# print('============================================================')
				print(str('<after FED> tense:{}, polarity:{}').format(out['tense'], out['polarity']))
				if out['tense'] < -tense_threshold:
					p2 += out['polarity']

# threshold for polarity classification
polarity_threshold = 0.05
print(str('p1:{}, p2:{}').format(p1,p2))
p = p1 # + p2
if p > polarity_threshold:
	# rise
	pass
elif p < -polarity_threshold:
	# fall
	pass
else:
	# neutral
	pass




