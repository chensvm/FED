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
print('for future lost data')
for fns in filenamestr_for_future:
    try:
        news_for_future.append(np.load(fns))
    except:
        print(fns)
# loading news after FED
print('for past lost data')
for fns in filenamestr_for_past:
    try:
        news_for_past.append(np.load(fns))
    except:
        print(fns)

# getting keywords which can affect US interest rate
keywords = []
text_file = open("us_interest_keyword.txt", "r")
keywords = text_file.read().split('\n')
text_file.close()
keywords_freq = [0] * len(keywords)

# threshold for past or future classification
tense_threshold = 0

# getting polarity for news before FED
p1 = 0.0
key = 0
out = {}
for day in news_for_future:
	for article in day:
		for sentence in article.split('\n'):
			key = 0
			if len(sentence) > 1:
			    temp_test = 0
			    for i in range(len(keywords)):
			        if keywords[i] in sentence:
			            out = determine_input(sentence)
			            keywords_freq[i] = keywords_freq[i] + 1
			            key = 1
			            temp_test = i
			            break
			    if(key == 1):
			        print(str('sentence:{}').format(sentence))
			        print(keywords[temp_test])
			        print(str('out[polarity]:{}').format(out['polarity']))
			    if(key == 0):
			        out['tense'] = 0.0
			        out['polarity'] = 0.0
				# print('============================================================')
			    #print(str('<before FED> tense:{}, polarity:{}').format(out['tense'], out['polarity']))
			    if out['tense'] > tense_threshold:
			        p1 += out['polarity']

# getting polarity for news after FED
p2 = 0.0
for day in news_for_past:
	for article in day:
		for sentence in article.split('\n'):
			key = 0
			if len(sentence) > 1:
			    temp_test = 0
			    for i in range(len(keywords)):
			        if keywords[i] in sentence:
			            out = determine_input(sentence)
			            keywords_freq[i] = keywords_freq[i] + 1
			            key = 1
			            temp_test = i
			            break
			    if(key == 1):
			        print(str('sentence:{}').format(sentence))
			        print(keywords[temp_test])
			        print(str('out[polarity]:{}').format(out['polarity']))
			    if(key == 0):
			        out['tense'] = 0.0
			        out['polarity'] = 0.0
				#print(str('sentence:{}').format(sentence))
				# print('============================================================')
				#print(str('<after FED> tense:{}, polarity:{}').format(out['tense'], out['polarity']))
			    if out['tense'] < -tense_threshold:
			        p2 += out['polarity']

# threshold for polarity classification
polarity_threshold = 0.05
print(str('p1:{}, p2:{}').format(p1,p2))
p = p1 # + p2
if p > polarity_threshold:
	# rise
    print('rise')
elif p < -polarity_threshold:
	# fall
	print('fall')
else:
	# neutral
	print('neutral')

for k in range(len(keywords)):
    print(str('keywords:{}, keywords_freq:{}').format(keywords[k], keywords_freq[k]))




