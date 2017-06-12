import numpy as np
import sys
from datetime import date, timedelta
from nltk_classify import determine_input

getyear = int(sys.argv[1])
getmonth = int(sys.argv[2])
getday = int(sys.argv[3])
getyear_end = int(sys.argv[4])
getmonth_end = int(sys.argv[5])
getday_end = int(sys.argv[6])
df = date(getyear,getmonth,getday)

datestr = []
while True:
	if df.year==getyear_end and df.month==getmonth_end and df.day==getday_end:
		datestr.append(df.strftime('%Y%m%d'))	
		break
	datestr.append(df.strftime('%Y%m%d'))
	df = df + timedelta(days=1)

file_path = '/tmp2/finance/nytimes/'
filenamestr = []
for d in datestr:
	filenamestr.append(file_path+str(d[:4])+str('/')+d+str('.npy'))

news = []
for fns in filenamestr:
	news.append(np.load(fns))

# TODO: 
p1 = 0.0
p2 = 0.0
for day in news:
	for article in day:
		out = determine_input(article)
		# print(str('sentence:{}').format(article))
		# print('============================================================')
		print(str('tense:{}, polarity:{}\n').format(out['tense'], out['polarity']))
		if out['tense'] > 0:
			p2 += out['polarity']

print('final polarity:'+str(p2))





