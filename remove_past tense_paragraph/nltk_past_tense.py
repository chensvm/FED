import nltk
import gensim
import os
import collections
import smart_open
import random
import copy
import numpy as np

from datetime import date, timedelta
#nltk.download()
#punkt
def split_paragraph(n, i):
    if(i <= 0):
        pass
        #print('second n')
        #print(n)
    news_paragraph = []
    change_line_switch = 0
    str = ''
    for k in range(len(n)):
        word = n[k]
        if(word == '\n' and change_line_switch == 0):
            temp = []
            temp = str
            news_paragraph.append(temp)
            change_line_switch = 1
            str = ''
        else:
            str = str + word
            change_line_switch = 0
    return news_paragraph

def score_past_tense(news_paragraph, i, test_index, test_para_index):# first send news_paragraph in here, then do gensim.utils.simple_preprocess(news_paragraph[k])
    irre_verb_array = np.load('./irre_past_verb.npy')
    past_tense_score = []  
    past_tense_score2 = []
    past_score_index = []
    ed = []
    news_paragraph_2 = []
    #len([phrase for phrase in nltk.Chunker(sentence) if phrase[1] == 'VP'])
    for para in range(len(news_paragraph)):
        #print('news_paragraph[para]')
        #print(news_paragraph[para])
        #print('sent_tokenize')
        sentences = nltk.sent_tokenize(news_paragraph[para])
        words = []
        #print('word_tokenize')
        for sentence in sentences:
            words.extend(nltk.word_tokenize(sentence))
        #print('pos_tag')
        tagged_words = nltk.pos_tag(words)
        #print('temp_score')
        temp_score = len([phrase for phrase in tagged_words if phrase[1] == 'VBD'])
        if(i == test_index and para == test_para_index):
            for tag_index in range(len(tagged_words)):
                phrase = tagged_words[tag_index]
                #if(phrase[1] == 'VBD'):
                    #print('VBD VBD VBD VBD VBD')
                    #print(phrase[0])
        past_tense_score.append(temp_score)
    length = len(past_tense_score)
    for i in range(length):
        max_index = past_tense_score.index(max(past_tense_score))
        #print('max(past_tense_score)')
        #print(max(past_tense_score))
        past_tense_score2.append(max(past_tense_score))
        past_score_index.append(max_index)
        past_tense_score[max_index] = -100
    return past_score_index, past_tense_score2
def drop_out(news_paragraph, past_score_index, past_tense_score, ucl):
    news_paragraph_2 = copy.deepcopy(news_paragraph)
    for i in range(len(news_paragraph_2)):
        if(past_tense_score[i] == 0):
            break
        if(past_tense_score[i] >= ucl):
            news_paragraph_2[past_score_index[i]] = ''
        else:
            break 
    return news_paragraph_2
"""inputfilenameprefix = '/tmp2/finance/'"""
"""
inputfilenameprefix = '/tmp2/finance/' 
df = date(2012,1,1)  # period of 1 month
df_month = df.month
# df = date(2012,1,31)
"""
inputfilenameprefix = '/tmp2/finance/nytimes/2006/' #change years
outputfilenameprefix = '/tmp2/article/delpast/2006/' #change years
df = date(2006,1,1)  # period of 1 month #change years
df_month = df.month
df_year = df.year

datestr = []
while True:
    if df_year != df.year:
        break
    datestr.append(df.strftime('%Y%m%d'))
    df_year = df.year
    #df_month = df.month
    df = df + timedelta(days=1)
filenamestr = []
for ds in datestr:
    filenamestr.append(inputfilenameprefix+ds+'.npy')

news = []
file_exist = []
file_noexist = []
fns_index = 0
for fns in filenamestr:
    try:
        news.append(np.load(fns))
        file_exist.append(datestr[fns_index])
        print(fns)
        print(datestr[fns_index])
        print('\n')
        fns_index = fns_index + 1
    except:
        print(fns)
        file_noexist.append(datestr[fns_index])
        print(fns)
        print(datestr[fns_index])
        print('\n')
        fns_index = fns_index + 1
#test = np.array(test)
#test = np.load('./past_verb.npy')
#print(test)
print('file_exist')
print(file_exist)
print('file_noexist')
print(file_noexist)


train_corpus = []

i = 0
file_index = 0 # storage file's index according to day
test_index = 0#watch news
test_para_index = 10 # watch news paragraph
magnification = 2 # magnification of standard deviation
latest_news = []
print('news length')
print(len(news))
for day in news:
    latest_day = []
    for n in day:
        #split here
        if(i == test_index):
            pass
            #print('n')
            #print(n)
            #print('simple_preprocess')
            #print(gensim.utils.simple_preprocess(n))
        news_paragraph = []
        past_tense_score = []
        past_score_index = []
        #split airticle into paragraph 
        news_paragraph = split_paragraph(n, i)

        #put the paragraph together test and put back to n
        whole_news = ''
        for para_index in range(len(news_paragraph)):
            whole_news = whole_news + news_paragraph[para_index]
        #test

        #score the past tense score to decide which paragraph has higher probability to be critized past
        #index
        if(i == test_index):
            print('news_paragraph[' + str(test_para_index) + ']')
            print(news_paragraph[test_para_index])
            print('i ', i)
            print('test_index', test_index)
            print('test_para_index', test_para_index)
        past_score_index,  past_tense_score = score_past_tense(news_paragraph, i, test_index, test_para_index)
        if(i == test_index):
            print('past_score_index')
            print(past_score_index)
            print('past_tense_score')
            print(past_tense_score)
        mean = np.mean(past_tense_score)
        std = np.std(past_tense_score)
        #drop out paragragh that scores beyond mean +- 2* standard deviation
        ucl = mean + magnification * std
        if(i == test_index):
            print('ucl')
            print(ucl)
        last_news_paragraph = drop_out(news_paragraph,past_score_index, past_tense_score, ucl)

        whole_news = ''
        for para_index in range(len(last_news_paragraph)):
            whole_news = whole_news + last_news_paragraph[para_index]
            whole_news = whole_news + '\n'
        latest_day.append(whole_news)
        """
        for k in range(len(news_paragraph)):
            train_corpus.append(gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(news_paragraph[k]), [i]))
            i = i + 1
        """
        i = i+1
    latest_day = np.array(latest_day)
    np.save(outputfilenameprefix + file_exist[file_index], latest_day)
    file_index = file_index + 1
    #write file here just as the same file I load ex: 20120128 
    latest_news.append(latest_day)
latest_news = np.array(latest_news)

#np.save('./2002_nopast', latest_news)
#print("save")
#test = np.load('./2002_nopast.npy')
#print('test length')
#print(len(test))
#print('test para')
#print(test[0][test_index])

"""
np.save('./latest_news', latest_news)
print("save")
test = np.load('./latest_news.npy')
print(latest_news[0][test_index])
"""
#print('train_corpus[1]')
#print(train_corpus[1])
"""
train_corpus = []
i = 0
for day in news:
    for n in day:
        train_corpus.append(gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(n), [i]))
        i = i + 1
#print(train_corpus[i-1])
"""
"""
print(train_corpus[0])
print('train_corpus[0][0]')
print(train_corpus[0][0])
print('count the')
print(train_corpus[0][0].count('the'))
"""

"""
irre_verb_array = np.load('./irre_past_verb.npy')
print(irre_verb_array)
print('count')
print(list(irre_verb_array).count('stung'))
past_tense_score = []
past_tense_score2 = []
past_score_index = []
ed = []
train_corpus_2 = copy.deepcopy(train_corpus)
length = len(train_corpus_2) # 1
for news in range(length):
    temp_score = 0
    ed_temp = []
    for irre in range(len(irre_verb_array)): 
        temp_score = temp_score + list(train_corpus_2[news][0]).count(irre_verb_array[irre])
    for word_index in range(len(train_corpus_2[news][0])):
        temp_new = train_corpus_2[news][0]
        word = temp_new[word_index]
        if(word[-2:] == 'ed'):
            print(word)
            ed_temp.append(word)
            temp_score = temp_score + 1
    ed.append(ed_temp)
    past_tense_score.append(temp_score)

    
for i in range(length):
    max_index = past_tense_score.index(max(past_tense_score))
    #print('max(past_tense_score)')
    #print(max(past_tense_score))
    past_tense_score2.append(max(past_tense_score))
    past_score_index.append(max_index)
    past_tense_score[max_index] = -100

for i in range(1):
    print('i')
    print(i)
    print(train_corpus[past_score_index[i]])

#test!!!!!!!!!!
past_test = []
test_index = 300
max_index = past_score_index[test_index]
for i in range(len(irre_verb_array)):
    if(irre_verb_array[i] in list(train_corpus[max_index][0])):
        past_test.append(irre_verb_array[i])
past_test = past_test + ed[max_index]
#past_test.append(ed_temp)
print('train_corpus')
print(print(train_corpus[past_score_index[test_index]]))
#print('past_score_index')
#print(past_score_index)
#print('past_tense_score2')
#print(past_tense_score2)
print('past_tense_score2')
print(past_tense_score2[test_index])
print('past_test length')
print(len(past_test))
print('past_test')
print(past_test)
"""