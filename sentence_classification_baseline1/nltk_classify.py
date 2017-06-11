import nltk
from nltk import word_tokenize, pos_tag
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np

from datetime import date, timedelta

# nltk.download('vader_lexicon')
# nltk.download('averaged_perceptron_tagger')

# returns dictionary with 2 values: 'tense', 'polarity'
def determine_input(sentence):
    out = {}
    # determine past or future (-1:past, +1:future)
    text = word_tokenize(sentence)
    tagged = pos_tag(text)
    future = len([word for word in tagged if word[1] == "MD"])
    present = len([word for word in tagged if word[1] in ["VBP", "VBZ","VBG"]])
    past = len([word for word in tagged if word[1] in ["VBD", "VBN"]]) 
    tsum = future + present + past
    #TODO: tense function (baseline)
    tense = (past/tsum)*-1.0 + (present/tsum)*0.25 + (future/tsum)*1.0
    # NewValue = (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
    out['tense'] = (((tense + 1.0) * (1.0 + 1.0)) / (1.25 + 1.0)) - 1.0

    # determine positive or negative (-1:neg, +1:pos)
    sid = SentimentIntensityAnalyzer()
    pol = sid.polarity_scores(sentence)
    #TODO: polarity function (baseline)
    out['polarity'] = pol['pos'] - pol['neg']

    return(out)