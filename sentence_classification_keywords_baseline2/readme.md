# Sentence_Classification_Keywords_Baseline2
### Determines the tense and the polarity of the sentence using nltk SentimentAnalyzer, WordTokenizer, and pos_tagger.  

### The function determine_input() takes in a single sentence as input, and outputs a dictionary:
### But only run for sentences which have keywords  
dict['tense'] = float ranging from -1.0 to 1.0 (-1.0:past, 1.0:future)  
dict['polarity'] = float ranging from -1.0 to 1.0 (-1.0:negative, 1.0:positive)  

### keywords in us_interest_keyword.txt

### baseline1.py usage:
> python3 baseline1.py <year_start> <month_start> <day_start> <year_FED> <month_FED> <day_FED> <year_end> <month_end> <day_end>

requirements: python 3.0, nltk  

* nltk_classify.py must be in same folder  