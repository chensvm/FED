Sentence classification using nltk SentimentAnalyzer, WordTokenizer, and pos_tagger.  
Determines the tense and the polarity of the sentence using nltk libraries.  

The function determine_input() takes in a single sentence as input, and outputs a dictionary:  
dict['tense'] = float ranging from -1.0 to 1.0 (-1.0:past, 1.0:future)  
dict['polarity'] = float ranging from -1.0 to 1.0 (-1.0:negative, 1.0:positive)  

See test.py for example.  

baseline1.py usage:
> python3 baseline1.py <year_start> <month_start> <day_start> <year_FED> <month_FED> <day_FED> <year_end> <month_end> <day_end>  
output: prediction with future tense

requirements: python 3.0, nltk  

* nltk_classify.py must be in same folder  