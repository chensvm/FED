## Sentence Classification Baseline with nltk  
Determines the tense and the polarity of the sentence using nltk SentimentAnalyzer, WordTokenizer, and pos_tagger.  

---

method:  
Get all sentences from news in between two meeting dates.  
Remove sentences with past reference.  
Classify senteces into positive and negative polarity.  
If pos - neg > threshold, predict rise.  
If neg - pos > threshold, predict fall.  
Else, predict maintain.  
> accuracy: 0.189 on testing data

---

### nltk_classify.py usage  
The function determine_input() takes in a single sentence as input, and outputs a dictionary:  
> dict['tense'] = float ranging from -1.0 to 1.0 (-1.0:past, 1.0:future)  
> dict['polarity'] = float ranging from -1.0 to 1.0 (-1.0:negative, 1.0:positive)  

See test.py for example.  

---

### baseline1.py usage:
> python3 baseline1.py <year_start> <month_start> <day_start> <year_FED> <month_FED> <day_FED> <year_end> <month_end> <day_end>

---

requirements: python 3.0, nltk  
nltk_classify.py must be in same folder  
