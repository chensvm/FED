## Sentence Classification Baseline with nltk  
Determines the tense and the polarity of the sentence using nltk SentimentAnalyzer, WordTokenizer, and pos_tagger.  

---

method:  
1.Get all news articles in between two meeting dates.  
2.Split articles into sentences.  
3.Remove articles and sentences with past reference.  
4.Classify articles and sentences into positive and negative polarity.  
5.Use articles OR sentences to predict rate.
If pos - neg > threshold, predict rise.  
If neg - pos > threshold, predict fall.  
Else, predict maintain.  

> accuracy on testing data:
> 0.189 (11/58) using classified sentences, threshold:50
> 0.448 (26/58) using classified articles, threshold:2  
> 0.379 (22/58) using classified articles, threshold:1   

> accuracy on training data:
> 0.352 (12/34) using classified sentences, threshold:50
> 0.264 (9/34) using classified articles, threshold:2
> 0.352 (12/34) using classified articles, threshold:1  

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
