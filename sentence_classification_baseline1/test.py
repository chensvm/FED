# must have nltk_classify.py in same folder
from nltk_classify import determine_input

# import sentences
sentences = ["VADER is smart, handsome, and funny.", 
			"VADER is bad, short, and ugly.",
			"PETER went to the market and bought some wine yesterday.",
			"PETER will go to the market and buy some wine tomorrow."]

# download if first time using
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')

for sentence in sentences:
	# sentence as input of function
	out = determine_input(sentence)
	print(str('sentence:{}').format(sentence))
	# outputs out['tense'] and out['popularity']
	print(str('tense:{}, polarity:{}\n').format(out['tense'], out['polarity']))