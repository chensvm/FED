#-*- coding: utf-8 -*-
import nltk
from nltk.stem.lancaster import LancasterStemmer
import json
import datetime
import numpy as np
import re
from datetime import timedelta, date
import csv
from nltk.corpus import stopwords

stemmer = LancasterStemmer()


def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

def loadDataSet():

    print "start loading training dataset"
    postingList = []
    words = []
    documents = []
    classVec = []

    posNum = 0
    negNum = 0
    neutralNum = 0

    with open('../fed_rates/fed_date_rate_training.csv', 'r') as c1:
        reader = csv.reader(c1)
        next(reader, None)  # skip the headers
        prevRow = "2004-11-01"
        for row in reader:

            if row[1] == "1":

                cur_year, cur_month, cur_day = str(row[0]).split("-")
                pre_year, pre_month, pre_day = prevRow.split("-")

                start_date = date(int(pre_year), int(pre_month), int(pre_day))
                end_date = date(int(cur_year), int(cur_month), int(cur_day))

                for single_date in daterange(start_date, end_date):


                    with open('../../../../tmp2/finance_data/filtered_articles/nytimes/' +str(single_date.strftime("%Y"))+"/"+ str(single_date.strftime("%Y%m%d")) + ".npy", 'r') as myfile:

                        print str(single_date.strftime("%Y-%m-%d"))
                        data = np.load(myfile)
                        if data.size == 0:
                            pass

                        else:
                            for news in data:
                                regEx = re.compile('\\W*')
                                listOfTokens = regEx.split(news)
                                listOfTokens = [tok.lower().encode('utf-8') for tok in listOfTokens if len(tok) > 0]
                                filtered_words = [word for word in listOfTokens if word not in stopwords.words('english')]

                                posNum += 1
                                #準備posting list
                                postingList.append(filtered_words)
                                words.extend(filtered_words)
                                # 同時建立一個分類向量搭配
                                classVec.append(1)
                                documents.append((filtered_words, 1))



                    myfile.close()

            elif row[1] == "-1":
                cur_year, cur_month, cur_day = str(row[0]).split("-")
                pre_year, pre_month, pre_day = prevRow.split("-")

                start_date = date(int(pre_year), int(pre_month), int(pre_day))
                end_date = date(int(cur_year), int(cur_month), int(cur_day))

                for single_date in daterange(start_date, end_date):

                    with open('../../../../tmp2/finance_data/filtered_articles/nytimes/' + str(single_date.strftime("%Y")) + "/" + str(
                            single_date.strftime("%Y%m%d")) + ".npy", 'r') as myfile:
                        print str(single_date.strftime("%Y-%m-%d"))

                        data = np.load(myfile)

                        if data.size == 0:
                            pass

                        else:

                            for news in data:

                                regEx = re.compile('\\W*')
                                listOfTokens = regEx.split(news)
                                listOfTokens = [tok.lower().encode('utf-8') for tok in listOfTokens if len(tok) > 0]
                                filtered_words = [word for word in listOfTokens if word not in stopwords.words('english')]

                                negNum += 1
                                postingList.append(filtered_words)
                                classVec.append(-1)

                                words.extend(filtered_words)
                                documents.append((filtered_words, -1))

                    myfile.close()

            elif row[1] == "0":
                cur_year, cur_month, cur_day = str(row[0]).split("-")
                pre_year, pre_month, pre_day = prevRow.split("-")

                start_date = date(int(pre_year), int(pre_month), int(pre_day))
                end_date = date(int(cur_year), int(cur_month), int(cur_day))

                for single_date in daterange(start_date, end_date):

                    with open('../../../../tmp2/finance_data/filtered_articles/nytimes/' + str(single_date.strftime("%Y")) + "/" + str(
                            single_date.strftime("%Y%m%d")) + ".npy", 'r') as myfile:

                        print str(single_date.strftime("%Y-%m-%d"))

                        data = np.load(myfile)

                        if data.size == 0:
                            pass

                        else:

                            for news in data:

                                regEx = re.compile('\\W*')
                                listOfTokens = regEx.split(news)
                                listOfTokens = [tok.lower().encode('utf-8') for tok in listOfTokens if len(tok) > 0]
                                filtered_words = [word for word in listOfTokens if word not in stopwords.words('english')]

                                neutralNum += 1
                                postingList.append(filtered_words)
                                classVec.append(0)
                                words.extend(filtered_words)
                                documents.append((filtered_words, 0))



                    myfile.close()

            else:
                print "pass posting list error: "+ row[0]
                pass

            prevRow = row[0]

    print "finish postingList"

    return postingList, classVec, words, documents



classes = [1, 0, -1]

postingList, classVec, words, documents = loadDataSet()
words = list(set(words))

print (len(documents), "documents")
print (len(classes), "classes", classes)
print (len(words), "unique stemmed words")

# create our training data
# Our training data is transformed into “bag of words”

training = []
output = []
# create an empty array for our output
output_empty = [0] * len(classes)

# training set, bag of words for each sentence
# documents : [list of words, class]
for doc in documents:
# initialize our bag of words
    bag = []
# list of tokenized words for the pattern
# 只看list of words部分
pattern_words = doc[0]
# stem each word
pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
# create our bag of words array
for w in words:
# 如果word中的任一個詞有出現過，就append 1
    bag.append(1) if w in pattern_words else bag.append(0)


training.append(bag)
# output is a '0' for each tag and '1' for current tag
output_row = list(output_empty)
output_row[classes.index(doc[1])] = 1
output.append(output_row)
print "output row: " + str(output_row)

# sample training/output
i = 0
w = documents[i][0]
print "finish bag of words"
# print ([stemmer.stem(word.lower()) for word in w])
print (output[i])


# compute sigmoid nonlinearity
# The Sigmoid function, which describes an S shaped curve.
# We pass the weighted sum of the inputs through this function to
# normalise them between 0 and 1.

def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return output


# convert output of sigmoid function to its derivative
# The derivative of the Sigmoid function.
# This is the gradient of the Sigmoid curve.
# It indicates how confident we are about the existing weight.
# Sigmoid derivative’ tells us about the slope of the curve on any point

def sigmoid_output_to_derivative(output):
    return output * (1 - output)


def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# bag-of-words
# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    print "finish bag of words"
    return (np.array(bag))

# dot-product calculation in our previously defined think() function
def think(sentence, show_details=False):
    print "begining of think function"
    x = bow(sentence.lower(), words, show_details)
    if show_details:

        print ("bow: ", x)
    # input layer is our bag of words
    l0 = x
    # matrix multiplication of input and hidden layer
    l1 = sigmoid(np.dot(l0, synapse_0))
    # output layer
    l2 = sigmoid(np.dot(l1, synapse_1))
    return l2

alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
# 0.001, 0.01 usually can not converge


# ANN and Gradient Descent code from https://iamtrask.github.io//2015/07/27/python-network-part2/
# Adjusting the synaptic weights each time.

def train(X, y, hidden_neurons=10, alpha=1, epochs=50000, dropout=False, dropout_percent=0.5):
    # self, training_set_inputs, training_set_outputs, number_of_training_iterations

    print "begin to train"

    print ("Training with %s neurons, alpha:%s, dropout:%s %s" % (
    hidden_neurons, str(alpha), dropout, dropout_percent if dropout else ''))
    print ("Input matrix: %sx%s    Output matrix: %sx%s" % (len(X), len(X[0]), 1, len(classes)))
    np.random.seed(1)
    # randomly generate a start point

    last_mean_error = 1
    # randomly initialize our weights with mean 0
    synapse_0 = 2 * np.random.random((len(X[0]), hidden_neurons)) - 1
    synapse_1 = 2 * np.random.random((hidden_neurons, len(classes))) - 1


    prev_synapse_0_weight_update = np.zeros_like(synapse_0)
    prev_synapse_1_weight_update = np.zeros_like(synapse_1)

    synapse_0_direction_count = np.zeros_like(synapse_0)
    synapse_1_direction_count = np.zeros_like(synapse_1)

    # we can train in different alpha

    for j in iter(range(epochs + 1)):# repeat untill slope ==0

        # Feed forward through layers 0, 1, and 2
        layer_0 = X
        layer_1 = sigmoid(np.dot(layer_0, synapse_0))
        #computing error

        if (dropout):
            layer_1 *= np.random.binomial([np.ones((len(X), hidden_neurons))], 1 - dropout_percent)[0] * (
            1.0 / (1 - dropout_percent))

        layer_2 = sigmoid(np.dot(layer_1, synapse_1))

        # how much did we miss the target value?
        # Calculate the error (The difference between the desired output
        # and the predicted output).
        layer_2_error = y - layer_2

        # Multiply the error by the input and again by the gradient of the Sigmoid curve.
        # This means less confident weights are adjusted more.
        # This means inputs, which are zero, do not cause changes to the weights.
        # adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))


        # Adjust the weights.
        if (j % 10000) == 0 and j > 5000:
            # if this 10k iteration's error is greater than the last iteration, break out
            if np.mean(np.abs(layer_2_error)) < last_mean_error:
                print ("delta after " + str(j) + " iterations:" + str(np.mean(np.abs(layer_2_error))))
                last_mean_error = np.mean(np.abs(layer_2_error))
            else:
                print ("break:", np.mean(np.abs(layer_2_error)), ">", last_mean_error)
                break

        # in what direction is the target value?
        # were we really sure? if so, don't change too much.
        #optimize to reduce error
        layer_2_delta = layer_2_error * sigmoid_output_to_derivative(layer_2)

        # how much did each l1 value contribute to the l2 error (according to the weights)?
        layer_1_error = layer_2_delta.dot(synapse_1.T)

        # in what direction is the target l1?
        # were we really sure? if so, don't change too much.
        layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)

        # calculate current slope at X position


        synapse_1_weight_update = (layer_1.T.dot(layer_2_delta))
        synapse_0_weight_update = (layer_0.T.dot(layer_1_delta))

        if (j > 0):
            synapse_0_direction_count += np.abs(
                ((synapse_0_weight_update > 0) + 0) - ((prev_synapse_0_weight_update > 0) + 0))
            synapse_1_direction_count += np.abs(
                ((synapse_1_weight_update > 0) + 0) - ((prev_synapse_1_weight_update > 0) + 0))

        # tune model with alpha
        synapse_1 += alpha * synapse_1_weight_update
        synapse_0 += alpha * synapse_0_weight_update

        prev_synapse_0_weight_update = synapse_0_weight_update
        prev_synapse_1_weight_update = synapse_1_weight_update

    now = datetime.datetime.now()

    # persist synapses
    synapse = {'synapse0': synapse_0.tolist(), 'synapse1': synapse_1.tolist(),
               'datetime': now.strftime("%Y-%m-%d %H:%M"),
               'words': words,
               'classes': classes
               }
    synapse_file = "synapses.json"

    with open(synapse_file, 'w') as outfile:
        json.dump(synapse, outfile, indent=4, sort_keys=True)
    print ("saved synapses to:", synapse_file)
# The synapse.json file contains all of our synaptic weights, this is our model.


X = np.array(training)
y = np.array(output)

# start_time = time.time()

train(X, y, hidden_neurons=20, alpha=0.1, epochs=100000, dropout=False, dropout_percent=0.2)

# elapsed_time = time.time() - start_time
# print ("processing time:", elapsed_time, "seconds")

# probability threshold
ERROR_THRESHOLD = 0.2
# load our calculated synapse values
synapse_file = 'synapses.json'
with open(synapse_file) as data_file:
    synapse = json.load(data_file)
    synapse_0 = np.asarray(synapse['synapse0'])
    synapse_1 = np.asarray(synapse['synapse1'])

def classify(sentence, show_details=False):
    results = think(sentence, show_details)

    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD ]
    results.sort(key=lambda x: x[1], reverse=True)
    return_results =[[classes[r[0]],r[1]] for r in results]
    print ("%s \n classification: %s" % (sentence, return_results))
    return return_results




print "begin processing of testing data"

ff = open('result.csv', 'w')
ff.write("date,rate\n")


with open('../fed_rates/fed_date_rate_testing.csv', 'r') as c1:
    reader = csv.reader(c1)
    next(reader, None)  # skip the headers
    prevRow = "1998-01-01"

    for row in reader:

        cur_year, cur_month, cur_day = str(row[0]).split("-")
        pre_year, pre_month, pre_day = prevRow.split("-")

        start_date = date(int(pre_year), int(pre_month), int(pre_day))
        end_date = date(int(cur_year), int(cur_month), int(cur_day))
        collection = []

        for single_date in daterange(start_date, end_date):

            with open('../../../../tmp2/finance_data/filtered_articles/nytimes/' + str(
                    single_date.strftime("%Y")) + "/" + str(single_date.strftime("%Y%m%d")) + ".npy",
                      'r') as myfile:

                print str(single_date.strftime("%Y-%m-%d"))

                data = np.load(myfile)

                if data.size == 0:
                    pass

                else:
                    for news in data:
                        collection.append(news)

        prevRow = row[0]

        ff.write(cur_year + '-' + cur_month + '-' + cur_day + ',' + str(classify(''.join(collection))) + "\n")

        print "######## finish one section ########"


def errorRate():
    with open('../fed_rates/fed_date_rate_testing.csv', 'r') as testingData:
        with open('result.csv', 'r') as result:
            reader_t = csv.reader(testingData)
            reader_r = csv.reader(result)


            next(reader_t, None)  # skip the headers
            next(reader_r, None)

            correctNum = 0

            testingResult = []

            for row in reader_t:
                testingResult.append(row[1])

            for row in reader_r:

                if row[1] == testingResult[reader_r.line_num-2]:
                    correctNum +=1





    print "Correct Classification Number: " + str(correctNum)
    print "Total Number: " + str(len(testingResult))
    print "correct Rate: "
    correctRate = float(correctNum)/float(len(testingResult))
    print correctRate






