from numpy import *
import re
import math
import sys
import os
from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier
from nltk.corpus import stopwords
reload(sys)
sys.setdefaultencoding('utf8')


def loadDataSet():
    postingList = []

    with open('../training_articles/positive.txt', 'r') as myfile:
        data = myfile.read().replace('\n', ' ')
        regEx = re.compile('\\W*')
        listOfTokens = regEx.split(data)
        listOfTokens = [tok.lower() for tok in listOfTokens if len(tok) > 0]
        #filtered_words = [word for word in listOfTokens if word not in stopwords.words('english')]
        postingList.append(listOfTokens)


    with open('../training_articles/negative.txt', 'r') as myfile:
        data = myfile.read().replace('\n', '')
        regEx = re.compile('\\W*')
        listOfTokens = regEx.split(data)
        listOfTokens = [tok.lower() for tok in listOfTokens if len(tok) > 0]
        # filtered_words = [word for word in listOfTokens if word not in stopwords.words('english')]

        postingList.append(listOfTokens)

    classVec = [1, -1]

    return postingList, classVec

def createVocabList(dataSet):
    print "###createVocabList###"
    vocabSet = set([])  #create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document) #union of the two sets
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):

    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            pass
            #print "the word: %s is not in my Vocabulary!" % word
    return returnVec

def trainNB0(trainMatrix, trainCategory):
    print "*******trainNB0"
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = ones(numWords)
    p1Num = ones(numWords)      #change to ones()

    p0Denom = 2.0
    p1Denom = 2.0                        #change to 2.0

    for i in range(numTrainDocs):

        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    p1Vect = log(p1Num/p1Denom)            #change to log()
    p0Vect = log(p0Num/p0Denom)            #change to log()
    return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):

    p1 = sum(vec2Classify * p1Vec) + math.log(pClass1)    #element-wise mult
    p0 = sum(vec2Classify * p0Vec) + math.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def testingNB():
    print "begin processing of testing data"
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)

    trainMat=[]
    for postinDoc in listOPosts:

        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))

    p0V,p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    testEntry =[]
    with open('../testing_articles/file.txt', 'r') as myfile:
        for line in myfile:
            line = line.replace('\n', '')
            data = load('../testing_articles/'+line)
            for news in data:
                regEx = re.compile('\\W*')
                listOfTokens = regEx.split(news)
                listOfTokens = [tok.lower().encode('utf-8') for tok in listOfTokens if len(tok) > 0]
                testEntry.append(listOfTokens)
                # testEntry = ['love', 'my', 'dalmation']
                thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
                print line, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb)

########################################################################################################

def readTrainingFileList():
    f = open('../training_articles/file.txt', 'w')
    for file in os.listdir("../training_articles"):
        if file == ".DS_Store":
            pass
        elif file == "file.txt":
            pass
        elif "20050203.npy" <= file <= "20060807.npy":
            f.write(file+", 1.0\n")
        elif "20060808.npy" <= file <= "20070807.npy":
            f.write(file + ", 0.0\n")
        elif "20070808.npy" <= file <= "20080430.npy":
            f.write(file + ", -1.0\n")
        elif "20080501.npy" <= file <= "20080916.npy":
            f.write(file + ", -1.0\n")
        elif "20080917.npy" <= file <= "20081216.npy":
            f.write(file + ", -1.0\n")

    f.close()


def readTestingFileList():
    f = open('../testing_articles/file.txt', 'w')
    for file in os.listdir("../testing_articles"):
        if file == ".DS_Store":
            pass
        elif file == "file.txt":
            pass
        else:
            f.write(file + "\n")  # python will convert \n to os.linesep

    f.close()

def loadNpy():

    with open('../training_articles/file.txt', 'r') as myfile:
        content = myfile.readlines()
        for x in content:
            fileName, cate = x.split(',')
            cate.replace(" ", "")
            cate.replace('\n', '')

            if str(cate).rstrip().replace(" ", "") == "1.0":
                print "1.0"
                with open('../training_articles/positive.txt', 'a') as f:
                    data = load('../training_articles/' + str(fileName))
                    for news in data:
                        news.replace('\n', ' ')
                        f.write(news + ' ')

            if str(cate).rstrip().replace(" ", "") == '0.0':
                print "0.0"
                with open('../training_articles/neutral.txt', 'a') as g:
                    data = load('../training_articles/' + str(fileName))
                    for news in data:
                        news.replace('\n', ' ')
                        g.write(news + ' ')

            if str(cate).rstrip().replace(" ", "") == '-1.0':
                print "-1.0"
                with open('../training_articles/negative.txt', 'a') as h:
                    data = load('../training_articles/' + str(fileName))
                    for news in data:
                        news.replace('\n', ' ')
                        h.write(news + ' ')


# def preprocessingText():
#     with open('../training_articles/neutral.txt', 'w') as g:
#         stop = set(stopwords.words('english'))
#         i for i in sentence.lower().split() if i not in stop

def textBlobClassfier():
    train = []
    # with open('../training_articles/neutral.txt', 'r') as g:
    #     articles = g.readlines()
    #     for sentence in articles:
    #         train.append("(" + "\'" + sentence + "\'"+", " + "\'" + "0.0" + "\'")

    with open('../training_articles/positive.txt', 'r') as h:
        articles = h.read().splitlines()

        for sentence in articles:
            if sentence is not None:
                if sentence == '':
                    pass
                elif sentence == 'None':
                    pass
                elif str(sentence) == '\n':
                    pass
                else:
                    print sentence
                    print train.append("(" + "\'" + sentence + "\'"+", " + "\'" + "1.0" + "\'")
            else:
                pass
    # with open('../training_articles/negative.txt', 'r') as f:
    #     articles = f.readlines()
    #     for sentence in articles:
    #         train.append("(" + "\'" + sentence + "\'"+", " + "\'" + "-1.0" + "\'")





    # train = [
    #     ('I love this sandwich.', 'pos'),
    #     ('This is an amazing place!', 'pos'),
    #     ('I feel very good about these beers.', 'pos'),
    #     ('This is my best work.', 'pos'),
    #     ("What an awesome view", 'pos'),
    #     ('I do not like this restaurant', 'neg'),
    #     ('I am tired of this stuff.', 'neg'),
    #     ("I can't deal with this", 'neg'),
    #     ('He is my sworn enemy!', 'neg'),
    #     ('My boss is horrible.', 'neg')
    # ]
    test = [
        ('The beer was good.', 'pos'),
        ('I do not enjoy my job', 'neg'),
        ("I ain't feeling dandy today.", 'neg'),
        ("I feel amazing!", 'pos'),
        ('Gary is a friend of mine.', 'pos'),
        ("I can't believe I'm doing this.", 'neg')
    ]
    cl = NaiveBayesClassifier(train)

    # Classify some text
    print(cl.classify("Their burgers are amazing."))  # "pos"
    print(cl.classify("I don't like their pizza."))  # "neg"

    # Classify a TextBlob
    blob = TextBlob("The beer was amazing. But the hangover was horrible. "
                    "My boss was not pleased.", classifier=cl)
    print(blob)
    print(blob.classify())

    for sentence in blob.sentences:
        print(sentence)
        print(sentence.classify())

    # Compute accuracy
    print("Accuracy: {0}".format(cl.accuracy(test)))

    # Show 5 most informative features
    cl.show_informative_features(5)


if __name__ == '__main__':
    # loadNpy()
   #textBlobClassfier()
    testingNB()
    # testingNB()
