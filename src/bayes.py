from numpy import *
import re
import math
import os
from nltk.corpus import stopwords

def readFileName():
    f = open('../training_articles/file.txt', 'w')
    for file in os.listdir("../training_articles"):
        if file == ".DS_Store":
            print ".DS_Store"
        else:
            print >>f, file

    f.close()

def loadDataSet():
    postingList = []

    with open('../training_articles/negative.txt', 'r') as myfile:
        data = myfile.read().replace('\n', ' ')
        regEx = re.compile('\\W*')
        listOfTokens = regEx.split(data)
        listOfTokens = [tok.lower() for tok in listOfTokens if len(tok) > 0]
        #filtered_words = [word for word in listOfTokens if word not in stopwords.words('english')]
        postingList.append(listOfTokens)


    with open('../training_articles/positive.txt', 'r') as myfile:
        data = myfile.read().replace('\n', '')
        regEx = re.compile('\\W*')
        listOfTokens = regEx.split(data)
        listOfTokens = [tok.lower() for tok in listOfTokens if len(tok) > 0]
        # filtered_words = [word for word in listOfTokens if word not in stopwords.words('english')]

        postingList.append(listOfTokens)

    classVec = [0, 1]

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

if __name__ == '__main__':
    testingNB()
