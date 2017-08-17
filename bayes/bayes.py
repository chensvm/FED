#-*- coding: utf-8 -*-

import numpy as np
import re
import csv
import math
from nltk.corpus import stopwords
from datetime import timedelta, date
import pandas as pd


def loadDataSet():

    print "start loading training dataset"
    postingList = []
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
                                # 同時建立一個分類向量搭配
                                classVec.append(1)

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

                    myfile.close()

            else:
                print "pass posting list error: "+ row[0]
                pass

            prevRow = row[0]

    print "finish postingList"
    # f = open ("postinglist.txt", 'w')
    # f.write(postingList)
    # f.close()
    #
    # g = open ("classVec.txt", 'w')
    # g.write(classVec)
    # g.close()



    return postingList, classVec

def createVocabList(dataSet):
    print "###createVocabList###"
    # 創建一個不重複詞的列表
    vocabSet = set([])  #create empty set
    for document in dataSet:
        #將voc list傳給set就會回傳一個不重複的詞列表
        vocabSet = vocabSet | set(document) #union of the two sets
        #藉由聯集，將新的不重複詞加入這個set當中

    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    #先建立一個和詞彙表等長的向量，暫時都設為0
    returnVec = [0]*len(vocabList)
    #比較文件與vocabList的差異，如果文件中有詞則為1，沒有則為0
    for word in inputSet:
        if word in vocabList:
            #如果詞彙表中的字在文件中有出現，則設為1
            returnVec[vocabList.index(word)] = 1
        else:
            pass
            #print "the word: %s is not in my Vocabulary!" % word
    return returnVec

def trainNB0(trainMatrix, trainCategory):

    #將每個詞出現與否作為一個feature

    print "*******trainNB0"

    posiCategoryNum = 0
    negaCategoryNum = 0
    netrualCategoryNum = 0



    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])


    for classNum in trainCategory:

        if classNum ==1:
            posiCategoryNum +=1
        elif classNum == 0:
            netrualCategoryNum +=1
        elif classNum == -1:
            negaCategoryNum +=1
        else:
            print "error"


    pPosi = float(posiCategoryNum)/float(numTrainDocs)
    pNeutral = float(netrualCategoryNum)/float(numTrainDocs)
    print "positive docs in taining data : " + str(posiCategoryNum)
    print "neutral docs in taining data : " + str(negaCategoryNum)


    # print  len(trainMatrix) #193
    # print  sum(trainCategory) #-78
    # print  float(numTrainDocs) #193

    # 為了避免機率為0，設定初始化出現次數為1
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    pNeNum = np.ones(numWords)

    #初始化p(wi|c1)和p(wi|c-1), p(wi|c0)
    # 為了避免機率為0，設定初始化分母為2
    p0Denom = 2.0
    p1Denom = 2.0
    pNeDenom = 2.0



    for i in range(numTrainDocs):


        if trainCategory[i] == 1:
            #當某個詞在某個文件中出現，該詞對應的p1Num或者p0Num, pNeNum就要加一

            p1Num += trainMatrix[i]
            # 該文件的總詞數也相應加一
            p1Denom += sum(trainMatrix[i])
        elif trainCategory[i] == 0:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
        elif trainCategory[i] == -1:
            pNeNum += trainMatrix[i]
            pNeDenom += sum(trainMatrix[i])


    #取log是為了避免值太小相乘溢位

    p1Vect = np.log(p1Num/p1Denom)           #change to log()
    p0Vect = np.log(p0Num/p0Denom)           #change to log()
    pNeVect = np.log(pNeNum/pNeDenom)        #change to log()

    #p1Vec是整個不重複詞向量中，每個詞計算出現在category 1的數量的向量

    return p0Vect, p1Vect, pNeVect, pNeutral, pPosi

def classifyNB(vec2Classify, p0Vec, p1Vec, pNeVec, pClass0, pClass1):

    p1 = sum(vec2Classify * p1Vec) + math.log(pClass1)    #element-wise mult
    p0 = sum(vec2Classify * p0Vec) + math.log(pClass0)
    pNe = sum(vec2Classify * pNeVec) + math.log(1.0 - pClass1 - pClass0)
    if p1 > p0 and p1 > pNe:
        return 1
    elif p0 > p1 and p0 > pNe:
        return 0
    elif pNe > p0 and pNe > p1:
        return -1
    else:
        pass

def bagOfWords2VecMN(vocabList, inputSet):

    #如果一個詞在文件中出現不只一次，在bag of words裡可以反映出對應的效果
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

    p0V, p1V, pNeV, pPosi, pNeg = trainNB0(np.array(trainMat), np.array(listClasses))
    # listClasses is classVec
    testEntry =[]

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
                            regEx = re.compile('\\W*')
                            listOfTokens = regEx.split(news)
                            listOfTokens = [tok.lower().encode('utf-8') for tok in listOfTokens if len(tok) > 0]
                            filtered_words = [word for word in listOfTokens if
                                              word not in stopwords.words('english')]
                            testEntry.append(filtered_words)

            prevRow = row[0]

            thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))

            ff.write(cur_year + '-' + cur_month + '-' + cur_day + ',' + str(
                classifyNB(thisDoc, p0V, p1V, pNeV, pPosi, pNeg)) + "\n")

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




def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)


if __name__ == '__main__':

    testingNB()
    errorRate()
