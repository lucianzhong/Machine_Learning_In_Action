#-*- coding: utf-8 -*-
from numpy import *
import operator    
from os import listdir
import matplotlib
import matplotlib.pyplot as plt


# create some experience samples
def loadDataSet():
    postingList=[ ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                  ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                  ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                  ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                  ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                  ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid'] ]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not  #1,侮辱  0,正常
    return postingList,classVec

# 创建词汇表, 不重复的词列表
def createVocabList(dataSet):
    vocabSet = set([])  #create empty set  #set() 函数创建一个无序不重复元素集，可进行关系测试，删除重复数据，还可以计算交集、差集、并集等
    for document in dataSet:
        vocabSet = vocabSet | set(document) #union of the two sets
    return list(vocabSet)

# 构造文档向量, the world is in the vacabulary or not
# 统计的单词是否在词库中出现
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)          #创建一个所含元素都为0的向量
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print ("the word: %s is not in my Vocabulary!" % word)
    return returnVec

#计算概率
#朴素贝叶斯分类器训练集
def trainNB0(trainMatrix,trainCategory):                 #传入参数为文档矩阵，每篇文档类别标签所构成的向量
    numTrainDocs = len(trainMatrix)                      #文档矩阵的长度
    numWords = len(trainMatrix[0])                       #第一个文档的单词个数
    pAbusive = sum(trainCategory)/float(numTrainDocs)    #任意文档属于侮辱性文档概率   #侮辱性文件出现的概率，这个例子只有两个分类，非侮辱性概率 = 1- 侮辱性的概率
    #侮辱性文件的个数除以文件总数 = 侮辱性文件出现的概率
    p0Num = ones(numWords); p1Num = ones(numWords)      #change to ones()   #单词出现的次数
    p0Denom = 2.0; p1Denom = 2.0                        #change to 2.0      #整个数据集中单词出现的次数
    for i in range(numTrainDocs):                       #遍历所有的文件
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]                     #累加侮辱词的频次
            p1Denom += sum(trainMatrix[i])              #对每篇文章的侮辱词的频次进行统计
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num/p1Denom)          #change to log()   #类别1，侮辱性文档的列表[log(P(F1|C1)...]
    p0Vect = log(p0Num/p0Denom)          #change to log()   #类别0，正常文档的列表
    # 取对数是为了防止多个很小的数相乘使得程序下溢出或者得到不正确答案
    return p0Vect,p1Vect,pAbusive


#导入另外两篇我们不知道分类的文档，通过已经训练好的朴素贝叶斯算法来对其进行分类，函数classify()是选取分类概率最大的类别
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)           #element-wise mult  #元素相乘  # 对应元素相乘，log中变为相加
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0

# testEntry = ['love', 'my', 'dalmation'], ['stupid', 'garbage']
def testingNB():
    listOPosts,listClasses = loadDataSet()                #产生文档矩阵和对应的标签
    myVocabList = createVocabList(listOPosts)             #创建并集
    trainMat=[]                                           #创建一个空的列表
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))         #使用词向量来填充trainMat列表
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))          #训练函数
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print ( testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb) )
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print (testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)  )

###################################################################################################################################333
#词袋模型
#统计侮辱性email出现概率，侮辱性邮件各单词出现的概率，非侮辱性email各单词出现概率
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


#文件解析和垃圾邮件测试函数
def textParse(bigString):    #input is big string, #output is word list
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2] 

#朴素贝叶斯算法之过滤垃圾邮件
def spamTest():
    docList=[]; classList = []; fullText =[]
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)#create vocabulary
    trainingSet = range(50); testSet=[]           #create test set
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])  
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
            print ( "classification error",docList[docIndex] )
    print ( 'the error rate is: ',float(errorCount)/len(testSet) )
    #return vocabList,fullText


if __name__ == "__main__":
    #listOPosts,listClasses=loadDataSet()
    #print("listOPosts",listOPosts)

    #myVocabList=createVocabList(listOPosts)
    #print("myVocabList",myVocabList)

    #setOfWords2Vec(myVocabList,listOPosts[0])
    #print("listOPosts[0]",listOPosts)

    #print ( setOfWords2Vec(myVocabList,myVocabList[1:3]) )



    testingNB()

    spamTest()
    """
        trainMat=[]
    for postinDoc in listOPosts:
            trainMat.append( setOfWords2Vec(myVocabList,postinDoc) )
    poV,p1V,pAb=trainNB0(trainMat,listClasses)
    # print("poV",poV)

    """