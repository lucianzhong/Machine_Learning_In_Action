#-*- coding: utf-8 -*-


"""
基于贝叶斯决策理论的分类方法
朴素贝叶斯
优点:在数据较少的情况下仍然有效,可以处理多类别问题。
缺点:对于输入数据的准备方式较为敏感。
适用数据类型:标称型数据

贝叶斯决策理论的核心思想,即选择具有最高概率的决策


对于分类而言,使用概率有时要比使用硬规则更为有效。贝叶斯概率及贝叶斯准则提供了一
种利用已知值来估计未知概率的有效方法。
可以通过特征之间的条件独立性假设,降低对数据量的需求。独立性假设是指一个词的出现
概率并不依赖于文档中的其他词。当然我们也知道这个假设过于简单。这就是之所以称为朴素贝
叶斯的原因。尽管条件独立性假设并不正确,但是朴素贝叶斯仍然是一种有效的分类器。
利用现代编程语言来实现朴素贝叶斯时需要考虑很多实际因素。下溢出就是其中一个问题,
它可以通过对概率取对数来解决。词袋模型在解决文档分类问题上比词集模型有所提高。还有其
他一些方面的改进,比如说移除停用词,当然也可以花大量时间对切分器进行优化。


"""

from numpy import *
import operator    
from os import listdir
import matplotlib
import matplotlib.pyplot as plt


##############################################################################################3
# 使用朴素贝叶斯进行文档分类
# create some experience samples
#文本看成单词向量或者词条向量,也就是说将句子转换为向量。考虑出现在所有文档中的所有单词,再决定将哪些词纳入词汇表或者说所要的词汇集合,然后必须要将每一篇文档转换为词汇表上的向量


# 函数 loadDataSet() 创建了一些实验样本。该函数返回的第一个变量是进行词条切分后的文档集合
# 第二个变量是一个类别标签的合。这里有两类,侮辱性和非侮辱性。这些文本的类别由人工标注,这些标注信息用于训练程序以便自动检测侮辱性留言
def loadDataSet():
    postingList=[ ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                  ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                  ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                  ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                  ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                  ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid'] ]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not  #1,侮辱  0,正常
    return postingList,classVec

# 函数 createVocabList() 会创建一个包含在所有文档中出现的不重复词的列表
def createVocabList(dataSet):
    vocabSet = set([])  #创建一个空集  #set() 函数创建一个无序不重复元素集，可进行关系测试，删除重复数据，还可以计算交集、差集、并集等
    for document in dataSet:
        vocabSet = vocabSet | set(document) # 创建两个集合的并集 / 将每篇文档返回的新词集合添加到该集合中 union of the two sets
    return list(vocabSet)

# 构造文档向量, the world is in the vacabulary or not
# 统计的单词是否在词库中出现  该函数的输入参数为词汇表及某个文档,输出的是文档向量,向量的每一元素为1或0,分别表示词汇表中的单词在输入文档中是否出现

def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)          #创建一个所含元素都为0的向量
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1  #遍历文档中的所有单词,如果出现了词汇表中的单词,则将输出的文档向量中的对应值设为1
        else: print ("the word: %s is not in my Vocabulary!" % word)
    return returnVec

#计算概率
#朴素贝叶斯分类器训练集
"""
利用贝叶斯分类器对文档进行分类时,要计算多个概率的乘积以获得文档属于某个类别的概
率,即计算 p(w 0 |1)p(w 1 |1)p(w 2 |1) 。如果其中一个概率值为0,那么最后的乘积也为0。为降低
这种影响,可以将所有词的出现数初始化为1,并将分母初始化为2
"""
# 输入参数为文档矩阵 trainMatrix ,以及由每篇文档类别标签所构成的向量trainCategory
def trainNB0(trainMatrix,trainCategory):                 #传入参数为文档矩阵，每篇文档类别标签所构成的向量
    numTrainDocs = len(trainMatrix)                      #文档矩阵的长度
    numWords = len(trainMatrix[0])                       #第一个文档的单词个数
    pAbusive = sum(trainCategory)/float(numTrainDocs)    #任意文档属于侮辱性文档概率   #侮辱性文件出现的概率，这个例子只有两个分类，非侮辱性概率 = 1- 侮辱性的概率
    #侮辱性文件的个数除以文件总数 = 侮辱性文件出现的概率
    p0Num = ones(numWords); p1Num = ones(numWords)      #change to ones()   # 单词出现的次数
    p0Denom = 2.0; p1Denom = 2.0                        #change to 2.0      # 初始化程序中的分子变量和分母变量
    for i in range(numTrainDocs):                       #遍历所有的文件
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]                     #累加侮辱词的频次
            p1Denom += sum(trainMatrix[i])              #对每篇文章的侮辱词的频次进行统计
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 对每个元素除以该类别中的总词数
    p1Vect = log(p1Num/p1Denom)          #change to log()   #类别1，侮辱性文档的列表[log(P(F1|C1)...]
    p0Vect = log(p0Num/p0Denom)          #change to log()   #类别0，正常文档的列表
    # 取对数是为了防止多个很小的数相乘使得程序下溢出或者得到不正确答案,采用自然对数进行处理不会有任何损失
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
# 将每个词的出现与否作为一个特征,这可以被描述为词集模型(set-of-words model),词集中,每个词只能出现一次
# 如果一个词在文档中出现不止一次,这可能意味着包含该词是否出现在文档中所不能表达的某种信息,这种方法被称为词袋模型(bag-of-words model)

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

######################################################################################
# 素贝叶斯算法之过滤垃圾邮件
# 函数 spamTest() 会输出在10封随机选择的电子邮件上的分类错误率

def spamTest():
    # 导入并解析文本文件 / 导入文件夹 spam 与 ham下的文本文件,并将它们解析为词列表
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
    # 随机构建训练集 / 本例中共有50封电子邮件,并不是很多,其中的10封电子邮件被随机选择为测试集
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])  
    trainMat=[]; trainClasses = []
    # 对测试集分类
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    # 随机选择数据的一部分作为训练集,而剩余部分作为测试集的过程称为留存交叉验证(hold-out cross validation)
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