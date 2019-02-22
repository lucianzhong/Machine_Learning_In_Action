#-*- coding: utf-8 -*-
from numpy import *
import operator    
from os import listdir
import matplotlib
import matplotlib.pyplot as plt


def createDataSet():
    group = array(  [ [1.0,1.1],[1.0,1.0],[0,0],[0,0.1] ]  )
    labels = ['A','A','B','B']
    return group, labels


'''
1）计算测试数据与各个训练数据之间的距离；
2）按照距离的递增关系进行排序；
3）选取距离最小的K个点；
4）确定前K个点所在类别的出现频率；
5）返回前K个点中出现频率最高的类别作为测试数据的预测分类
'''
#the basic KNN algorithm
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  # (4*2 the shape of dataSet)
    #print("dataSetSize",dataSet.shape)
    diffMat = tile(inX, (dataSetSize,1)) - dataSet  #numpy.tile([0,0],(2,1))#在列方向上重复[0,0]1次，行2次,array([[0, 0],[0, 0]])
    #print("diffMat",tile(inX, (dataSetSize,1)))
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1) #axis=1以后就是将一个矩阵的每一行向量相加
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort() #argsort函数返回的是数组值从小到大的索引值
    #print("sortedDistIndicies",sortedDistIndicies)
    classCount={}   #dirctionary
    # get the nearest K
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1  #get('e', 0) tells python to look for the key 'e' in the dictionary. If it's not found it returns 0
        #print("classCount",classCount)
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    #print("sortedClassCount",sortedClassCount)
    return sortedClassCount[0][0]

# txt file to array
def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())         #get the number of lines in the file
    returnMat = zeros((numberOfLines,3))        #prepare matrix to return
    classLabelVector = []                       #prepare labels return   
    fr = open(filename)
    index = 0
    for line in fr.readlines(): #readlines() 方法用于读取所有行(直到结束符 EOF)并返回列表
        line = line.strip()     #Python strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列
        listFromLine = line.split('\t')  #Python split() 通过指定分隔符对字符串进行切片
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

#normalization  归一化数据
def autoNorm(dataSet):
    minVals = dataSet.min(0) #axis=0; 每列的最小值
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))   #element wise divide
    return normDataSet, ranges, minVals

#####################################################################################################
#实验说明：预测约会对象对用户是否具有吸引力
#the date test
def datingClassTest():
    hoRatio = 0.10      #hold out 50%
    datingDataMat,datingLabels = file2matrix('datingTestSet.txt')       #load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print "the total error rate is: %f" % (errorCount/float(numTestVecs))
    print errorCount



#the classify a person based on inputs
def classifyPerson():
	resultList=['not at all','in small doses','in large doses']
	percentTats=float(raw_input("percentage of time playing video game?"))
	ffile=float(raw_input("frequent flier miles earned per year ?"))
	icecream=float(raw_input("ice cream consumed per year ?"))
	datingDataMat,datingLabels = file2matrix('datingTestSet.txt')       #load data setfrom file
	normMat, ranges, minVals = autoNorm(datingDataMat)
	inArr=array([percentTats,ffile,icecream])
	classification_result=classify0( (inArr-minVals) / ranges, normMat, datingLabels, 3 )
	return ( "how much you like the person",resultList[classification_result-1])



#####################################################################################################
#kNN算法_手写识别实例
def img2vector(filename):
    returnVect = zeros((1,1024)) #vector(1*1024)
    fr = open(filename)
    for i in range(32):  # image 32*32
        lineStr = fr.readline()
        #print("lineStr",lineStr)  #('lineStr', '00000000001111111111111111100000\r\n')
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')           #load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')        #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr): errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))






if __name__ == "__main__":
    group,labels=createDataSet()
    # print("group",group)
    # print("labels",labels)

    print( classify0([0,0],group,labels,3) )


    datingDataMat,datingLabels=file2matrix('datingTestSet.txt')
    #print("datingLabels",datingLabels)

    #using Matplotlib
    '''
    fig=plt.figure()
    ax=fig.add_subplot(111)  #111”表示“1×1网格，第一子图”，“234”表示“2×3网格，第四子图”。
    ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15*array(datingLabels),15*array(datingLabels)) # scatter
    plt.show()
    '''


    # the data test
    #print(datingClassTest())

    #print( classifyPerson() )

    handwritingClassTest()