#-*- coding: utf-8 -*-

# KNN
"""
那么动作片具有哪些共有特征,使得动作片之间非常类似,
而与爱情片存在着明显的差别呢?动作片中也会存在接吻镜头,爱情片中也会存在打斗场景,我们
不能单纯依靠是否存在打斗或者亲吻来判断影片的类型。但是爱情片中的亲吻镜头更多,动作片中
的打斗场景也更频繁,基于此类场景在某部电影中出现的次数可以用来进行电影分类。本章第一节
基于电影中出现的亲吻、打斗出现的次数,使用k-近邻算法构造程序,自动划分电影的题材类型。

简单地说,k-近邻算法采用测量不同特征值之间的距离方法进行分类。
k-近邻算法
优点:精度高、对异常值不敏感、无数据输入假定。
缺点:计算复杂度高、空间复杂度高。
适用数据范围:数值型和标称型

它的工作原理是:存在一个样本数
据集合,也称作训练样本集,并且样本集中每个数据都存在标签,即我们知道样本集中每一数据
与所属分类的对应关系。输入没有标签的新数据后,将新数据的每个特征与样本集中数据对应的
特征进行比较,然后算法提取样本集中特征最相似数据(最近邻)的分类标签。一般来说,我们
只选择样本数据集中前k个最相似的数据,这就是k-近邻算法中k的出处,通常k是不大于20的整数。
最后,选择k个最相似数据中出现次数最多的分类,作为新数据的分类。


k-近邻算法是分类数据最简单最有效的算法,本章通过两个例子讲述了如何使用k-近邻算法
构造分类器。k-近邻算法是基于实例的学习,使用算法时我们必须有接近实际数据的训练样本数
据。k-近邻算法必须保存全部数据集,如果训练数据集的很大,必须使用大量的存储空间。此外,
由于必须对数据集中的每个数据计算距离值,实际使用时可能非常耗时

"""

from numpy import *
import operator             # 运算符模块
from os import listdir
import matplotlib
import matplotlib.pyplot as plt

##################################################################################################
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
    print("dataSetSize",dataSet.shape)
    diffMat = tile(inX, (dataSetSize,1)) - dataSet  #numpy.tile([0,0],(2,1))#在列方向上重复[0,0]1次，行2次,array([[0, 0],[0, 0]])
    #print("diffMat",diffMat)
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1) #axis=1以后就是将一个矩阵的每一行向量相加
    distances = sqDistances**0.5   # 距离计算  欧氏距离公式
    #print("distances",distances)
    sortedDistIndicies = distances.argsort() #argsort函数返回的是数组值从小到大的索引值
    #print("sortedDistIndicies",sortedDistIndicies)
    classCount={}   #dirctionary
    # get the nearest K  选择距离最小的k个点
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1  #get('e', 0) tells python to look for the key 'e' in the dictionary. If it's not found it returns 0
        #print("classCount",classCount)
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)  # 按照第二个元素的次序对元组进行排序,此处的排序为逆序,即按照从最大到最小次序排序,最后返回发生频率最高的元素标签
    #print("sortedClassCount",sortedClassCount) 
    return sortedClassCount[0][0]


############################################################################################
# 示例:使用 k-近邻算法改进约会网站的配对效果

# txt file to array
def file2matrix(filename):
    fr = open(filename)                         # 得到文件行数
    numberOfLines = len(fr.readlines())         #get the number of lines in the file
    returnMat = zeros((numberOfLines,3))        #prepare matrix to return 
    # 创建返回的NumPy矩阵  然后创建以零填充的矩阵NumPy 为了简化处理,我们将该矩阵的另一维度设置为固定值 3 ,你可以按照自己的实际需求增加相应的代码以适应变化的输入值。循环处理文件中的每行数据
    classLabelVector = []                       #prepare labels return   
    fr = open(filename)
    index = 0
    # 解析文件数据到列表
    for line in fr.readlines():                     # readlines() 方法用于读取所有行(直到结束符 EOF)并返回列表
        line = line.strip()                         # Python strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列
        listFromLine = line.split('\t')             # Python split() 通过指定分隔符对字符串进行切片
        returnMat[index,:] = listFromLine[0:3]              # 选取前3个元素,将它们存储到特征矩阵中
        classLabelVector.append(int(listFromLine[-1]))      # 将列表的最后一列存储到向量 classLabelVector 中
        index += 1
    return returnMat,classLabelVector

#normalization  归一化数据
# 在处理这种不同取值范围的特征值时,我们通常采用的方法是将数值归一化,如将取值范围处理为0到1或者1到1之间。
def autoNorm(dataSet):
    # 为了归一化特征值,我们必须使用当前值减去最小值,然后除以取值范围。
    minVals = dataSet.min(0) #axis=0; 每列的最小值
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1)) #特征值矩阵有1000×3个值,而 minVals 和 range 的值都为1×3。为了解决这个问题,我们使用NumPy库中 tile() 函数将变量内容复制成输入矩阵同样大小的矩阵
    normDataSet = normDataSet/tile(ranges, (m,1))   #element wise divide 特征值相除
    return normDataSet, ranges, minVals

#####################################################################################################
#实验说明：预测约会对象对用户是否具有吸引力
#the date test
def datingClassTest():
    hoRatio = 0.10      #hold out 50%
    # 使用了 file2matrix 和 autoNorm() 函数从文件中读取数据并将其转换为归一化特征值
    datingDataMat,datingLabels = file2matrix('datingTestSet.txt')       #load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print ( "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]) )
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print ("the total error rate is: %f" % (errorCount/float(numTestVecs)) )
    print (errorCount)



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
# kNN算法_手写识别实例
# 需要识别的数字已经使用图形处理软件,处理成具有相同的色彩和大小 1 :宽高是32像素×32像素的黑白图像。

# 将图像格式化处理为一个向量。我们将把一个32×32的二进制图像矩阵转换为1×1024的向量,
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
    trainingFileList = listdir('trainingDigits')           #load the training set 获取目录内容
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]       # 从文件名解析分类数字
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0]) # 该目录下的文件按照规则命名,如文件9_45.txt的分类是9,它是数字9的第45个实例
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
        print ("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr) )
        if (classifierResult != classNumStr): errorCount += 1.0
    print ("\nthe total number of errors is: %d" % errorCount )
    print ("\nthe total error rate is: %f" % (errorCount/float(mTest)) )





if __name__ == "__main__":

    """
    # the bacis KNN algorithm
    group,labels=createDataSet()
    # print("group",group)
    # print("labels",labels)
    print( classify0([0,0],group,labels,3) )
    """
    
    """
    #Read data from .txt file
    datingDataMat,datingLabels=file2matrix('datingTestSet.txt')
    #print("datingLabels",datingLabels)
    #using Matplotlib    
    fig=plt.figure()
    ax=fig.add_subplot(111)  #111”表示“1×1网格，第一子图”，“234”表示“2×3网格，第四子图”。
    ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15*array(datingLabels),15*array(datingLabels)) # scatter
    plt.show()
    # 散点图使用datingDataMat矩阵的第二、第三列数据,分别表示特征值“玩视频游戏所耗时间百分比”和“每周所消费的冰淇淋公升数”

   


    # the data test
    print(datingClassTest())

    #print( classifyPerson() )
    """

    #handwritingClassTest()
