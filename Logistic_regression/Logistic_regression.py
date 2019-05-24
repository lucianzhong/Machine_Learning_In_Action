#-*- coding: utf-8 -*-

"""
利用Logistic
回归进行分类的主要思想是:根据现有数据对分类边界线建立回归公式,以此进行分类。这里的“回归”一词源于最佳拟合,表示要找到最佳拟合参数集,

基于 Logistic 回归和 Sigmoid 函数的分类

优点:计算代价不高,易于理解和实现。
缺点:容易欠拟合,分类精度可能不高。
适用数据类型:数值型和标称型数据

为了实现Logistic回归分类器,我们可以在每个特征上都乘以一个回归系数,然后把所有的结果值相加,将这个总和代入Sigmoid函数中,进而得到一个范围在0~1之间的数值。任
何大于0.5的数据被分入1类,小于0.5即被归入0类。所以,Logistic回归也可以被看成是一种概率估计

Logistic回归的目的是寻找一个非线性函数Sigmoid的最佳拟合参数,求解过程可以由最优化
算法来完成。在最优化算法中,最常用的就是梯度上升算法,而梯度上升算法又可以简化为随机
梯度上升算法。
随机梯度上升算法与梯度上升算法的效果相当,但占用更少的计算资源。此外,随机梯度上
升是一个在线算法,它可以在新数据到来时就完成参数更新,而不需要重新读取整个数据集来进
行批处理运算。
机器学习的一个重要问题就是如何处理缺失数据。这个问题没有标准答案,取决于实际应用
中的需求。现有一些解决方案,每种方案都各有优缺点。

"""




import numpy as np
import matplotlib.pyplot as plt

# loadDataSet() ,它的主要功能是打开文本文件 testSet.txt 并逐行读取。每行前两个值分别是X1和X2,第三个值是数据对应的类别标签
def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))


#梯度上升算法-计算回归系数
#梯度上升优化方法：每次更新参数时都需要遍历数据
# 第一个参数是 dataMatIn ,它是一个2维NumPy数组,每列分别代表每个不同的特征,每行则代表每个训练样本。我们现在采用的是100个样本的简单数据集,它包含了两个特征X1和X2,再加上第0维特征X0,所以 dataMathln 里存放的将是100×3的矩阵

def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)                                      #转换成numpy的mat
    labelMat = np.mat(classLabels).transpose()                          #转换成numpy的mat,并进行转置
    m, n = np.shape(dataMatrix)                                         #返回dataMatrix的大小。m为行数,n为列数。
    alpha = 0.001                                                       #移动步长,也就是学习速率,控制更新的幅度。
    maxCycles = 500                                                     #最大迭代次数
    weights = np.ones((n,1))
    for k in range(maxCycles):
        # 变量 h 不是一个数而是一个列向量,列向量的元素个数等于样本个数,这里是100。对应地,运算 dataMatrix * weights 代表的不止一次乘积计算,事实上该运算包含了300次的乘积
        h = sigmoid(dataMatrix * weights)                               #梯度上升矢量化公式
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights.getA()               # getA()是numpy的一个函数，numpy.matrix.getA,  matrix to array     

# 画出决策边界
def plotBestFit(weights):
    dataMat, labelMat = loadDataSet()                                   #加载数据集
    dataArr = np.array(dataMat)                                         #转换成numpy的array数组
    n = np.shape(dataMat)[0]                                            #数据个数
    xcord1 = []; ycord1 = []                                            #正样本
    xcord2 = []; ycord2 = []                                            #负样本
    for i in range(n):                                                  #根据数据集标签进行分类
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])    #1为正样本
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])    #0为负样本
    fig = plt.figure()
    ax = fig.add_subplot(111)                                           #添加subplot
    ax.scatter(xcord1, ycord1, s = 20, c = 'red', marker = 's',alpha=.5)#绘制正样本
    ax.scatter(xcord2, ycord2, s = 20, c = 'green',alpha=.5)            #绘制负样本
    x = np.arange(-3.0, 3.0, 0.1)
    #因此我们设定0=w0x0+w1x1+w2x2,然后接触X2和X1的关系式(即分割线的方程,X0 =1)
    #据梯度上升发，求出了最优化的参数weights，带入logistics分类器，y = (-weights[0]-weights[1]*x)/weights[2]，预测测试样本
    y = (-weights[0] - weights[1] * x) / weights[2]         # 最佳拟合直线
    ax.plot(x, y)
    plt.title('BestFit')                                                #绘制title
    plt.xlabel('X1'); plt.ylabel('X2')                                  #绘制label
    plt.show()      

# 随机梯度上升算法
# 梯度上升算法在每次更新回归系数时都需要遍历整个数据集,该方法在处理100个左右的数据集时尚可,但如果有数十亿样本和成千上万的特征,那么该方法的计算复杂度就太高了。一种改进方法是一次仅用一个样本点来更新回归系数,该方法称为随机梯度上升算法
def stocGradAscent0(dataMatrix, classLabels):
    dataMatrix=np.array(dataMatrix)
    m,n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)   #initialize to all ones
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h                  #提出增量的方法，每遍历一个样本就修改一次weighs
        weights = weights + alpha * error * dataMatrix[i]
    return weights

# 改进的随机梯度上升算法
# 1迭代150次改进的增量算法1，这样还是比迭代200次全集时间要约简不少，并且准确率不低；
# 2步长alpha不是固定的，这样可以开始收敛速度大，后来越来越准确的时候收敛速度慢点，后面加0.0001是防止alpha为0。步长为0原地不动迭代就没有意义；
# 3为了避免上图的波动情况，随机选取样本点训练，然后再原数据集中删除，避免重复使用；下图可以看出，选用随机点可以避免周期性波动，波动确实可以变小；
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    dataMatrix=np.array(dataMatrix)
    m,n = np.shape(dataMatrix)
    weights = np.ones(n)   #initialize to all ones
    for j in range(numIter):
        dataIndex = list (range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    # alpha 每次迭代时需要调整
            randIndex = int(np.random.uniform(0,len(dataIndex)))   # 随机选取更新 go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex]*weights))  #随机选取更新
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

#nX*weights大于0.5，则分类到1，否则分类到0；主意,inX和weights都是向量
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

#######################################################################################
# 从疝气病症，估计病马的死亡率
# colicTest() ,是用于打开测试集和训练集,并对数据进行格式化处理的函数
def colicTest():
    frTrain = open('horseColicTraining.txt'); 
    frTest = open('horseColicTest.txt')
    trainingSet = []; 
    trainingLabels = []

    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21])) # 导入训练集,同前面一样,数据的最后一列仍然是类别标签
    # 函数 stocGradAscent1() 来计算回归系数向量
    trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels, 1000)
    errorCount = 0; 
    numTestVec = 0.0

    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights))!= int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print ( "the error rate of this test is: %f" % errorRate   )
    return errorRate


# 调用函数 colicTest() 10次并求结果的平均值
def multiTest():
    numTests = 10; errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest()
    print ("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests))   )


if __name__ == "__main__":
    dataArr,labelMat=loadDataSet()
    #print("dataArr",dataArr)
    #print("labelMat",labelMat)

    """
    #梯度上升算法
    weights =  gradAscent(dataArr,labelMat)
    print("weights",weights)
    plotBestFit(weights)
    """




    #weights_1=stocGradAscent0(dataArr,labelMat)

    #weights_2=stocGradAscent1(dataArr,labelMat)
    #plotBestFit(weights_1)


    multiTest()



