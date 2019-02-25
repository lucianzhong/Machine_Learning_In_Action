#-*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


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

#梯度上升优化方法：每次更新参数时都需要遍历数据
def gradAscent(dataMatIn, classLabels):
	dataMatrix = np.mat(dataMatIn)										#转换成numpy的mat
	labelMat = np.mat(classLabels).transpose()							#转换成numpy的mat,并进行转置
	m, n = np.shape(dataMatrix)											#返回dataMatrix的大小。m为行数,n为列数。
	alpha = 0.001														#移动步长,也就是学习速率,控制更新的幅度。
	maxCycles = 500														#最大迭代次数
	weights = np.ones((n,1))
	for k in range(maxCycles):
		h = sigmoid(dataMatrix * weights)								#梯度上升矢量化公式
		error = labelMat - h
		weights = weights + alpha * dataMatrix.transpose() * error
	return weights.getA()		


def plotBestFit(weights):
	dataMat, labelMat = loadDataSet()									#加载数据集
	dataArr = np.array(dataMat)											#转换成numpy的array数组
	n = np.shape(dataMat)[0]											#数据个数
	xcord1 = []; ycord1 = []											#正样本
	xcord2 = []; ycord2 = []											#负样本
	for i in range(n):													#根据数据集标签进行分类
		if int(labelMat[i]) == 1:
			xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])	#1为正样本
		else:
			xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])	#0为负样本
	fig = plt.figure()
	ax = fig.add_subplot(111)											#添加subplot
	ax.scatter(xcord1, ycord1, s = 20, c = 'red', marker = 's',alpha=.5)#绘制正样本
	ax.scatter(xcord2, ycord2, s = 20, c = 'green',alpha=.5)			#绘制负样本
	x = np.arange(-3.0, 3.0, 0.1)
	y = (-weights[0] - weights[1] * x) / weights[2]
	ax.plot(x, y)
	plt.title('BestFit')												#绘制title
	plt.xlabel('X1'); plt.ylabel('X2')									#绘制label
	plt.show()		

#随机梯度上升算法
def stocGradAscent0(dataMatrix, classLabels):
    dataMatrix=np.array(dataMatrix)
    m,n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)   #initialize to all ones
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

# 改进的随机梯度上升算法
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    dataMatrix=np.array(dataMatrix)
    m,n = np.shape(dataMatrix)
    weights = np.ones(n)   #initialize to all ones
    for j in range(numIter):
        dataIndex = list (range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    #apha decreases with iteration, does not 
            randIndex = int(np.random.uniform(0,len(dataIndex)))#go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex]*weights))  #随机选取更新
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights




#从疝气病症，估计病马的死亡率

def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

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
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels, 1000)
    errorCount = 0; numTestVec = 0.0
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



def multiTest():
    numTests = 10; errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest()
    print ("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests))   )




if __name__ == "__main__":
	dataArr,labelMat=loadDataSet()
	#print("dataMat",dataMat)

	weights =  gradAscent(dataArr,labelMat)


	#plotBestFit(weights)

	weights_1=stocGradAscent0(dataArr,labelMat)

	weights_2=stocGradAscent1(dataArr,labelMat)
	#plotBestFit(weights_2)

	multiTest()

