#-*- coding: utf-8 -*-

"""
优点:结果易于理解,计算上不复杂。
缺点:对非线性的数据拟合不好。
适用数据类型:数值型和标称型数据。


与分类一样,回归也是预测目标值的过程。回归与分类的不同点在于,前者预测连续型变量,
而后者预测离散型变量。回归是统计学中最有力的工具之一。在回归方程里,求得特征对应的最
佳回归系数的方法是最小化误差的平方和。给定输入矩阵 X ,如果 X T X 的逆存在并可以求得的话,
回归法都可以直接使用。数据集上计算出的回归方程并不一定意味着它是最佳的,可以使用预测
值 yHat 和原始值 y 的相关性来度量回归方程的好坏。
当数据的样本数比特征数还少时候,矩阵 X T X 的逆不能直接计算。即便当样本数比特征数多
时, X T X 的逆仍有可能无法直接计算,这是因为特征有可能高度相关。这时可以考虑使用岭回归,
因为当 X T X 的逆不能计算时,它仍保证能求得回归参数。
岭回归是缩减法的一种,相当于对回归系数的大小施加了限制。另一种很好的缩减法是lasso。
Lasso难以求解,但可以使用计算简便的逐步线性回归方法来求得近似结果。
缩减法还可以看做是对一个模型增加偏差的同时减少方差。偏差方差折中是一个重要的概
念,可以帮助我们理解现有模型并做出改进,从而得到更好的模型。

"""


from numpy import *
import operator    
from os import listdir
import matplotlib
import matplotlib.pyplot as plt
from time import sleep
import json
import urllib3
from urllib.request import urlopen

#数据导入函数 
def loadDataSet(fileName):      #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) - 1 #get number of fields 
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

"""
用来计算最佳拟合直线。该函数首先读入 x 和 y 并将它们保存到矩阵中;然后计算 X T X ,然后判断它的行列式是否为零,如果行列式为零,那么计算逆矩阵的时候将出现错误。NumPy提供一个线性代数的
库linalg,其中包含很多有用的函数。可以直接调用 linalg.det() 来计算行列式。最后,如果行列式非零,计算并返回 w 。如果没有检查行列式是否为零就试图计算矩阵的逆,将会出现错误。
NumPy的线性代数库还提供一个函数来解未知矩阵,如果使用该函数,那么代码 ws=xTx.I *(xMat.T*yMat) 应写成 ws=linalg.solve(xTx, xMat.T*yMatT) 
"""
#标准回归函数
def standRegres(xArr,yArr):
    xMat = mat(xArr);
    yMat = mat(yArr).T
    xTx = xMat.T*xMat
    if linalg.det(xTx) == 0.0:                              #判断矩阵是否可逆,np.linalg.det()矩阵求行列式（标量）
        print ("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws


#标准回归函数
def result_of_standRegres():
	xArr,yArr=loadDataSet('ex0.txt')
	ws= standRegres(xArr,yArr) 
	print("ws  ",ws)
	xMat=mat(xArr)         # 转换为 NumPy 矩 阵数据类型
	yMat=mat(yArr)
	fig=plt.figure()
	ax=fig.add_subplot(111)
	ax.scatter(xMat[:,1].flatten().A[0],yMat.T[0:,0].flatten().A[0]) # plot the original data
	xCopy=xMat.copy()
	xCopy.sort()
	#print("xCopy ",xCopy)
	yHat=xCopy*ws
	ax.plot(xCopy[:,0],yHat)
	plt.show()


#######################################################################################################
#局部加权线性回归
#因为线性回归求得具有最小均方误差的无偏估计。所以他可能出现欠拟合现象。 
#模型欠拟合将不能有好的预测结果，所以有些方法允许在估计中引入一些偏差，从而降低预测的均方误差。 
#中一个方法是局部加权线性回归（LWLR），我们给待预测点附近的每个点赋予一定的权重，然后在这个子集上基于最小均值方差来进行普通的回归。每次预测均需要事先选取出对应的数据子集
# w = (X^T WX)^-1 * X^TWy

def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = mat(xArr);
    yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))                     #创建对角权重矩阵 / 权重矩阵是一个方阵,阶数等于样本点个数
    for j in range(m):                          #next 2 lines create weights matrix
        diffMat = testPoint - xMat[j,:]    
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))       # 随着样本点与待预测点距离的递增,权重将以指数级衰减
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print ("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))                    #给待预测点附近的每个点赋予一定的权重，然后在这个子集上基于最小均值方差来进行普通的回归
    return testPoint * ws

def lwlrTest(testArr,xArr,yArr,k=1.0):  #loops over all the data points and applies lwlr to each one
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat

def result_of_lwlr():
    xArr,yArr=loadDataSet('ex0.txt')
    yHat=lwlrTest(xArr,xArr,yArr,1)  # change k=1, k=0.001, k=0.03
    xMat=mat(xArr)
    yMat=mat(yArr)
    srtInd=xMat[:,1].argsort(0)  #argsort函数返回的是数组值从小到大的索引值
    xSort=xMat[srtInd][:,0,:]
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xMat[:,1].flatten().A[0],yMat.T[0:,0].flatten().A[0]) # plot the original data
    ax.plot(xSort[:,1],yHat[srtInd])
    plt.show()
######################################################################################################################

#预测鲍鱼年龄
#calcaulate error
def rssError(yArr,yHatArr): #yArr and yHatArr both need to be arrays
    return ((yArr-yHatArr)**2).sum()

def rssError_result():
    abX,abY=loadDataSet('abalone.txt')
    yHat01 = lwlrTest(abX[0:99],abX[0:99],abY[0:99],0.1 )
    yHat1  = lwlrTest(abX[0:99],abX[0:99],abY[0:99],1   )
    yHat10 = lwlrTest(abX[0:99],abX[0:99],abY[0:99],10  )

    error_01 = rssError(abY[0:99],yHat01.T )
    error_1  = rssError(abY[0:99],yHat1.T  )
    error_10 = rssError(abY[0:99],yHat10.T )

    print("error_01,error_1,error_10  ",error_01,error_1,error_10)
    # 从上述结果可以看到,在上面的三个参数中,核大小等于10时的测试误差最小,但它在训练集上的误差却是最大的


######################################################################################################################
# 岭回归
# 如果特征比样本点还多( n > m ),也就是说输入数据的矩阵 X 不是满秩矩阵。非满秩矩阵在求逆时会出现问题。为了解决这个问题,统计学家引入了岭回归(ridge regression)的概念
# 给定lamda数值的情况下，计算回归系数
# 岭回归就是在矩阵 X T X 上加一个λ I 从而使得矩阵非奇异,进而能对 X T X + λ I 求逆
# 首先构建矩阵 X T X ,然后用 lam 乘以单位矩阵(可调用NumPy库中的方法 eye() 来生成)

def ridgeRegres(xMat,yMat,lam=0.2):  # 如果没指定lambda,则默认为0.2。由于lambda是Python保留的关键字,因此程序中使用了 lam 来代替
    xTx = xMat.T*xMat
    denom = xTx + eye(shape(xMat)[1])*lam
    if linalg.det(denom) == 0.0:
        print ("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xMat.T*yMat)
    return ws

#测试在一组lamda上的结果
def ridgeTest(xArr,yArr):
    xMat = mat(xArr); 
    yMat=mat(yArr).T
    yMean = mean(yMat,0)    #所有元素的平均值
    yMat = yMat - yMean     #to eliminate X0 take mean off of Y
    #regularize X's
    xMeans = mean(xMat,0)   #calc mean then subtract it off
    xVar = var(xMat,0)      #calc variance of Xi then divide by it
    xMat = (xMat - xMeans)/xVar
    numTestPts = 30
    wMat = zeros((numTestPts,shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat,yMat,exp(i-10))
        wMat[i,:]=ws.T
    return wMat

#########################################################################################################################################

#预测乐高套装玩具的价格
def searchForSet(retX, retY, setNum, yr, numPce, origPrc):
    sleep(10)
    myAPIstr = 'AIzaSyD2cR2KFyx12hXu6PFU-wrWot3NXvko8vY'
    searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json' % (myAPIstr, setNum)
    pg = urlopen(searchURL)
    retDict = json.loads(pg.read())
    for i in range(len(retDict['items'])):
        try:
            currItem = retDict['items'][i]
            if currItem['product']['condition'] == 'new':
                newFlag = 1
            else: newFlag = 0
            listOfInv = currItem['product']['inventories']
            for item in listOfInv:
                sellingPrice = item['price']
                if  sellingPrice > origPrc * 0.5:
                    print ("%d\t%d\t%d\t%f\t%f" % (yr,numPce,newFlag,origPrc, sellingPrice) )
                    retX.append([yr, numPce, newFlag, origPrc])
                    retY.append(sellingPrice)
        except: print ('problem with item %d' % i)

def setDataCollect(retX, retY):
    searchForSet(retX, retY, 8288, 2006, 800, 49.99)
    searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
    searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
    searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
    searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
    searchForSet(retX, retY, 10196, 2009, 3263, 249.99)

def predict_Lego_price():
    lgX=[]
    lgY=[]
    setDataCollect(lgX,lgY)

##############################################################################################################################3

if __name__ == "__main__":
    #result_of_standRegres()
    #result_of_lwlr()
    rssError_result()
    
    #predict_Lego_price()


