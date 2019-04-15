#-*- coding: utf-8 -*-
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
	xMat=mat(xArr)
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
    weights = mat(eye((m)))                     #创建对角权重矩阵
    for j in range(m):                          #next 2 lines create weights matrix
        diffMat = testPoint - xMat[j,:]    
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))       #权重大小以指数级衰减
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

######################################################################################################################
#岭回归
#给定lamda数值的情况下，计算回归系数
def ridgeRegres(xMat,yMat,lam=0.2):
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
    yMean = mean(yMat,0) #所有元素的平均值
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


