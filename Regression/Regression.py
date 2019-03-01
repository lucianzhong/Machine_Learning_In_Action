#-*- coding: utf-8 -*-
from numpy import *
import operator    
from os import listdir
import matplotlib
import matplotlib.pyplot as plt
from time import sleep
import json
import urllib2



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

def standRegres(xArr,yArr):
    xMat = mat(xArr); yMat = mat(yArr).T
    xTx = xMat.T*xMat
    if linalg.det(xTx) == 0.0:
        print ("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws


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
	print("xCopy ",xCopy)
	yHat=xCopy*ws
	ax.plot(xCopy[:,0],yHat)
	plt.show()


#局部加权线性回归

def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = mat(xArr); yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))
    for j in range(m):                      #next 2 lines create weights matrix
        diffMat = testPoint - xMat[j,:]     #
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print ("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

def lwlrTest(testArr,xArr,yArr,k=1.0):  #loops over all the data points and applies lwlr to each one
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat

def lwlrTestPlot():
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


#预测鲍鱼年龄


#calcaulate error
def rssError(yArr,yHatArr): #yArr and yHatArr both need to be arrays
    return ((yArr-yHatArr)**2).sum()

def rssError_result():
    abX,abY=loadDataSet('abalone.txt')
    yHat01=lwlrTest(abX[0:99],abX[0:99],abY[0:99],0.1)
    yHat1=lwlrTest(abX[0:99],abX[0:99],abY[0:99],1)
    yHat10=lwlrTest(abX[0:99],abX[0:99],abY[0:99],10)

    error_01=rssError(abY[0:99],yHat01.T)
    error_1=rssError(abY[0:99],yHat1.T)
    error_10=rssError(abY[0:99],yHat10.T)

    print("error_01,error_1,error_10  ",error_01,error_1,error_10)



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



if __name__ == "__main__":
	#result_of_standRegres()
	#lwlrTestPlot()
    rssError_result()


