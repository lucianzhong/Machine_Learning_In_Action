#-*- coding: utf-8 -*-
from numpy import *
import numpy as np
import operator    
from os import listdir
import matplotlib
import matplotlib.pyplot as plt
from time import sleep

#读取数据
def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

#数据可视化
def print_data():
    dataArr,labelArr=loadDataSet('testSet.txt')
    #print("dataArr",dataArr)
    datMat=mat(dataArr)
    #print("datMat",datMat)
    plt.figure()
    plt.plot(datMat[:,0],datMat[:,1],'.')
    plt.show()

#############################################################################################################################
#简化版SMO算法
'''
  创建一个alpha向量并将其初始化为0向量
  当迭代次数小于最大迭代次数时（外循环）
        对数据集中的每个数据向量（内循环）：
            如果改数据向量可以被优化：
                随机选择另外一个数据向量
                同时优化这两个向量
                如果两个向量都不能被优化，推出内循环
    如果所有向量都没有被优化，增加迭代数目，继续下一次循环
'''

#随机选择alpha
# i是第一个alpha的下标， m是所有alpha的数目
def selectJrand(i,m):
    j=i #we want to select any j not equal to i
    while (j==i):
        j = int(random.uniform(0,m))
    return j

#修剪alpha
#调整大于H，或者小于L的alpha值
def clipAlpha(aj,H,L):
    if aj > H: 
        aj = H
    if L > aj:
        aj = L
    return aj

#Platt SMO中的外循环确定要优化的alpha对
#简化版本在数值集合上遍历每一个alpha，然后在剩下的alpa集合中随机选择另一个alpha,构建alpha对

# toler:容错率 
#常数C用于控制“最大化间隔”和“保证大部分点点函数间隔小于1.0”,  C - 松弛变量

def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = mat(dataMatIn);                                 #转换为numpy的mat存储
    labelMat = mat(classLabels).transpose()
    b = 0;                                                       #初始化b参数，统计dataMatrix的维度
    m,n = shape(dataMatrix)  #(100,2)
    alphas = mat(zeros((m,1)))                                   #初始化alpha参数，设为0
    iter = 0                                                     #初始化迭代次数
    while (iter < maxIter):                                       #最多迭代matIter次
        alphaPairsChanged = 0
        for i in range(m):
             #                       100*100                      100*2   *  2*1      
            fXi = float( multiply(alphas,labelMat).T * ( dataMatrix*dataMatrix[i,:].T )  ) + b  #(100,1) ##步骤1：计算误差Ei 预测分类结果
            Ei = fXi - float(labelMat[i])  #if checks if an example violates KKT conditions #和实际结果比较，计算误差
            #优化alpha，更设定一定的容错率
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):  #如果误差较大，对alpha优化
                j = selectJrand(i,m)                                                                               #随机选择另一个与alpha_i成对优化的alpha_j
                fXj = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b                        #步骤1：计算误差Ej,   预测分类结果
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy();                                                                      #保存更新前的aplpha值，使用深拷贝
                alphaJold = alphas[j].copy();
                #保证alpha[j]属于区间[0,C]
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])                #步骤2：计算上下界L和H
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])

                if L==H: print ("L==H");continue  # continue,for循环结束，进行下一轮循环
                #步骤3：计算eta
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T  #alpha[j]的最优修改量
                if eta >= 0: print ("eta>=0"); continue  #进行下一轮循环
                #步骤4：更新alpha_j
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                #步骤5：修剪alpha_j
                alphas[j] = clipAlpha(alphas[j],H,L)
                #步骤6：更新alpha_i
                if (abs(alphas[j] - alphaJold) < 0.00001):print ("j not moving enough"); continue # the change of alpha
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])#update i by the same amount as j,the update is in the oppostie direction
                #步骤7：更新b_1和b_2
                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                #步骤8：根据b_1和b_2更新b
                if (0 < alphas[i]) and (C > alphas[i]): b = b1
                elif (0 < alphas[j]) and (C > alphas[j]): b = b2
                else: b = (b1 + b2)/2.0   #设置常数项b
                #统计优化次数
                alphaPairsChanged += 1
                #打印统计信息
                print ("iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
        #更新迭代次数
        if (alphaPairsChanged == 0): iter += 1
        else: iter = 0
        print ("iteration number: %d" % iter)
    #print("shape(alphas)   ",shape(alphas))
    #print("shape(dataMatrix[i,:].T  ",shape(dataMatrix[i,:].T))  #(2,1)
    return b,alphas


def showClassifer(dataMat, w, b):
    #绘制样本点
    data_plus = []                                  #正样本
    data_minus = []                                 #负样本
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus)              #转换为numpy矩阵
    data_minus_np = np.array(data_minus)            #转换为numpy矩阵
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1], s=30, alpha=0.7)   #正样本散点图
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1], s=30, alpha=0.7) #负样本散点图
    #绘制直线
    x1 = max(dataMat)[0]
    x2 = min(dataMat)[0]
    a1, a2 = w
    b = float(b)
    a1 = float(a1[0])
    a2 = float(a2[0])
    y1, y2 = (-b- a1*x1)/a2, (-b - a1*x2)/a2
    plt.plot([x1, x2], [y1, y2])
    #找出支持向量点
    for i, alpha in enumerate(alphas):
        if abs(alpha) > 0:
            x, y = dataMat[i]
            plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='red')
    plt.show()

def get_w(dataMat, labelMat, alphas):
    alphas, dataMat, labelMat = np.array(alphas), np.array(dataMat), np.array(labelMat)
    w = np.dot((np.tile(labelMat.reshape(1, -1).T, (1, 2)) * dataMat).T, alphas)
    return w.tolist()


#############################################################################################################################3


#完整版的Platt SMO算法

#利用核函数，将数据映射到高位空间
def kernelTrans(X, A, kTup): #calc the kernel or transform data to a higher dimensional space
    m,n = shape(X)
    K = mat(zeros((m,1)))
    if kTup[0]=='lin': K = X * A.T   #linear kernel
    elif kTup[0]=='rbf':
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow*deltaRow.T
        K = exp(K/(-1*kTup[1]**2)) #divide in NumPy is element-wise not matrix like Matlab
    else: raise NameError('Houston We Have a Problem -- That Kernel is not recognized')
    return K

#数据对象，方便访问
class optStruct:
    def __init__(self,dataMatIn, classLabels, C, toler, kTup):  # Initialize the structure with the parameters 
        self.X = dataMatIn                            #数据矩阵
        self.labelMat = classLabels                   #数据标签
        self.C = C                                    #松弛变量
        self.tol = toler                              #容错率
        self.m = shape(dataMatIn)[0]                  #数据矩阵行数
        self.alphas = mat(zeros((self.m,1)))          #根据矩阵行数初始化alpha参数为0   
        self.b = 0                                    #初始化b参数为0
        self.eCache = mat(zeros((self.m,2)))          #根据矩阵行数初始化虎误差缓存，第一列为是否有效的标志位，第二列为实际的误差E的值。
        self.K = mat(zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)

# 计算误差, 计算E值
# oS - 数据结构,   k - 标号为k的数据,   Ek - 标号为k的数据误差
def calcEk(oS, k):
    fXk = float(multiply(oS.alphas,oS.labelMat).T*oS.K[:,k] + oS.b)  
    Ek = fXk - float(oS.labelMat[k])
    return Ek


# 选择第二个alpha/内循环的alpha
#  i - alpha_i的索引值,    m - alpha参数个数,    j - alpha_j的索引值
#  i - 标号为i的数据的索引值,   oS - 数据结构,      Ei - 标号为i的数据误差
#   Returns:  j, maxK - 标号为j或maxK的数据的索引值,      Ej - 标号为j的数据误差
def selectJ(i, oS, Ei):                                  # this is the second choice -heurstic, and calcs Ej
    maxK = -1; maxDeltaE = 0; Ej = 0                     #初始化
    oS.eCache[i] = [1,Ei]                                #根据Ei更新误差缓存
    validEcacheList = nonzero(oS.eCache[:,0].A)[0]       #返回误差不为0的数据的索引值
    if (len(validEcacheList)) > 1:                       #有不为0的误差
        for k in validEcacheList:                        #遍历,找到最大的Ek
            if k == i: continue                          #不计算i,浪费时间don't calc for i, waste of time
            Ek = calcEk(oS, k)                           #计算Ek
            deltaE = abs(Ei - Ek)                        #计算|Ei-Ek|
            if (deltaE > maxDeltaE):                     #找到maxDeltaE
                maxK = k; 
                maxDeltaE = deltaE; 
                Ej = Ek
        return maxK, Ej                                  #返回maxK,Ej
    else:                                                #没有不为0的误差  in this case (first time around) we don't have any valid eCache values
        j = selectJrand(i, oS.m)                         #随机选择alpha_j的索引值
        Ej = calcEk(oS, j)                               #计算Ej
    return j, Ej

#更新E值
def updateEk(oS, k):                                      #  计算Ek,并更新误差缓存 after any alpha has changed update the new value in the cache
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]


# 第二个alpha的选择中的启发式算法,优化的SMO算法,  
# i - 标号为i的数据的索引值, oS - 数据结构,   Returns: 1 - 有任意一对alpha值发生变化,   0 - 没有任意一对alpha值发生变化或变化太小
def innerL(i, oS):
    Ei = calcEk(oS, i)                                      #步骤1：计算误差Ei
    #优化alpha,设定一定的容错率
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJ(i, oS, Ei)                           #使用内循环启发方式2选择alpha_j,并计算Ej   this has been changed from selectJrand and it is the difference between the simple SMO
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();           #保存更新前的aplpha值，使用深拷贝
        if (oS.labelMat[i] != oS.labelMat[j]):                                      #步骤2：计算上下界L和H
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H: print ("L==H"); return 0
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j]                               #步骤3：计算eta, changed for kernel
        if eta >= 0: print ("eta>=0"); return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta                                #步骤4：更新alpha_j
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)                                  #步骤5：修剪alpha_j
        updateEk(oS, j)                                                             #更新Ej至误差缓存,  added this for the Ecache
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): print ("j not moving enough"); return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])    #步骤6：更新alpha_i,  update i by the same amount as j
        updateEk(oS, i)                                                             #更新Ei至误差缓存,  added this for the Ecache #the update is in the oppostie direction
        #步骤7：更新b_1和b_2
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]- oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        #步骤8：根据b_1和b_2更新b
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1 + b2)/2.0
        return 1
    else: return 0

#完整的线性SMO算法
#完整的外循环Platt SMO代码
def smoP(dataMatIn, classLabels, C, toler, maxIter,kTup=('lin', 0)):    #full Platt SMO
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler, kTup)
    iter = 0
    entireSet = True; alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:   #go over all
            for i in range(oS.m):        
                alphaPairsChanged += innerL(i,oS)
                print ("fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged) )
            iter += 1
        else:#go over non-bound (railed) alphas
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print ("non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)  )
            iter += 1
        if entireSet: entireSet = False #toggle entire set loop
        elif (alphaPairsChanged == 0): entireSet = True  
        print ("iteration number: %d" % iter)
    return oS.b,oS.alphas


#基于alpha计算w，得到超平面
def calcWs(alphas,dataArr,classLabels):
    X = mat(dataArr); labelMat = mat(classLabels).transpose()
    m,n = shape(X)
    w = zeros((n,1))
    for i in range(m):
        w += multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w






#在测试中使用核函数
def testRbf(k1=1.3):
    dataArr,labelArr = loadDataSet('testSetRBF.txt')
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1)) #C=200 important
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    svInd=nonzero(alphas.A>0)[0]
    sVs=datMat[svInd] #get matrix of only support vectors
    labelSV = labelMat[svInd];
    print ( "there are %d Support Vectors" % shape(sVs)[0]  )
    m,n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print ( "the training error rate is: %f" % (float(errorCount)/m)  )
    dataArr,labelArr = loadDataSet('testSetRBF2.txt')
    errorCount = 0
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    m,n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1    
    print  ( "the test error rate is: %f" % (float(errorCount)/m) )   

##################################################################################################################################



#基于SVM的手写数字识别

def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def loadImages(dirName):
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)           #load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9: hwLabels.append(-1)
        else: hwLabels.append(1)
        trainingMat[i,:] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels    

def testDigits(kTup=('rbf', 10)):
    dataArr,labelArr = loadImages('digits/trainingDigits')
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    svInd=nonzero(alphas.A>0)[0]
    sVs=datMat[svInd] 
    labelSV = labelMat[svInd];
    print ("there are %d Support Vectors" % shape(sVs)[0] )
    m,n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print ("the training error rate is: %f" % (float(errorCount)/m) )
    dataArr,labelArr = loadImages('digits/testDigits')
    errorCount = 0
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    m,n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1    
    print ( "the test error rate is: %f" % (float(errorCount)/m) )



#####################################################################################################################################
if __name__ == "__main__":
    #print_data()
    dataMat, labelMat = loadDataSet('testSet.txt')
    b,alphas = smoSimple(dataMat, labelMat, 0.6, 0.001, 40)
    w = get_w(dataMat, labelMat, alphas)
    showClassifer(dataMat, w, b)

    #print("dataArr",dataArr)

    # the SMO simple example
    #b,alphas=smoSimple(dataArr,labelArr,0.6,0.001,40)
    #print("b  ",b)
    #print("alphas[alphas>0]  ",alphas[alphas>0])


    #b,alpha=smoP(dataArr,labelArr,0.6,0.001,40)

    #testRbf(k1=1.3)

    #testDigits(kTup=('rbf', 10))