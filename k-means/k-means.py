#-*- coding: utf-8 -*-
from numpy import *
import operator    
from os import listdir
import matplotlib
import matplotlib.pyplot as plt

from numpy import *

# 从文本中构建矩阵，加载文本文件，然后处理
def loadDataSet(fileName):      #general function to parse tab -delimited floats  # 通用函数，用来解析以 tab 键分隔的 floats（浮点数）
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list ( map(float,curLine) ) #map all elements to float()     # 映射所有的元素为 float（浮点数）类型
        dataMat.append(fltLine)
    return dataMat


#计算欧式距离
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2))) #la.norm(vecA-vecB)


#构建一个包含k个随机质心的集合
#为给定数据集构建一个包含 k 个随机质心的集合。随机质心必须要在整个数据集的边界之内，这可以通过找到数据集每一维的最小和最大值来完成。
#然后生成 0~1.0 之间的随机数并通过取值范围和最小值，以便确保随机点在数据的边界之内。
def randCent(dataSet, k):
    n = shape(dataSet)[1]
    #print("n",n)
    centroids = mat(zeros((k,n)))#create centroid mat
    for j in range(n):#create random cluster centers, within bounds of each dimension
        minJ = min(dataSet[:,j])        
        rangeJ = float(max(dataSet[:,j]) - minJ)                    # 范围 = 最大值 - 最小值
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1))      # 随机生成   [:,0] the first colume
    return centroids


# 算法会创建k个质心，然后将每个点分配到最近的质心，再重新计算质心。重复迭代，直到数据点的簇分配结点不再改变直到质心不再改变
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]                   # 行数
    clusterAssment = mat(zeros((m,2)))      #create mat to assign data points  to a centroid, also holds SE of each point
    centroids = createCent(dataSet, k)      # 创建质心，随机k个质心
    clusterChanged = True 
    while clusterChanged:
        clusterChanged = False
        for i in range(m):                  #循环每一个数据点并分配到最近的质心中去
            minDist = inf;
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])          # 计算数据点到质心的距离
                if distJI < minDist:                                    # 如果距离比 minDist（最小距离）还小，更新 minDist（最小距离）和最小质心的 index（索引）
                    minDist = distJI;
                    minIndex = j
            if clusterAssment[i,0] != minIndex: clusterChanged = True   # 簇分配结果改变
            clusterAssment[i,:] = minIndex,minDist**2                   # 更新簇分配结果为最小质心的 index（索引），minDist（最小距离）的平方
        #print (centroids)
        for cent in range(k):                                                   # re-calculate centroids  # 更新质心
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]       #get all the point in this cluster,   # 获取该簇中的所有点
            centroids[cent,:] = mean(ptsInClust, axis=0)                        #assign centroid to mean,# 将质心修改为簇中所有点的平均值，mean 就是求平均值的
    return centroids, clusterAssment


'''
二分k-均值：
	将所有的点看成一个簇
	当簇的数目小于k时
	对于每一个簇：
		计算总误差
		在给定的簇上面进行k-均值聚类（k=2）
		计算该簇一分为二之后的总误差
	选择使得误差最小的那个簇进行划分
'''

# 二分 KMeans 聚类算法, 基于 kMeans 基础之上的优化，以避免陷入局部最小值
def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    centroid0 = mean(dataSet, axis=0).tolist()[0]   # 质心初始化为所有数据点的均值
    centList =[centroid0] #create a list with one centroid   # 初始化只有 1 个质心的 list
    for j in range(m):#calc initial Error   # 计算所有数据点到初始质心的距离平方误差
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2    # 保存每个数据点的簇分配结果和平方误差
    while (len(centList) < k):  # 当质心数量小于 k 时
        lowestSSE = inf
        for i in range(len(centList)):   # 对每一个质心
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]#get the data points currently in cluster i   # 获取当前簇 i 下的所有数据点
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)    # 将当前簇 i 进行二分 kMeans 处理
            sseSplit = sum(splitClustAss[:,1])#compare the SSE to the currrent minimum   # 将二分 kMeans 结果中的平方和的距离进行求和
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])   # 将未参与二分 kMeans 分配结果中的平方和的距离进行求和
            print ("sseSplit, and notSplit: ",sseSplit,sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE:   # 总的（未拆分和已拆分）误差和越小，越相似，效果越优化，划分的结果更好（注意：这里的理解很重要，不明白的地方可以和我们一起讨论）
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        # 找出最好的簇分配结果 
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) #change 1 to 3,4, or whatever   # 调用二分 kMeans 的结果，默认簇是 0,1. 当然也可以改成其它的数字
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit  # 更新为最佳质心
        print ('the bestCentToSplit is: ',bestCentToSplit )
        print ('the len of bestClustAss is: ', len(bestClustAss) )
        # 更新质心列表
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]#replace a centroid with two best centroids    # 更新原质心 list 中的第 i 个质心为使用二分 kMeans 后 bestNewCents 的第一个质心
        centList.append(bestNewCents[1,:].tolist()[0])  # 添加 bestNewCents 的第二个质心
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss#reassign new clusters, and SSE  # 重新分配最好簇下的数据（质心）以及SSE
    return mat(centList), clusterAssment








if __name__ == "__main__":
    dataMat=mat( loadDataSet('testSet.txt') )
    #print("dataMat",(dataMat))
    print ( randCent(dataMat,2) )

    """
    plt.figure()
    plt.plot(dataMat[:,0],dataMat[:,1],'.')
    plt.show()
    """




    #print ( distEclud(dataMat[0],dataMat[1]) )
    #print("dataMat[0]",dataMat[0])

    myCentroids,clusterAssing=kMeans(dataMat,4)
    print("myCentroids",myCentroids)
    print("clusterAssing",clusterAssing)

    plt.figure()
    plt.plot(dataMat[:,0],dataMat[:,1],'.')
    plt.plot(myCentroids[:,0],myCentroids[:,1],'ro')
    plt.show()

    #dataMat2=mat( loadDataSet('testSet2.txt') )
    #biKmeans(dataMat2,3)


