#-*- coding: utf-8 -*-
from numpy import *
import numpy as np
from math import log
import operator    
from os import listdir
import matplotlib
import matplotlib.pyplot as plt
from graphviz import Digraph

#加载数据
def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine)) #map all elements to float()
        dataMat.append(fltLine)
    return dataMat

#树回归中再找到分割特征和分割值之后需要将数据进行划分以便构建子树或者叶子节点
#根据给定的特征编号和特征值对数据集进行分割
def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:]   #nonzero:返回非零元素的目录,返回值为元组， 两个值分别为两个维度， 包含了相应维度上非零元素的目录值
    mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:]
    return mat0,mat1

#生成叶结点
def regLeaf(dataSet):#returns the value used for each leaf
    return np.mean(dataSet[:,-1])
#误差估计函数
def regErr(dataSet):
    return np.var(dataSet[:,-1]) * shape(dataSet)[0]  #方差var



'''
对每个特征：
	对每个特征值：
		将数据切分成两份
		计算切分的误差
		如果当前误差小于当前最小误差，将当前切分设置为最小误差的切分
返回最佳切分的特征和阀值
'''
#选取最佳分割特征和分割值了，这里我们通过找打使得分割后的方差最小的分割点最为最佳分割点 / 找到数据的最佳二元切分方式函数:
 #   dataset: 待划分的数据集
 #   leafType: 创建叶子节点的函数
 #   errType: 计算数据误差的函数
 #   opt: ops - 用户定义的参数构成的元组/回归树参数.
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    tolS = ops[0]                                       #tolS允许的误差下降值,
    tolN = ops[1]                                       #tolN切分的最少样本数
    #如果当前所有值相等,则退出。(根据set的特性)
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1: #exit cond 1
        return None, leafType(dataSet)
    
    m,n = shape(dataSet)                                #统计数据集合的行m和列n
    S = errType(dataSet)                                #默认最后一个特征为最佳切分特征,计算其误差估计
    #分别为最佳误差,最佳特征切分的索引值,最佳特征值
    bestS = inf; bestIndex = 0; bestValue = 0
    for featIndex in range(n-1):                                                    # 遍历所有特征
        for splitVal in set((dataSet[:,featIndex].T.A.tolist())[0]):                # 遍历所有特征值
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)              # 按照当前特征和特征值分割数据
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): continue         # 如果数据少于tolN,则退出
            newS = errType(mat0) + errType(mat1)                                    # 计算误差估计
            if newS < bestS:                                                        # 如果误差估计更小,则更新特征索引值和特征值
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    #如果误差减少不大则退出
    if (S - bestS) < tolS: 
        return None, leafType(dataSet) #exit cond 2
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)                      # 根据最佳的切分特征和特征值切分数据集合
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):                           # 如果切分出的数据集很小则退出
        return None, leafType(dataSet)
    return bestIndex,bestValue                              # bestIndex - 最佳切分特征      bestValue - 最佳特征值



'''
找到最佳的待切分特点：
	如果该节点不能再分，就将该节点存为叶节点
	执行二元切分
	在左子树调用createTree()方法
	在右子树调用createTree()方法
'''
#创建树：数据集，leafType:建立叶子节点函数，errType:计算误差函数，ops其他参数
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):#assume dataSet is NumPy Mat so we can array filtering
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)                 #选择最佳切分特征和特征值
    if feat == None: return val                                                  #如果没有特征,则返回特征值
    retTree = {}                                                                 #回归树 dictionary   dict = {'a': 1, 'b': 2, 'b': '3'}
    retTree['spInd'] = feat   #bestIndex
    retTree['spVal'] = val    #bestValue
    lSet, rSet = binSplitDataSet(dataSet, feat, val)                             #分成左数据集和右数据集
    #创建左子树和右子树
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree


#判断测试输入变量是否是一棵树点
def isTree(obj):
    return (type(obj).__name__=='dict')

#对树进行塌陷处理(即返回树平均值)
def getMean(tree):
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0



#后剪枝
#树的节点过多位过拟合，需进行裁剪pruning
'''
函数prune()的伪代码：
    基于已知的树的切分测试数据：
        如果存在任一子集是一棵树，则在该子集上递归剪枝过程
        计算当前两个叶节点合并后的误差
        计算不合并的误差
        如果合并会降低误差的话，就将叶节点合并
'''
def prune(tree, testData):
    if shape(testData)[0] == 0: return getMean(tree)                                #如果测试集为空,则对树进行塌陷处理
    if (isTree(tree['right']) or isTree(tree['left'])):                             #如果有左子树或者右子树,则切分数据集
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']): tree['left'] = prune(tree['left'], lSet)               #处理左子树(剪枝)
    if isTree(tree['right']): tree['right'] =  prune(tree['right'], rSet)           #处理右子树(剪枝)
    #if they are now both leafs, see if we can merge them
    if not isTree(tree['left']) and not isTree(tree['right']):                      #如果当前结点的左右结点为叶结点
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(power(lSet[:,-1] - tree['left'],2)) + sum(power(rSet[:,-1] - tree['right'],2))                  #计算没有合并的误差
            
        treeMean = (tree['left']+tree['right'])/2.0                                 #计算合并的均值
        errorMerge = sum(power(testData[:,-1] - treeMean,2))                        #计算合并的误差
        if errorMerge < errorNoMerge:                                               #如果合并的误差小于没有合并的误差,则合并
            print ("merging")
            return treeMean
        else: return tree
    else: return tree


##################################################################################################################################

# 模型树
# 模型树的叶节点生成函数,获取标准线性回归系数
def linearSolve(dataSet):   #helper function used in two places
    m,n = shape(dataSet)
    X = mat(ones((m,n)));   #建立两个全部元素为1的(m,n)矩阵和(m,1)矩阵
    Y = mat(ones((m,1)))
    X[:,1:n] = dataSet[:,0:n-1]; Y = dataSet[:,-1]#and strip out Y
    xTx = X.T*X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)                                #求线性回归的回归系数
    return ws,X,Y

#建立模型树叶节点函数
def modelLeaf(dataSet):
    ws,X,Y = linearSolve(dataSet)
    return ws
#模型树平方误差计算函数
def modelErr(dataSet):
    ws,X,Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat,2))


########################################################################################################################################



#实例：数回归和标准回归到比较


#用树回归进行预测

def regTreeEval(model, inDat):
    return float(model)

def modelTreeEval(model, inDat):
    n = shape(inDat)[1]
    X = mat(ones((1,n+1)))
    X[:,1:n+1]=inDat
    return float(X*model)


def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree): return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']): return treeForeCast(tree['left'], inData, modelEval)
        else: return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']): return treeForeCast(tree['right'], inData, modelEval)
        else: return modelEval(tree['right'], inData)
        
def createForeCast(tree, testData, modelEval=regTreeEval):
    m=len(testData)
    yHat = mat(zeros((m,1)))
    for i in range(m):
        yHat[i,0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat


def compare_results():
	trainMat=mat( loadDataSet('bikeSpeedVsIq_train.txt') )
	testMat=mat( loadDataSet('bikeSpeedVsIq_test.txt') )

#####################################################################################################################
# plot the tree
def plotRoot(root, child, id=''):
    id += str(root)
    dot.node(id, str(root))
    if type(child).__name__ == 'dict':
        for key in child:
            dot.node(id+str(key), str(key))
            dot.edge(id, id+str(key))
            plotRoot(key, child[key], id)
    else:
        dot.node(id+str(child), str(child))
        dot.edge(id, id+str(child))

def plotTree(tree):
    for key in tree:
        plotRoot(key, tree[key])





if __name__ == "__main__":

    """
    testMat = np.mat(np.eye(4))
    mat0, mat1 = binSplitDataSet(testMat, 1, 0.5)
    print('原始集合:\n', testMat)
    print('mat0:\n', mat0)
    print('mat1:\n', mat1)
    """


    """
    dataset = loadDataSet('ex0.txt')
    dataset = np.array(dataset)
    # 绘制散点
    plt.scatter(dataset[:, 1], dataset[:, 2])
    plt.show()
    """

    """
    dataset = loadDataSet('ex00.txt')
    dataset = np.array(dataset)
    # 绘制散点
    plt.scatter(dataset[:, 0], dataset[:, 1])
    plt.show()
    """

    """
    myDat = loadDataSet('ex00.txt')
    myMat = np.mat(myDat)
    feat, val = chooseBestSplit(myMat, regLeaf, regErr, (1, 4))
    print(feat)
    print(val)                  #切分的最佳特征为第1列特征，最佳切分特征值为0.48813，这个特征值怎么选出来的？就是根据误差估计的大小，我们选择的这个特征值可以使误差最小化。

    myDat = loadDataSet('ex00.txt')
    myMat = np.mat(myDat)
    print(createTree(myMat))


    train_filename = 'ex2.txt'
    train_Data = loadDataSet(train_filename)
    train_Mat = np.mat(train_Data)
    tree = createTree(train_Mat)
    print(tree)
    test_filename = 'ex2test.txt'
    test_Data = loadDataSet(test_filename)
    test_Mat = np.mat(test_Data)
    print(prune(tree, test_Mat))
    """

    #这种分段的数据，回归树拟合它可是最合适不过了
    trainMat = np.mat( loadDataSet('bikeSpeedVsIq_train.txt') )
    testMat = np.mat( loadDataSet('bikeSpeedVsIq_test.txt') )

    myTree = createTree(trainMat)

    yHat=createForeCast(myTree,testMat[:,0])

    plt.plot(testMat[:,0],yHat,'.')
    plt.plot(testMat[:,0],testMat[:,1],'*')
    plt.show()







    #y=createForeCast(tree, myMat[:,0], modelEval=regTreeEval)


    #plt.plot(x, y, c='r')
    #plt.show()

    """
    dot = Digraph()
    plotTree(tree)
    dot.view()
    """



