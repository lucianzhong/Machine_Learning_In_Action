#-*- coding: utf-8 -*-
from numpy import *


def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine)) #map all elements to float()
        dataMat.append(fltLine)
    return dataMat



'''
找到最佳的待切分特点：
	如果该节点不能再分，就将该节点存为叶节点
	执行二元切分
	在左子树调用createTree()方法
	在右子树调用createTree()方法
'''

#输入：数据集合，待切分的特征，该特征的某个值
#切分得到两个子集
def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:]   #nonzero:返回非零元素的目录,返回值为元组， 两个值分别为两个维度， 包含了相应维度上非零元素的目录值
    mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:]
    return mat0,mat1


def regLeaf(dataSet):#returns the value used for each leaf
    return mean(dataSet[:,-1])

def regErr(dataSet):
    return var(dataSet[:,-1]) * shape(dataSet)[0]  #方差var



'''
对每个特征：
	对每个特征值：
		将数据切分成两份
		计算切分的误差
		如果当前误差小于当前最小误差，将当前切分设置为最小误差的切分
返回最佳切分的特征和阀值
'''
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    tolS = ops[0]; tolN = ops[1]
    #if all the target variables are the same value: quit and return value
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1: #exit cond 1
        return None, leafType(dataSet)
    m,n = shape(dataSet)
    #the choice of the best feature is driven by Reduction in RSS error from mean
    S = errType(dataSet)
    bestS = inf; bestIndex = 0; bestValue = 0
    for featIndex in range(n-1):
        for splitVal in set((dataSet[:,featIndex].T.A.tolist())[0]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS: 
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    #if the decrease (S-bestS) is less than a threshold don't do the split
    if (S - bestS) < tolS: 
        return None, leafType(dataSet) #exit cond 2
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):  #exit cond 3
        return None, leafType(dataSet)
    return bestIndex,bestValue#returns the best feature to split on
                              #and the value used for that split



'''
找到最佳的待切分特点：
	如果该节点不能再分，就将该节点存为叶节点
	执行二元切分
	在左子树调用createTree()方法
	在右子树调用createTree()方法
'''
#创建树：数据集，leafType:建立叶子节点函数，errType:计算误差函数，ops其他参数
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):#assume dataSet is NumPy Mat so we can array filtering
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)#choose the best split
    if feat == None: return val #if the splitting hit a stop condition return val
    retTree = {}  #dictionary   dict = {'a': 1, 'b': 2, 'b': '3'}
    retTree['spInd'] = feat   #bestIndex
    retTree['spVal'] = val    #bestValue
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree



#树的节点过多位过拟合，需进行裁剪pruning
'''
函数prune()的伪代码：
	基于已知的树的切分测试数据：
		如果存在任一子集是一棵树，则在该子集上递归剪枝过程
		计算当前两个叶节点合并后的误差
		计算不合并的误差
		如果合并会降低误差的话，就将叶节点合并
'''

#当前节点是否为叶节点
def isTree(obj):
    return (type(obj).__name__=='dict')

#
def getMean(tree):
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0

def prune(tree, testData):
    if shape(testData)[0] == 0: return getMean(tree) #if we have no test data collapse the tree
    if (isTree(tree['right']) or isTree(tree['left'])):#if the branches are not trees try to prune them
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']): tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']): tree['right'] =  prune(tree['right'], rSet)
    #if they are now both leafs, see if we can merge them
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(power(lSet[:,-1] - tree['left'],2)) +\
            sum(power(rSet[:,-1] - tree['right'],2))
        treeMean = (tree['left']+tree['right'])/2.0
        errorMerge = sum(power(testData[:,-1] - treeMean,2))
        if errorMerge < errorNoMerge: 
            print ("merging")
            return treeMean
        else: return tree
    else: return tree





#模型树
#将数据格式化为目标变量Y和自变量X
def linearSolve(dataSet):   #helper function used in two places
    m,n = shape(dataSet)
    X = mat(ones((m,n))); Y = mat(ones((m,1)))#create a copy of data with 1 in 0th postion
    X[:,1:n] = dataSet[:,0:n-1]; Y = dataSet[:,-1]#and strip out Y
    xTx = X.T*X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws,X,Y


def modelLeaf(dataSet):#create linear model and return coeficients
    ws,X,Y = linearSolve(dataSet)
    return ws

def modelErr(dataSet):
    ws,X,Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat,2))


def binSplitDataSet_example():
	testMat=mat(eye(4))
	print("testMat   ",  testMat)
	mat0,mat1=binSplitDataSet(testMat,1,0.5)
	print("mat0  ",mat0)
	print("mat1  ",mat1)




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







if __name__ == "__main__":
	binSplitDataSet_example()

	#myDat=loadDataSet('ex00.txt')
	#myMat=mat(myDat)
	#print ( createTree(myMat) )

	#myDat=loadDataSet('ex0.txt')
	#myMat=mat(myDat)
	#print ( createTree(myMat) )

	#myDat2=loadDataSet('ex2.txt')
	#myMat2=mat(myDat2)
	#myTree2=createTree(myMat2,ops=(0,1))

	#myDatTest=loadDataSet('ex2test.txt')
	#myMatTest=mat(myDatTest)

	#prune(myTree2,myMatTest)

	#myMat2=mat( loadDataSet('exp2.txt') )
	#myTree2=createTree( myMat2, leafType=modelLeaf, errType=modelErr, ops=(1,10))
	#print(myTree2)

