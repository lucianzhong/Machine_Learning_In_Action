#-*- coding: utf-8 -*-
from numpy import *
from math import log
import operator    
from os import listdir
import matplotlib
import matplotlib.pyplot as plt
from graphviz import Digraph

"""
决策树:
判定鱼类和非鱼类
根据以下 2 个特征，将动物分成两类：鱼类和非鱼类。
特征：
不浮出水面是否可以生存
是否有脚蹼

"""

def createDataSet():
    dataSet = [ [1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'],  [0, 1, 'no'], [0, 1, 'no'] ]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels

#计算香农熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)                    # 求list的长度，表示计算参与训练的数据量
    #print("numEntries",numEntries)
    labelCounts = {}                             #字典数据结构，键为label
    for featVec in dataSet: #the the number of unique elements and their occurance  # 将当前实例的标签存储，即每一行数据的最后一个数据代表的是标签
        currentLabel = featVec[-1] # myDat [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
        #print("currentLabel",currentLabel)
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1    # 为所有可能的分类创建字典，如果当前的键值不存在，则扩展字典并将当前键值加入字典。每个键值都记录了当前类别出现的次数。
    #print("labelCounts",labelCounts)
    shannonEnt = 0.0
    #print("labelCounts",labelCounts)
    for key in labelCounts:                             # 对于 label 标签的占比，求出 label 标签的香农熵
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2) #log base 2
    return shannonEnt


# 将指定特征的特征值等于 value 的行剩下列作为子数据集。
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
    	#print("featVec",featVec)
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]     #chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)  # the difference between 'extend' and 'append'
    return retDataSet


# choose the best split based on max entropy   选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1                   #the last column is used for the labels   # 求第一行有多少列的 Feature, 最后一列是label列嘛
    #print("numFeatures",numFeatures)
    baseEntropy = calcShannonEnt(dataSet)               # 数据集的原始信息熵
    bestInfoGain = 0.0;                                 # 最优的信息增益值
    bestFeature = -1                                    #最优的Featurn编号

    for i in range(numFeatures):        #iterate over all the features  2
        featList = [example[i] for example in dataSet]#create a list of all the examples of this feature  # 获取对应的feature下的所有数据
        #print("featList",featList)
        uniqueVals = set(featList)       #get a set of unique values    # 获取剔重后的集合，使用set对list数据进行去重
        #print("uniqueVals",uniqueVals)
        newEntropy = 0.0
        for value in uniqueVals:                                        # 遍历某一列的value集合，计算该列的信息熵 
            subDataSet = splitDataSet(dataSet, i, value)                # 遍历当前特征中的所有唯一属性值，对每个唯一属性值划分一次数据集，计算数据集的新熵值，并对所有唯一特征值得到的熵求和
            #print("value",value)
            #print("subDataSet",subDataSet)
            prob = len(subDataSet)/float(len(dataSet))                  # 计算概率
            newEntropy += prob * calcShannonEnt(subDataSet)             # 计算信息熵  #计算新的香农熵的时候使用的是子集
        infoGain = baseEntropy - newEntropy     #calculate the info gain; ie reduction in entropy
        if (infoGain > bestInfoGain):       #compare this to the best gain so far
            bestInfoGain = infoGain         #if better than current best, set to best
            bestFeature = i
    return bestFeature                      #returns an integer

#有的时候数据集已经处理了所有属性，但是类标签依旧不是唯一的，这种情况通常采用多数表决的方式
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]



# 如果数据集的最后一列的第一个值出现的次数=整个集合的数量，也就说只有一个类别，就只直接返回结果就行
# 第一个停止条件：所有的类标签完全相同，则直接返回该类标签。
# count() 函数是统计括号中的值在list中出现的次数
def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    # 如果数据集只有1列，那么最初出现label次数最多的一类，作为结果
    if classList.count(classList[0]) == len(classList): 
        return classList[0]#stop splitting when all of the classes are equal
    # 第二个停止条件：使用完了所有特征，仍然不能将数据集划分成仅包含唯一类别的分组。
    if len(dataSet[0]) == 1: #stop splitting when there are no more features in dataSet
        return majorityCnt(classList)

    # 选择最优的列，得到最优列对应的label含义
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]                 # 获取label的名称
    myTree = {bestFeatLabel:{}}                      # 初始化myTree
    # 注：labels列表是可变对象，在PYTHON函数中作为参数时传址引用，能够被全局修改
    # 所以这行代码导致函数外的同名变量被删除了元素，造成例句无法执行，提示'no surfacing' is not in list
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]       #copy all of labels, so trees don't mess up existing labels
        # 遍历当前选择特征包含的所有属性值，在每个数据集划分上递归调用函数createTree()
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree                      


def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()
    
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)
    


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


#测试算法：使用决策树执行分类
def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


if __name__ == "__main__":
    myDat,labels=createDataSet()
    #print("myDat",myDat)
    #print("labels",labels)


    print ( calcShannonEnt(myDat) )  #0.9709505944546686,

    #print ("splitDataSet(myDat, 1, 1)",splitDataSet(myDat, 1, 1))
    #print ("splitDataSet(myDat, 0, 0)",splitDataSet(myDat, 0, 0))


    #myDat[0][-1]='maybe'
    #print ( calcShannonEnt(myDat) )  #1.37095059445  熵越高，混合的数据越多
    #print ( splitDataSet(myDat,0,1))
    #print ( chooseBestFeatureToSplit(myDat) )

    myTree=createTree(myDat,labels)
    print ("myTree",myTree)

    dot = Digraph()
    plotTree(myTree)
    dot.view()

    # 使用决策树预测隐形眼睛类型
    with open('lenses.txt') as f:
        lenses = [item.strip().split('\t') for item in f.readlines()]
        lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
        lensesTree = createTree(lenses, lensesLabels)

    dot = Digraph()
    plotTree(lensesTree)
    dot.view()

