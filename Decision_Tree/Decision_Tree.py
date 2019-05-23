#-*- coding: utf-8 -*-



"""
k-近邻算法可以完成很多分类任务,但是它最大的缺点就是无法给出数据的内在含义,决策树的主要优势就在于数据形式非常容易理解
决策树的一个重要任务是为了数据中所蕴含的知识信息,因此决策树可以使用不熟悉的数据集合,并从中提取出一决策树的构造系列规则,在这些机器根据数据集创建规则时,就是机器学习的过程。专家系统中经常使用决策
树,而且决策树给出结果往往可以匹敌在当前领域具有几十年工作经验的人类专家

优点:计算复杂度不高,输出结果易于理解,对中间值的缺失不敏感,可以处理不相关特
征数据。
缺点:可能会产生过度匹配问题。 
适用数据类型:数值型和标称型。

在划分数据集之前之后信息发生的变化称为信息增益,知道如何计算信息增益,我们就可以
计算每个特征值划分数据集获得的信息增益,获得信息增益最高的特征就是最好的选择。
在可以评测哪种数据划分方式是最好的数据划分之前,我们必须学习如何计算信息增益。集
合信息的度量方式称为香农熵或者简称为熵,这个名字来源于信息论之父克劳德·香农。


决策树分类器就像带有终止块的流程图,终止块表示分类结果。开始处理数据集时,我们首
先需要测量集合中数据的不一致性,也就是熵,然后寻找最优方案划分数据集,直到数据集中的
所有数据属于同一分类。ID3算法可以用于划分标称型数据集。构建决策树时,我们通常采用递
归的方法将数据集转化为决策树。一般我们并不构造新的数据结构,而是使用Python语言内嵌的
数据结构字典存储树节点信息。

"""

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
    for featVec in dataSet:            #the the number of unique elements and their occurance  # 将当前实例的标签存储，即每一行数据的最后一个数据代表的是标签
        currentLabel = featVec[-1]     # myDat [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
        #print("currentLabel",currentLabel)
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1    # 为所有可能的分类创建字典，如果当前的键值不存在，则扩展字典并将当前键值加入字典。每个键值都记录了当前类别出现的次数。
    #print("labelCounts",labelCounts)
    shannonEnt = 0.0
    #print("labelCounts",labelCounts)
    for key in labelCounts:                             # 对于 label 标签的占比，求出 label 标签的香农熵
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2) # log base 2, 以2为底求对数
    return shannonEnt


# 将指定特征的特征值等于 value 的行剩下列作为子数据集 / 按照给定特征划分数据集  / 当我们按照某个特征划分数据集时,就需要将所有符合要求的元素抽取出来
def splitDataSet(dataSet, axis, value):         # 待划分的数据集、划分数据集的特征、需要返回的特征的值
    retDataSet = []                             # 创建新的list对象
    for featVec in dataSet:
    	#print("featVec",featVec)
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]     #chop out axis used for splitting  将符合特征的数据抽取出来
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)  # the difference between 'extend' and 'append'
    return retDataSet


# 我们将对每个特征划分数据集的结果计算一次信息熵,然后判断按照哪个特征划分数据集是最好的划分方式。
# choose the best split based on max entropy   选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1                   #the last column is used for the labels   # 求第一行有多少列的 Feature, 最后一列是label列嘛
    #print("numFeatures",numFeatures)
    baseEntropy = calcShannonEnt(dataSet)               # 数据集的原始信息熵
    bestInfoGain = 0.0;                                 # 最优的信息增益值
    bestFeature = -1                                    #最优的Featurn编号
    # 第1个 for 循环遍历数据集中的所有特征。使用列表推导(List Comprehension)来创建新的列表,将数据集中所有第i个特征值或者所有可能存在的值写入这个新list中
    for i in range(numFeatures):        #iterate over all the features  2
        featList = [example[i] for example in dataSet]#create a list of all the examples of this feature  # 获取对应的feature下的所有数据
        #print("featList",featList)
        uniqueVals = set(featList)       #get a set of unique values    # 获取剔重后的集合，使用set对list数据进行去重  创建唯一的分类标签列表
        #print("uniqueVals",uniqueVals)
        newEntropy = 0.0
        # 遍历当前特征中的所有唯一属性值,对每个特征划分一次数据集
        for value in uniqueVals:                                        # 遍历某一列的value集合，计算该列的信息熵 
            # 计算每种划分方式的信息熵
            subDataSet = splitDataSet(dataSet, i, value)                # 遍历当前特征中的所有唯一属性值，对每个唯一属性值划分一次数据集，计算数据集的新熵值，并对所有唯一特征值得到的熵求和
            #print("value",value)
            #print("subDataSet",subDataSet)
            prob = len(subDataSet)/float(len(dataSet))                  # 计算概率
            newEntropy += prob * calcShannonEnt(subDataSet)             # 计算信息熵  #计算新的香农熵的时候使用的是子集
        infoGain = baseEntropy - newEntropy     #calculate the info gain; ie reduction in entropy
        if (infoGain > bestInfoGain):       # 计算最好的信息增益 compare this to the best gain so far
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
# 递归结束的条件是:程序遍历完所有划分数据集的属性,或者每个分支下的所有实例都具有相同的分类。如果所有实例具有相同的分类,则得到一个叶子节点或者终止块。任何到达叶子节点的数据必然属于叶子节点的分类,

def createTree(dataSet,labels):                         # 数据集和标签列表
    classList = [example[-1] for example in dataSet]
    # 如果数据集只有1列，那么最初出现label次数最多的一类，作为结果
    # 类别完全相同则停止继续划分
    if classList.count(classList[0]) == len(classList): 
        return classList[0]#stop splitting when all of the classes are equal
    # 第二个停止条件：使用完了所有特征，仍然不能将数据集划分成仅包含唯一类别的分组。
    # 遍历完所有特征时返回出现次数最多的
    if len(dataSet[0]) == 1: #stop splitting when there are no more features in dataSet
        return majorityCnt(classList)

    # 选择最优的列，得到最优列对应的label含义
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]                 # 获取label的名称
    myTree = {bestFeatLabel:{}}                      # 初始化myTree
    # 注：labels列表是可变对象，在PYTHON函数中作为参数时传址引用，能够被全局修改
    # 所以这行代码导致函数外的同名变量被删除了元素，造成例句无法执行，提示'no surfacing' is not in list
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]  # 得到列表包含的所有属性值
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
    featIndex = featLabels.index(firstStr)   # 将标签字符串转换为索引
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


    #熵越高,则混合的数据也越多,我们可以在数据集中添加更多的分类,观察熵是如何变化的
    #myDat[0][-1]='maybe'
    #print ( calcShannonEnt(myDat) )  #1.37095059445  熵越高，混合的数据越多
    #print ( splitDataSet(myDat,0,1))
    #print ( chooseBestFeatureToSplit(myDat) )

    myTree=createTree(myDat,labels)
    print ("myTree",myTree)

    dot = Digraph()
    plotTree(myTree)
    dot.view()

################################################################################
    # 使用决策树预测隐形眼睛类型
    with open('lenses.txt') as f:
        lenses = [item.strip().split('\t') for item in f.readlines()]
        lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
        lensesTree = createTree(lenses, lensesLabels)

    dot = Digraph()
    plotTree(lensesTree)
    dot.view()

