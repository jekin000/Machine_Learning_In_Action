import math
import operator

def createDataSet():
	group = [[1,1,'yes'] ,[1,1,'yes'] ,[1,0,'no'] ,[0,1,'no'],[0,1,'no']]
	labels = ['no surfacing','flippers']
	return group,labels


def calcShannonEnt(dataSet):
	m = len(dataSet)
	labelCounts = {}
	for featVec in dataSet:
		currentLabel = featVec[-1]
		if currentLabel  not in labelCounts.keys():
			labelCounts[currentLabel] = 1
		else:
			labelCounts[currentLabel] += 1

	se = 0.0
	for key in labelCounts.keys():
		prob = float(labelCounts[key])/m
		se -= prob * math.log(prob,2)	
	return se

def splitDataSet(dataSet,axis,value):
	retDataSet = []
	for featVec in dataSet:
		if featVec[axis] == value:
			reduceFeatVec = featVec[:axis]
			reduceFeatVec = featVec[axis+1:]
			retDataSet.append(reduceFeatVec)
	return retDataSet

def chooseBestFeatureToSplit(dataSet):
	numFeatures = len(dataSet[0]) - 1 
	baseEntropy = calcShannonEnt(dataSet)
	bestInfoGain = 0.0; bestFeature = -1
	newEntropy = 0.0
	for i in range(numFeatures):
		featList = [example[i] for example in dataSet]
		uniqueVals = set(featList)
		for value in uniqueVals:
			subDataSet = splitDataSet(dataSet,i,value)
			prop = len(subDataSet) / float(len(dataSet))
			newEntropy += prop * calcShannonEnt(subDataSet)
		infoGain = baseEntropy - newEntropy
		if (infoGain > bestInfoGain):
			bestInfoGain = infoGain
			bestFeature  = i
	return bestFeature

def majorityCnt(classList):
	classCount = {}
	for key in classList:
		if key not in classCount.keys():
			classCount[key] = 1
		else:
			classCount[key] += 1
	sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
	return sortedClassCount[0][0]
	
def createTree(dataSet,labels):
	# 'yes' or 'no'
	classList = [example[-1] for example in dataSet]

	# all result is same labels	
	if (classList.count(classList[0]) == len(classList)):
		# one 'yes' or one 'no'
		return classList[0]
	# the result, for example is:  ['yes','no','no'], surface and flipper col has been used.
	if (len(dataSet[0]) == 1):
		return majorityCnt(classList)

	bestFeat = chooseBestFeatureToSplit(dataSet)
	bestFeatLabel = labels[bestFeat]
	myTree = {bestFeatLabel:{}}
	del(labels[bestFeat])	
	featValues = [example[bestFeat] for example in dataSet]
	uniqueVals = set(featValues)
	# 1 or 0
	for value in uniqueVals:
		#copy list
		subLabels = labels[:]
		myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
	return myTree

def classify(inputTree,featLabels,testVec):
	ftLabel = inputTree.keys()[0]
	ftDvide = inputTree[ftLabel]
	#list.index
	featIndex = featLabels.index(ftLabel)

	#do decision
	result = ftDvide.get(testVec[featIndex])
	if type(result).__name__ == 'dict':
		classifyLabel = classify(result,featLabels,testVec)
	else:
		classifyLabel = result
	return classifyLabel

def storeTree(inputTree,filename):
	import pickle
	fw = open(filename,'w')
	pickle.dump(inputTree,fw)
	fw.close()

def grabTree(filename):
	import pickle
	fr = open(filename)
	return pickle.load(fr)







