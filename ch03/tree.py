import math

def createDataSet():
	group = [[1,1,'yes'] ,[1,1,'yes'] ,[1,0,'no'] ,[0,1,'no'],[0,1,'no']]
	labels = ['surface','foot']
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
	for i in range(numFeatures):
		featList = [example[i] for example in dataSet]
		uniqueVals = set(featList)
	return
