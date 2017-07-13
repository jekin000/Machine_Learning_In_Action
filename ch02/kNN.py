import numpy as np
import operator

def createDataSet():
	group = np.array([[1.0,1.1] ,[1.0,1.0] ,[0,0] ,[0,0.1]])
	labels = ['A','A','B','B']
	return group,labels


#kNN
#1.count k most min distance sample
#2.count the these sample's label
#3.return most max count label
def classify0(inX,dataSet,labels,k):
	#numpy.ndarray.shape[0] means rows count
	#numpy.ndarray.shape[s] means cols count
	dataSetSize = dataSet.shape[0]	

	#numpy.tile(m1,parm) repeat m1 parm counts
	#cp inX with dataSet rows
	# for i in inX: inX[i] - dataSet[i]
	diffMat = np.tile(inX,(dataSetSize,1)) - dataSet

	sqDiffMat = diffMat**2
	sqDistances = sqDiffMat.sum(axis=1)
	distances = sqDistances**0.5
	# argsort => [1.48660687,1.41421356,0.0.1] index=[0,1,2,3] -> [0,0.1,1.41421356,1.48660687] -> index=[2,3,1,0]
	sortedDistIndicies = distances.argsort()

	classCount = {}
	for i in range(k):
		# get [2,3,1,0]'s label which is ['B','B','A','A']
		voteIlabel = labels[sortedDistIndicies[i]]
		# dict.get(key,default), dict = {'B':2} if k=2,  dict=['B':2,'A':1] if k=3
		classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
	#operator.itemgetter = sorted(sortedClassCount[1]) sorted by item[1]
	#suppose dict={'A':2,'B':3,'C':4}  after sort , it is dict{'C':4,'B':3,'A':2}
	sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
	return sortedClassCount[0][0]


def file2matrix(filename):
	fr = open(filename) 
	arrayOfLines = fr.readlines()
	numberOfLines = len(arrayOfLines)
	#generate a zero np.array which cols=3 and rows=numberOfLines
	returnMat = np.zeros((numberOfLines,3))
	classLabelVector = []
	index = 0
	for line in arrayOfLines:
		line = line.strip()
		listFromLine = line.split('\t')
	        # [index:] -> index rows, each-row from 0->end	
		# [0:3] from 0, len=3
		returnMat[index:] = listFromLine[0:3]
		#[-1] = last element of list
		classLabelVector.append(int(listFromLine[-1]))
		index = index + 1
	return returnMat,classLabelVector 

def autoNorm(dataSet):
	#min of each col	
	minVals = dataSet.min(0)
	maxVals = dataSet.max(0)
	ranges = maxVals - minVals
	normDataSet = np.zeros(np.shape(dataSet))
	m = dataSet.shape[0]
	#val = each(val - minval)
	normDataSet = dataSet - np.tile(minVals,(m,1))
	#val = each(val/range)
	normDataSet = normDataSet / np.tile(ranges,(m,1))
	return normDataSet,ranges,minVals

def datingClassTest():
	hoRatio = 0.10
	# k=3->5%, k=7->4%, k=15->6%
	k       = 7
	datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
	normMat,ranges,minVals = autoNorm(datingDataMat)
	m = normMat.shape[0]
	numTestVecs = int(m*hoRatio)
	errorCount = 0.0
	
	for i in range(numTestVecs):
		#normMat[i,:]  i means i row; : means all cols 
		#normMat[numTestVecs:m,:]  means from numTestVecs to m-1 row; : means all cols
		#normMat is 1000*3
		#datingLabels is 1000*1
		classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:]\
			,datingLabels[numTestVecs:m],k)
		print "the classifier came back with: %d, the real answer is:%d"\
			% (classifierResult,datingLabels[i])
		if (classifierResult != datingLabels[i]):
			errorCount += 1.0
	print "the total error rate is:%f" % (errorCount/float(numTestVecs))


def classifyPerson():
	k = 7
	resultList = ['not at all','in small doses','in large doses']
	percentTats = float(raw_input("Percentage of time spent playing video games?"))
	ffMiles = float(raw_input("Frequent flier miles earned per year?"))
	iceCream = float(raw_input("Liters of ice cream consumed per year?")) 
	datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
	normMat,ranges,minVals = autoNorm(datingDataMat)
	inArr = np.array([ffMiles,percentTats,iceCream])
	classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,k)
	print "You will probably like this person:" , resultList[classifierResult - 1]
	return

