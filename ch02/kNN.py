import numpy as np
import operator

def createDataSet():
	group = np.array([[1.0,1.1] ,[1.0,1.0] ,[0,0] ,[0,0.1]])
	labels = ['A','A','B','B']
	return group,labels


def classify0(inX,dataSet,labels,k):
	#numpy.ndarray.shape[0] means rows count
	#numpy.ndarray.shape[s] means cols count
	dataSetSize = dataSet.shape[0]	

	#numpy.tile(m1,parm) repeat m1 parm counts
	diffMat = np.tile(inX,(dataSetSize,1)) - dataSet

	sqDiffMat = diffMat**2
	sqDistances = sqDiffMat.sum(axis=1)
	distances = sqDistances**0.5
	# argsort => [1.48660687,1.41421356,0.0.1] -> [0,0.1,1.41421356,1.48660687] -> index=[2,3,1,0]
	sortedDistIndicies = distances.argsort()

	classCount = {}
	for i in range(k):
		# get [2,3,1,0]'s label
		voteIlabel = labels[sortedDistIndicies[i]]
		# dict.get(key,default)
		classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
	#operator.itemgetter = sorted(sortedClassCount[1]) sorted by item[1]
	sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
	return sortedClassCount[0][0]


