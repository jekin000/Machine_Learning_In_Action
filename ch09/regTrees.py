from numpy import *

def binSplitDataSet(dataset,feature,value):
	mat0 = dataset[nonzero(dataset[:,feature] >  value)[0],:]
	mat1 = dataset[nonzero(dataset[:,feature] <= value)[0],:]
	return mat0,mat1


