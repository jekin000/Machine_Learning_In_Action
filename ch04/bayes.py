def loadDataSet():
	postingList = [
		['my','dog','has','flea','problems','help','please']
		,['maybe','not','take','him','to','dog','park','stupid']
		,['my','dalmation','is','so','cute','I','love','him']
		,['stop','posting','stupid','worthless','garbage']
		,['mr','licks','ate','my','steak','how','to','stop','him']
	]

	classVec = [0,1,0,1,0,1]
	return postingList,classVec

def createVocabList(dataSet):
	vocabSet = set([])
	for doc in dataSet:
		vocabSet = vocabSet | set(doc)	
	return  list(vocabSet)
	


def setOfWords2Vec(vocabList,inputSet):
	returnVec = [0] * len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] = 1
	return returnVec



