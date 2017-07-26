import numpy as np

def loadDataSet():
	postingList = [
		['my','dog','has','flea','problems','help','please']
		,['maybe','not','take','him','to','dog','park','stupid']
		,['my','dalmation','is','so','cute','I','love','him']
		,['stop','posting','stupid','worthless','garbage']
		,['mr','licks','ate','my','steak','how','to','stop','him']
		,['quit','buying','worthless','dog','food','stupid']
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


def trainNB0(trainMatrix,trainCategory):
	#calc doc's p1, p0 = 1 - p1
	numTrainDocs = len(trainMatrix)
	pAbusive = sum(trainCategory) / float(numTrainDocs)

	#calc p(w|ci) , p(w|ci) =  sum_word_matrix(docs)/sum_word_count(ci)
	numWords = len(trainMatrix[0])
	p0Num = np.zeros(numWords)
	p1Num = np.zeros(numWords)
	p0Denom = 0.0
	p1Denom = 0.0
	for i in range(numTrainDocs):
		if trainCategory[i] == 1:
			p1Num += trainMatrix[i]
			p1Denom += sum(trainMatrix[i])
		else:
			p0Num += trainMatrix[i]
			p0Denom += sum(trainMatrix[i])

	p0Vect = p0Num/p0Denom
	p1Vect = p1Num/p1Denom 

	return p0Vect,p1Vect,pAbusive

