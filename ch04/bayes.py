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

def trainNB1(trainMatrix,trainCategory):
	numTrainDocs = len(trainMatrix)
	pAbusive = sum(trainCategory) / float(numTrainDocs)

	numWords = len(trainMatrix[0])
	# to forbid a0 * a1 * a2 when a1=0
	p0Num = np.ones(numWords)
	p1Num = np.ones(numWords)
	p0Denom = 2.0
	p1Denom = 2.0

	for i in range(numTrainDocs):
		if trainCategory[i] == 1:
			p1Num += trainMatrix[i]
			p1Denom += sum(trainMatrix[i])
		else:
			p0Num += trainMatrix[i]
			p0Denom += sum(trainMatrix[i])

	#p(W|c1) = p(w1,w2,w3|c1) = p1Vect = e.g. [0.5,0.25,0.25]
	#p(w1,w2,w3|c1) = p(w1|c1)*p(w2|c1)*p(w3|c1) = 0.5*.0.25*0.25 --> mybe too small and not exactly so
	#bayes = p(ci|W) = p(W|ci)*p(ci)/p(W) = p(W|ci)*p(ci) ; to forbid too small, we use log
	#bayes = log(p(W|ci) * p(ci)) = log(p(W|ci)) + log(p(ci)) = log(p(w1|ci)) + log(p(w2|ci)) + log(p(w3|ci)) + log(p(ci))
	#so , we use log here, p1Vect = [log(p(w1|c1)),log(p(w2|c1)), log(p(w3|c1))]
	p0Vect = np.log(p0Num/p0Denom)
	p1Vect = np.log(p1Num/p1Denom)

	return p0Vect,p1Vect,pAbusive
