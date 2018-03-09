import numpy as np
import os

'''
Bayes basic:
1. doc1 -> wordlist1;
   doc2 -> wordlist2:
   docN -> wordlistN;

2. transfer wordlists to vocab dictionary;
3. split dictionary to TrainningSet and TestingSet;
4. CALC TrainningSet & TestingSet to TrainningVector & TestingVector;
5. CALC P0,P1,PAbusive by TrainningVector;
6. CALC predict by TesttingVector, and classify it by test label

===>localwords
BCC world = p1 		(actually, we would use forum city, such as New York)
BCC england = p0        (such as Boston)
give a new word-vector, predict it is world or england
 

'''
gSpamTrainingParm = {'p0Vect':np.array([])
	,'p1Vect':np.array([])
	,'pAbusive':0.0
	,'vocabList':[]}
gConfig = {"debugLevel":False}

def setConfigDebugLevelEnable():
	gConfig['debugLevel'] = True
	return

def setConfigDebugLevelDisable():
	gConfig['debugLevel'] = False
	return
def getConfig():
	conf = gConfig
	return conf

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
	
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


def setOfWords2Vec(vocabList,inputSet):
	returnVec = [0] * len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] = 1
	return returnVec


#For better comprehension
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

#enhance to trainNB0
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

def printDebug(s):
	if gConfig['debugLevel']:
		print s
	return None

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
	p1 = np.sum(vec2Classify*p1Vec) + np.log(pClass1)
	p0 = np.sum(vec2Classify*p0Vec) + np.log(1-pClass1)
	printDebug('p1='+str(p1)+';p0='+str(p0))
	if p1 > p0:
		return 1
	else:
		return 0

def testNB():
	docs,labels =  loadDataSet()
	words = createVocabList(docs)
	wordsVec = []
	for eachdoc in docs:
		wordsVec.append(setOfWords2Vec(words,eachdoc))

	p0Vect,p1Vect,pAbusive = trainNB1(np.array(wordsVec),np.array(labels))
	newdoc1 = ['stupid','garbage']
	newVec1 = np.array(setOfWords2Vec(words,newdoc1))
	# I buying him is classify as 1....
	newdoc0 = ['love','my','dalmation']
	newVec0 = np.array(setOfWords2Vec(words,newdoc0))
	print newdoc1,'Classified as:',classifyNB(newVec1,p0Vect,p1Vect,pAbusive)
	print newdoc0,'Classified as:',classifyNB(newVec0,p0Vect,p1Vect,pAbusive)

def textParse(bigString):
	wordlen = 2 # 0.046
	#wordlen = 3  # 0.075
	#wordlen = 4  # 0.062
	import re
	#split by char which is not 0~9,a~z; it seems that _ not include
        listOfTokens = re.split(r'\W*',bigString)	
	listOfTokens = [tok.lower() for tok in listOfTokens if len(tok)>wordlen]
	return listOfTokens

def chunks(l,n):
	for i in xrange(0,len(l),n):
		yield l[i:i+n]

def readEmailDir(edir,maxcnt,dirlabel): 
	totalEmails = os.listdir(edir)
	docs 	= []
	labels 	= []
	names   = []
	emails = list(chunks(totalEmails,maxcnt))[0] 
	for e in emails:
		doc = []
		f = open(edir+'/'+e,'r')
		docs.append(textParse(f.read()))
		labels.extend([dirlabel])
		names.append(os.path.join(edir,e))
		f.close()
	return docs,labels,names

def spamTestMany():
	finalcnt = 10
	cnt = 100	
	finalErrRates = [] 
	for i in range(finalcnt):
		errrates = []
		for i in range(cnt):
			errrates.append(spamTest()) 
		finalErrRates.append(reduce(lambda x,y:x+y,errrates)/len(errrates))
	return reduce(lambda x,y:x+y,finalErrRates)/len(finalErrRates)

def getSpamTrainningParm():
	parm = gSpamTrainingParm
	return parm 

def predictSpam(eml):
	f = open(eml,'r')
	doc = textParse(f.read())
	f.close()
	vec = setOfWords2Vec(gSpamTrainingParm['vocabList'],doc)
	return classifyNB(np.array(vec)
		,gSpamTrainingParm['p0Vect']
		,gSpamTrainingParm['p1Vect']
		,gSpamTrainingParm['pAbusive'])


def checkMailVocab(eml):
	f = open(eml,'r')
	doc = textParse(f.read())
	f.close()
	inCnt = 0
	for word in doc:
		if word in gSpamTrainingParm['vocabList']:
			inCnt += 1
	return float(inCnt)/len(doc)

def getDocCnt(spamdir,hamdir):
	spams = os.listdir(spamdir)
	hams  = os.listdir(hamdir)	
	return min(len(spams),len(hams))*2
#######################################
def calcBayes():
	spamdir = 'spam'
	hamdir  = 'ham'

	#spam = 1378
	#ham  = 2949
	#max totalDocCnt = 1378
	totalDocCnt = getDocCnt(spamdir,hamdir) 
	# test
	#totalDocCnt = 600, 0.14333
	totalDocCnt = 100
	testSetCnt =  totalDocCnt/2
	docs    = []
	labels  = []
	names   = []

	d,l,n = readEmailDir(spamdir,testSetCnt,1)
	docs.extend(d)
	labels.extend(l)
	names.extend(n)
	print '==========load spam sample============'

	d,l,n = readEmailDir(hamdir,testSetCnt,0)
	docs.extend(d)
	labels.extend(l)
	names.extend(n)
	print '==========load ham sample============='

	vocabList = createVocabList(docs)
	print '==========create Vocalblist==========='

	trainningSet = range(len(docs))
	testSet = []
	testlabels = []
	testnames  = []
	for i in range(testSetCnt):
		import random
		testidx = int(random.uniform(0,len(trainningSet)))
		testSet.append(trainningSet[testidx])
		testlabels.append(labels[testidx])
		testnames.append(names[testidx])
		del(trainningSet[testidx])		
		del(labels[testidx])
		del(names[testidx])
	
	print '==========prepared training/testing set==========='

	print '==========calc train vect==========='
	trainlabels = labels	
	trainnames  = names
	trainWordVec = []
	
	printPercentCnt = 1
	for i in trainningSet:
		trainWordVec.append(bagOfWords2VecMN(vocabList,docs[i]))
		if printPercentCnt%4 == 0:
			print '{0:.5}% '.format(printPercentCnt/(len(trainningSet)*1.0) * 100.0)
		printPercentCnt += 1

	print '==========calc test vect==========='

	testWordVec  = []
	printPercentCnt = 1
	for i in testSet:
		testWordVec.append(bagOfWords2VecMN(vocabList,docs[i]))
		if printPercentCnt%4 == 0:
			print '{0:.5}% '.format(printPercentCnt/(len(testSet)*1.0) * 100.0)
		printPercentCnt += 1

	print '==========training==========='
	p0Vect,p1Vect,pAbusive = trainNB1(np.array(trainWordVec),np.array(trainlabels))

	# for further test, it will open the independant API for long term
	gSpamTrainingParm['p0Vect'] = p0Vect
	gSpamTrainingParm['p1Vect'] = p1Vect
	gSpamTrainingParm['pAbusive'] = pAbusive
	gSpamTrainingParm['vocabList'] = vocabList

	errcnt = 0
	tlidx  = 0
	#For CSDMC2010, tag 0 = SPAM, tag 1 = normal
	fpcnt  = 0
	fncnt  = 0
	print '==========classifing==========='
	for vec in  testWordVec:
		forecast = classifyNB(np.array(vec),p0Vect,p1Vect,pAbusive)
		if forecast != testlabels[tlidx]:
			if forecast == 0:
				fpcnt += 1
			else:
				fncnt += 1
			print '{name} fail.'.format(name=testnames[tlidx])
			#print vec
			#print '\n'
			errcnt += 1
		tlidx += 1
	return float(errcnt)/testSetCnt,float(fpcnt)/testSetCnt,float(fncnt)/testSetCnt

	
if __name__ == '__main__': 
	errrate,fprate,fnrate =  calcBayes()
	print 'total fault rate = {all}%; fp rate = {fp}%, fn rate={fn}%'.format(all=errrate*100.0
		,fp=fprate*100.0
		,fn=fnrate*100.0)

	#In the end, consider we use the module to predict new eml which have many new words.
	#Because our vector is maked base on vocablist, and it is do not have new words,
	#so, the new words will not contain in vector, and in other words, they no use for predict.
