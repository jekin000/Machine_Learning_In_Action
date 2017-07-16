import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

decisionNode = dict(boxstyle='sawtooth',fc='0.8')
leafNode     = dict(boxstyle='round4',fc='0.8')
arrow_args   = dict(arrowstyle='<-')

def plotNode(nodeTxt,centerPt, parentPt, nodeType):
	createPlot.ax1.annotate(nodeTxt
		,xy=parentPt,xycoords='axes fraction'
		,xytext=centerPt,textcoords='axes fraction'
		,va='center'
		,ha='center'
		,bbox=nodeType
		,arrowprops=arrow_args
		)

def createPlotTest():
	fig = plt.figure(1,facecolor='white')
	fig.clf()
	createPlot.ax1 = plt.subplot(111,frameon=False)
	plotNode('Decision Node',(0.5,0.1),(0.1,0.5),decisionNode)
	plotNode('Leaf Node ',(0.8,0.1),(0.3,0.8),leafNode)
	fig.savefig('test.png')
	return


def retrieveTree(i):
	listOfTrees = [
				{'no surfacing':{0:'no',1:{'flippers':{0:'no',1:'yes'}}}}
				,{'no surfacing':{0:'no',1:{'flippers':{0:{'head':{0:'no',1:'yes'}},1:'no'}}}}
			]
	return listOfTrees[i]

def getNumLeafs(myTree):
	numLeafs = 0
	firstStr = myTree.keys()[0]
	secondDict = myTree[firstStr]
	for key in secondDict.keys():
		if type(secondDict[key]).__name__ == 'dict':
			numLeafs += getNumLeafs(secondDict[key])
		else:
			numLeafs += 1
	return numLeafs

def getTreeDepth(myTree):
	depth = 0
	firstStr = myTree.keys()[0]
	secondDict = myTree[firstStr]
	for key in secondDict.keys():
		if type(secondDict[key]).__name__ == 'dict':
			thisdepth = 1+getTreeDepth(secondDict[key])
		else:
			thisdepth = 1
		if thisdepth > depth:
			depth = thisdepth

	return depth

def plotMidText(cntrPt,parentPt,txtString):
	xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
	yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
	createPlot.ax1.text(xMid,yMid,txtString)

def plotTree(myTree,parentPt,nodeTxt):
	numLeafs = getNumLeafs(myTree)
	depth    = getTreeDepth(myTree)
	firstStr = myTree.keys()[0] 
	cntrPt = (plotTree.xOff+(1.0+float(numLeafs))/2.0/plotTree.totalW,plotTree.yOff)
	plotMidText(cntrPt,parentPt,nodeTxt)
	plotNode(firstStr,cntrPt,parentPt,decisionNode)
	secondDict = myTree[firstStr]
	plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
	for key in secondDict.keys():
		if type(secondDict[key]).__name__ == 'dict':
			plotTree(secondDict[key],cntrPt,str(key))
		else:
			plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
			plotNode(secondDict[key],(plotTree.xOff,plotTree.yOff),cntrPt,leafNode)
			plotMidText((plotTree.xOff,plotTree.yOff),cntrPt,str(key))
	plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD

def createPlot(inTree):
	fig = plt.figure(1,facecolor='white')
	fig.clf()
	#unknown
	axprops = dict(xticks=[],yticks=[])
	createPlot.ax1 = plt.subplot(111,frameon=False, **axprops)
	plotTree.totalW = float(getNumLeafs(inTree))	
	plotTree.totalD = float(getTreeDepth(inTree))
	#xOff/yOff record the node which already drawed
	plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;
	plotTree(inTree,(0.5,1.0),'')
	fig.savefig('tree.png')


