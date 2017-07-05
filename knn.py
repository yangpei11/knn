# *-* coding:utf-8 *-*
import numpy as np
import operator
import matplotlib
import matplotlib.pyplot as plt
from os import listdir

def createDataSet():
	group = np.array([ [1.0, 1.1], [1.0, 1.0], [0,0], [0,0.1] ])
	Labels = ['A', 'A', 'B', 'B']
	return group, Labels
	
	
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()     
    classCount={}          
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
		
	
def file2matrix(filename):
	f = open(filename)
	Lines = f.readlines()
	numberOfLines = len(Lines)
	returnMat = np.zeros( (numberOfLines ,3) ) #特征向量
	classLabelVector = []
	index = 0
	for line in Lines:
		line = line.strip()
		vectorList = line.split('\t')
		returnMat[index, :] = vectorList[0:3]
		classLabelVector.append( int(vectorList[-1]) )
		index += 1
	
	return returnMat, classLabelVector
#vectorMat, labelsOfVector = file2matrix('datingTestSet2.txt')	
def draw(vectorMat, labelsOfVector):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(vectorMat[:, 0], vectorMat[:, 1], 15*np.array(labelsOfVector), 15*np.array(labelsOfVector))
	plt.title('test')
	plt.xlabel('X')
	plt.ylabel('Y')
	plt.show()
	
	
def autoNorm(dataSet):
	minVals = dataSet.min(0)
	maxVals = dataSet.max(0)
	ranges = maxVals - minVals
	normDataSet = np.zeros( np.shape(dataSet) )
	m = dataSet.shape[0]
	normDataSet = dataSet - np.tile(minVals, (m,1))
	normDataSet = normDataSet/np.tile(ranges, (m,1))
	return normDataSet, ranges, minVals
	
def datingClassTest():
	hoRatio= 0.1 #取前10%为测试集
	datingDataMat, datingLabels = file2matrix('datingTestSet2.txt') #取出原始特征向量和标记
	normMat, ranges, minVals = autoNorm(datingDataMat) #归一化操作
	m = normMat.shape[0]
	numTestVecs = int(m*hoRatio)
	errorCount = 0.0
	for i in range(numTestVecs):
		testResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
		if (testResult != datingLabels[i]):
			errorCount += 1.0
	print errorCount
	print numTestVecs
	print(u"错误率是 %f" %(errorCount/float(numTestVecs)) )
	
def img2vector(filename):
	returnVect = np.zeros( (1,1024) )
	f = open(filename)
	for i in range(32):
		line = f.readline()
		for j in range(32):
			returnVect[0, 32*i+j] = int( line[j] )
	return returnVect
	
def handwritingClassTest():
	#提取特征向量和标记
	hwLabels = []
	tranningFileList = listdir('trainingDigits')
	m = len(tranningFileList)
	tranningMat = np.zeros( (m, 1024))
	
	for i in range(m):
		filename = tranningFileList[i]
		tranningMat[i, :] = img2vector('trainingDigits/%s' %filename)
		hwLabels.append( int(tranningFileList[i].split('_')[0]) )
	
	testFileList = listdir('testDigits')
	errorCount = 0.0
	mTest = len(testFileList)
	for i in range(mTest):
		#取得测试的向量和Labels
		filename = testFileList[i]
		testVector = img2vector('testDigits/%s' %filename)
		label = int(filename.split('_')[0] )
		returnLabel = classify0(testVector, tranningMat, hwLabels, 11)
		if returnLabel != label:
			errorCount += 1
	print( u'错误率为%f' %(errorCount/float(mTest)) )
	
	
""" simple_test
dataSet, labels  = createDataSet()
print classify0([1.0, 1.0], dataSet, labels, 3)   
"""

"""  draw
vectorMat, labelsOfVector = file2matrix('datingTestSet2.txt')
draw(vectorMat, labelsOfVector)
"""

""" 约会
datingClassTest()
"""

"""  手写"""
handwritingClassTest()






	



	












	
	
