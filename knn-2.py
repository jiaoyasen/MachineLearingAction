import numpy as np
import matplotlib.pyplot as plt
import operator



def file2matrix(filename):
    file = open(filename)
    arrayOfLines = file.readlines()
    numberOfLines = len(arrayOfLines)
    returnMat = np.zeros([numberOfLines, 3])
    classLabelVector = []
    index = 0
    for line in arrayOfLines:
        line = line.strip()
        tempLine = line.split("\t")
        returnMat[index][:3] = tempLine[:3]
        classLabelVector.append(tempLine[-1])
        index += 1
    classLabelVector = np.array(classLabelVector)
    return returnMat, classLabelVector


# ## The Classifier

# In[2]:


def classify0(inX, dataSet, labels, k):
    if inX.ndim == 1:
        if inX.shape != dataSet.shape[1]:
            assert ("The shape of TestData is unproperty!")
    elif inX.ndim == 0:
        assert ("The shape of TestData is unproperty!")
    else:
        if inX.shape[1] != dataSet.shape[1]:
            assert ("The shape of TestData is unproperty!")
    inX_result = []
    for in_row in inX:
        diffMat = dataSet - in_row
        sqDiffMat = np.power(diffMat, 2)
        sqDistances = np.sum(sqDiffMat, axis=1)
        distances = np.power(sqDistances, 0.5)
        sortedDistIndicies = distances.argsort()
        classCount = {}
        for i in np.arange(k):
            voteLabel = labels[sortedDistIndicies[i]]
            classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
        inX_result.append(sortedClassCount[0][0])
    return inX_result


# ## The Nomalization

# In[ ]:


def autoNorm(dataSet):
    data_Max = np.max(dataSet)
    data_Min = np.min(dataSet)
    ranges = data_Max - data_Min
    normDataSet = np.zeros(np.shape(dataSet))
    normDataSet = dataSet - data_Min
    normDataSet = normDataSet / ranges
    return normDataSet, ranges, data_Min


# ## The Tester

# In[ ]:


def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    testVecsIndex = np.random.choice(np.arange(m), numTestVecs, replace=False)
    trainVecsIndex = np.array([]).astype(int)
    for i in np.arange(m):
        if i not in testVecsIndex:
            trainVecsIndex = np.append(trainVecsIndex,i)
    trainData = normMat[trainVecsIndex]
    trainDataLabel = datingLabels[trainVecsIndex]
    testData = normMat[testVecsIndex]
    testDataLabel = datingLabels[testVecsIndex]
    classifierResult = classify0(testData, trainData, trainDataLabel, 3)
    for i in np.arange(numTestVecs):
        print("The classifier came back with {},the real answer is {}".format(classifierResult[i], testDataLabel[i]))
        if (classifierResult[i] != testDataLabel[i]): errorCount += 1.0
    print("The total error rate is :{}".format(errorCount / float(numTestVecs)))


if __name__ == "__main__":
    datingClassTest()



