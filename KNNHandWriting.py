import numpy as np
import matplotlib.pyplot as plt
import operator
import knn
from os import listdir


def img2vecotr(filename):
    returnVect = np.zeros([1,1024])
    file = open(filename)
    for i in np.arange(32):
        lineStr = file.readline()
        for j in np.arange(32):
            returnVect[0, 32*i + j ] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros([m,1024])
    for i in np.arange(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vecotr('trainingDigits/{}'.format(fileNameStr))
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in np.arange(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vecotr('testDigits/{}'.format(fileNameStr))
        classifierResult = knn.knn_classify(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: {}, the real answer is: {}".format(classifierResult, classNumStr))
        if(classifierResult != classNumStr ):errorCount += 1.0
    print("the total number of errors is:{}".format(errorCount))
    print("the total error rate is:{}".format(errorCount/float(mTest)))


handwritingClassTest()
