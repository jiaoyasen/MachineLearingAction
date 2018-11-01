import numpy as np
import matplotlib.pyplot as plt
import operator
import mpl_toolkits.mplot3d


def file2matrix(filename):
    '''
     根据文件路径从磁盘读取数据文件
     将读取的数据文件进行处理
     生成对用格式的数据
     原始格式为feat1\tfeat2\tfeat3\tlabel
     返回数据矩阵，标签向量
    :param filename:
    :return: returnMat, classLabelVector_np
    '''
    file = open(filename)#利用句柄读写文件
    arrayOfLines = file.readlines()#多行读取
    numberOfLines = len(arrayOfLines)
    returnMat = np.zeros([numberOfLines, 3])
    classLabelVector = []
    index = 0
    for line in arrayOfLines:
        line = line.strip()
        temp = line.split("\t")
        returnMat[index, :3] = temp[0:3]
        index += 1
        classLabelVector.append(temp[-1])
        classLabelVector_np = np.array(classLabelVector)#标签向量
    scatter(returnMat,classLabelVector_np,None)
    file.close()
    return returnMat, classLabelVector_np


def scatter(dataSet , dataSet_labels, testSet):
    '''
    由于数据具有3个特征，用3D图形显示数据分布情况
    书上是2维的，既然都是3个特征的数据，干脆高大上一点
    :param dataSet:
    :param dataSet_labels:
    :param testSet:
    :return:
    '''
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    color_index = 0
    color_array = ['r','g','b','k','w','y','m','c']
    label_array = set(dataSet_labels)
    print(label_array)
    for item in label_array:
        label_index = np.where(dataSet_labels == str(item))
        print(label_index)
        ax.scatter(dataSet[label_index,0], dataSet[label_index,1], dataSet[label_index,2],c=color_array[color_index])
        color_index += 1
        if color_index == len(color_array)-1:
            color_index = 0
#    ax.scatter(testSet[0], testSet[1], testSet[2], c='k')
    plt.show()


def normalization(dataSet):
    '''
    数据归一化
    (x-x_min)/(x_max-x_min)
    :param dataSet:
    :return:
    '''
    dataSet_min = np.min(dataSet)
    dataSet_max = np.max(dataSet)
    return (dataSet - dataSet_min)/(dataSet_max - dataSet_min)


def knn_classify(inX, dataSet, labels, k):
    '''
    KNN核心算法，使用欧式距离计算K个近邻，并确定类别最多的类型
    :param inX:
    :param dataSet:
    :param labels:
    :param k:
    :return:
    '''
    if inX.ndim == 1: #数据格式判断，k不能大于数据数
        if inX.shape != dataSet.shape[1]:
            assert ("The shape of TestData is unproperty!")
    elif inX.ndim == 0:
        assert ("The shape of TestData is unproperty!")
    else:
        if inX.shape[1] != dataSet.shape[1]:
            assert ("The shape of TestData is unproperty!")


    diffMat = inX - dataSet
    sqDiffMat = np.power(diffMat,2)
    sqDistance = np.sum(sqDiffMat, axis = 1)
    distances = np.power(sqDistance, 0.5)#计算欧式距离
    sortDistanceIndicies = np.argsort(distances)
    classCount = {}
    if k > len(labels):
        k = len(labels)
    for i in np.arange(k):
        votelLabel = labels[sortDistanceIndicies[i]]
        classCount[votelLabel] = classCount.get(votelLabel,0) + 1
    sortClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse=True)#根据value对字典进行排序
    return sortClassCount[0][0]


def datingClassTest(fileName , n_Fold):
    '''
    k-fold cv对数据集进行Test，并计算平均错误率，评价标准很多，后续应该逐渐补全
    书上的程序只是提取前10%的数据做测试，这样结果的可信度不高，在这里进行了改进
    :param fileName:
    :param n_Fold:
    :return:
    '''
    hoRatio = 1.0 / n_Fold
    datingDataMat, datingLabels = file2matrix(fileName)#读取数据
    normMat = normalization(datingDataMat)#标准化数据
    m = normMat.shape[0]
    fold = m / n_Fold#将数据按fold进行划分
    numTestVecs = int(m*hoRatio)
    errorCount = 0
    total_errorCount = 0
    shuffleIndex = np.arange(m)
    np.random.shuffle(shuffleIndex)#随机打乱数据
    normMat = normMat[shuffleIndex]
    datingLabels = datingLabels[shuffleIndex]
    for i in np.arange(n_Fold):#每一fold进行计算
        TestVecs = normMat[int(fold * i):int(fold * (i+1))]
        TestVecs_label = datingLabels[int(fold * i):int(fold * (i+1))]
        TrainVecs = np.vstack((normMat[0:int(fold * i)], normMat[int(fold * (i+1)):]))#数据拆分成训练数据和测试数据，应该有更好的办法，暂时没想到
        TrainVecs_label = np.hstack((datingLabels[0:int(fold * i)], datingLabels[int(fold * (i+1)):]))
        for j in np.arange(TestVecs.shape[0]):
            classifierResult = knn_classify(TestVecs[j], TrainVecs, TrainVecs_label, 3)#对每个测试数据进行预测
            print(
                "The classifier came back with: {}, the real answer is: {}".format(classifierResult, TestVecs_label[j]))
            if (classifierResult != TestVecs_label[j]): errorCount += 1
        total_errorCount += errorCount
        print("No.{} fold errorCountNum = {}".format(i+1, errorCount))
        print("No.{} fold TrainVecsNum = {}".format(i+1,TrainVecs.shape[0]))
        print("No.{} fold TestVecsNum = {}".format(i+1,TestVecs.shape[0]))
        print("No.{} fold numTestVecs = {}".format(i+1, numTestVecs))
        print("No.{} fold error rate is :{}".format(i+1, errorCount / float(numTestVecs)))
        errorCount = 0
    avgErrorRate = total_errorCount / m
    print("Total Vecs Num = {}".format(m))
    print("Total error Vec Num = {}".format(total_errorCount))
    print("Avg error rate = {}".format(avgErrorRate))#显示平均错误率


if __name__ == "__main__":
    datingClassTest('datingTestSet.txt',5)