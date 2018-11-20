import numpy as np


def loadDataSet(fileName):  # 读取数据，老生常谈，就不多说了，按行读，以特定的符号分割数据
    dataMat = [];
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])  # 前两维是数据
        labelMat.append(float(lineArr[2]))  # 第三位的是标签
    return dataMat, labelMat


def selectJrand(i, m):  # i是第一个alpha的下标，m是所有alpha的数目，选择一个不等于i的值
    j = i
    while (j == i):
        j = int(np.random.uniform(0, m))  # 随机生成[0,m)之间的整数
    return j


def clipAlpha(aj, H, L):  # 用于调整alpha的值，使其满足限制条件，在H和L之间
    if aj > H:  # 比H大的就是H
        aj = H
    if L > aj:  # 比L小的就是L
        aj = L
    return aj


class optStruct: #建了一个类用于存储求参数过程中的中间值
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        self.X = dataMatIn#输入数据
        self.labelMat = classLabels#输入数据标签
        self.C = C#参数C
        self.tol = toler#用于限定alpha的阈值
        self.m = np.shape(dataMatIn)[0]#样本数
        self.alphas = np.mat(np.zeros((self.m, 1)))#初始化一个m维列向量，用于初始化alpha的值，均设为0
        self.b = 0#参数b，初始值为0
        self.eCache = np.mat(np.zeros((self.m, 2)))#用于记录alpha的计算中间值的缓存，（m，2），每一个样本有一个元祖，用来存该位置是否可用以及误差
        self.K = np.mat(np.zeros([self.m,self.m]))#用于计算核函数
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:],kTup)#根据kTup元组中的核函数类型以及参数，处理数据


def calcEk(oS, k):#用于计算误差值
    fXk = np.float(np.multiply(oS.alphas, oS.labelMat).T * oS.K[:,k] + oS.b)#核函数形式的预测值计算
    Ek = fXk - float(oS.labelMat[k])#计算误差
    return Ek


def selectJ(i, oS, Ei):#用于选择alphaJ
    maxK = -1;
    maxDeltaE = 0;
    Ej = 0
    oS.eCache[i] = [1, Ei]#将该位置的缓存置为可用，并更新误差值
    validEcacheList = np.nonzero(oS.eCache[:, 0].A)[0]#找到所有可用的alpha值
    if (len(validEcacheList)) > 1:#假如不止一个，就找误差变化最大的
        for k in validEcacheList:
            if k == i: continue
            Ek = calcEk(oS, k)
            deltaE = np.abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k;
                maxDeltaE = deltaE;
                Ej = Ek
        return maxK,Ej#找到了误差变化最大的alphaJ
    else:#假如没有，就随机选一个
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej


def updateEk(oS, k):#更新误差值
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]


def innerL(i, oS):#内循环过程
    Ei = calcEk(oS, i)#对于参数aplhaI，计算误差
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C) or (oS.labelMat[i] * Ei > oS.tol) and (
        oS.alphas[i] > 0)):#判断其是否是违反限制条件最严重的
        j, Ej = selectJ(i, oS, Ei)#如果是，就选alphaJ
        alphaIold = oS.alphas[i].copy()#需要将alpha都存起来，后面会修改以及比较
        alphaJold = oS.alphas[j].copy()
        if (oS.labelMat[i] != oS.labelMat[j]):#yi不等于yj的情况，看alphaJ-alphaI
            L = max(0, oS.alphas[j] - oS.alphas[i])#确定alpha的裁剪边界
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)#yi==yj的情况，看alphaI+alphaI
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H: print("L == H");return 0#如果裁剪边界上下界相同，结束计算
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j]#计算eta，是用来算alphaJ的变化的
        if eta >= 0: print("eta>=0");return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta#调整lphaJ
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)#裁剪alphaJ
        updateEk(oS, j)#更新alpha在缓存中的状态
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):#假如alphaJ没有微小变化，结束计算
            print("j not moving enough");
            return 0
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])#修改alphaI
        updateEk(oS, i)#更新其在缓存中状态
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i,i]  - oS.labelMat[j] * \
                                                                (oS.alphas[j] - alphaJold) * oS.K[i,j]#计算b1
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i,j] - oS.labelMat[j] * \
                                                                (oS.alphas[j] - alphaJold) * oS.K[j,j]#计算b2

        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):#alphaI满足支持向量的条件，b=b1
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):#alphaJ满足支持向量的条件，b=b2
            oS.b = b2
        else:#都不满足
            oS.b = (b1 + b2) / 2.0#就折中
        return 1
    else:
        return 0


def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):#外循环
    oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler,kTup)#对数据进行核函数处理
    iter = 0
    entireSet = True;
    alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):#没到最大循环次数以及（alpha对有修改或是整个数据集还在遍历）进行循环
        alphaPairsChanged = 0#初始化alpha对修改为0
        if entireSet:#没有可更改的alpha对，就对整个数据集遍历，算作一次迭代
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)#对每个样本计算alphaI，看能否找到对应alphaJ更新
                print("fullSet, iter:{} i:{},pairs changed {}".format(iter, i, alphaPairsChanged))
                iter += 1#找到了循环加一
        else:#另一种情况是分析不在限制边界上的样本
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]#找到了不在限制边界上的样本，不是支持向量的那些
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print("non-bound,iter:{} i:{}, pairs changed {}".format(iter, i, alphaPairsChanged))
            iter += 1
        if entireSet:#两种方式交替进行
            entireSet = False
        elif (alphaPairsChanged == 0):#alpha对没有可以修改的，就换另一种方式
            entireSet = True
        print("iteration number: {}".format(iter))
    return oS.b, oS.alphas


def calcWs(alphas, dataArr, classLabels):#计算w的值
    X = np.mat(dataArr);
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(X)
    w = np.zeros([n, 1])
    for i in range(m):
        w += np.multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w


def kernelTrans(X, A, kTup):#核函数变化
    m, n = np.shape(X)
    K = np.mat(np.zeros((m, 1)))
    if kTup[0] == 'lin':#会对kTup元组中存储的核类型以及参数进行读取
        K = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        K = np.exp(K / (-1 * kTup[1] ** 2))
    else:
        raise NameError('Houston We Have a Problem -- That Kernel is not recognized')#这个特别扯，休斯顿发火箭失败
    return K


def testRbf(k1=1.3):#测试rbf核
    dataArr, labelArr = loadDataSet('testSetRBF.txt')
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000,('rbf',k1))
    datMat = np.mat(dataArr);labelMat = np.mat(labelArr).transpose()
    svInd = np.nonzero(alphas.A > 0)[0]#找到大于0的alpha
    sVs = datMat[svInd]#这些是支持向量，其实还应该小于C吧
    labelSV = labelMat[svInd]#获取这些向量的标签
    print("there are {} Support Vetors".format(np.shape(sVs)[0]))
    m,n = np.shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i,:],('rbf',k1))#对数据做核处理
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b#预测值
        if(np.sign(predict)!=np.sign(labelArr[i])):errorCount += 1#预测值与真实值符号不相等，预测错误
    print("the training error rate is:{}".format(float(errorCount)/m))#计算错误率
    dataArr, labelArr = loadDataSet('testSetRBF2.txt')
    errorCount = 0
    datMat = np.mat(dataArr);labelMat = np.mat(labelArr).transpose()
    m,n = np.shape(datMat)
    for i in range(m):#同上的情况，换了数据集
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf',k1))
        predict = kernelEval.T * np.multiply(labelSV,alphas[svInd]) +b
        if np.sign(predict) != np.sign(labelArr[i]):errorCount += 1
    print("the test error rate is :{}".format(float(errorCount)/m))

testRbf()

def img2vecotr(filename):
    returnVect = np.zeros([1,1024])
    file = open(filename)
    for i in np.arange(32):
        lineStr = file.readline()
        for j in np.arange(32):
            returnVect[0, 32*i + j ] = int(lineStr[j])
    return returnVect

def loadImages(dirName):#读取图片文件，同KNN那一章的处理代码
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)#列出目录中的全部文件
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))#图片文件展开为(1,1024)的格式
    for i in range(m):#处理图片文件，解析文件名
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9:hwLabels.append(-1)#二分类，9是负样本，其余的为正样本
        else: hwLabels.append(1)
        trainingMat[i,:] = img2vecotr('%s/%s'%(dirName,fileNameStr))
    return trainingMat,hwLabels

def testDigits(kTup = ('rbf',10)):#同样的计算过程，不赘述了
    dataArr,labelArr = loadImages('trainingDigits')
    b,alphas = smoP(dataArr,labelArr, 200, 0.0001, 10000, kTup)
    datMat = np.mat(dataArr);labelMat = np.mat(labelArr).transpose()
    svInd = np.nonzero(alphas.A > 0)[0]
    sVs = datMat[svInd]
    labelSV = labelMat[svInd]
    print("there are {} Support Vecotors".format(np.shape(sVs)[0]))
    m,n = np.shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i,:],kTup)
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]):errorCount +=1
    print("the training error rate is :{}".format(float(errorCount)/m))
    dataArr,labelArr = loadImages('testDigits')
    errorCount = 0
    datMat = np.mat(dataArr);labelMat = np.mat(labelArr).transpose()
    m,n = np.shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        predict = kernelEval.T * np.multiply(labelSV,alphas[svInd])+ b
        if np.sign(predict) != np.sign(labelArr[i]):errorCount += 1
    print("the test error rate is :{}".format(float(errorCount)/m))
testDigits(('rbf',20))