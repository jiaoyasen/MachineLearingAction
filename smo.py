import numpy as np


def loadDataSet(fileName):
    dataMat = [];labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat


def selectJrand(i,m):  #i是第一个alpha的下标，m是所有alpha的数目，选择一个不等于i的值
    j=i
    while(j==i):
        j = int(np.random.uniform(0,m))#随机生成[0,m)之间的整数
    return j


def clipAlpha(aj, H, L):  #用于调整alpha的值，使其满足限制条件，在H和L之间
    if aj>H:
        aj = H
    if L>aj:
        aj = L
    return aj


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = np.mat(dataMatIn); labelMat = np.mat(classLabels).transpose()
    b = 0; m,n = np.shape(dataMatrix)
    alphas = np.mat(np.zeros([m,1]))
    iter = 0
    while(iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(np.multiply(alphas, labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b
            Ei = fXi - float(labelMat[i])
            if((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and \
                                                                          (alphas[i] > 0 )):
                j = selectJrand(i,m)
                fXj = float(np.multiply(alphas, labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if(labelMat[i] != labelMat[j]):
                    L = np.max(0,alphas[j] - alphas[i])
                    H = np.min(C, C + alphas[j] - alphas[i])
                else:
                    L = np.max(0, alphas[j] + alphas[i] - C)
                    H = np.min(C, alphas[j] + alphas[i])
                if L==H:print("L==H");continue
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T -dataMatrix[j,:]*dataMatrix[j,:].T
                if eta >= 0:print("eta>=0");continue
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if(np.abs(alphas[j] - alphaJold) < 0.001):print("j not moving enough");continue
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])
                b1 = b -Ei - labelMat[i]*(alphas[i] - alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - \
                    labelMat[j]*(alphas[j] - alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej - labelMat[i]*(alphas[i] - alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - \
                    labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j, :] * dataMatrix[j, :].T
                if(0 < alphas[i]) and (C > alphas[i]):b = b1
                elif (0 < alphas[j]) and ( C > alphas[j]):b = b2
                else: b = (b1 + b2)/2.0
                alphaPairsChanged += 1
                print("iter: {} i:{} pairs changed {}".format(iter, i, alphaPairsChanged))
        if(alphaPairsChanged == 0):iter += 1
        else : iter = 0
        print("iteration number:{}".format(iter))
    return b,alphas

