import numpy as np


def loadDataSet(fileName):#读取数据，老生常谈，就不多说了，按行读，以特定的符号分割数据
    dataMat = [];labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])#前两维是数据
        labelMat.append(float(lineArr[2]))#第三位的是标签
    return dataMat,labelMat


def selectJrand(i,m):  #i是第一个alpha的下标，m是所有alpha的数目，选择一个不等于i的值
    j=i
    while(j==i):
        j = int(np.random.uniform(0,m))#随机生成[0,m)之间的整数
    return j


def clipAlpha(aj, H, L):  #用于调整alpha的值，使其满足限制条件，在H和L之间
    if aj>H:#比H大的就是H
        aj = H
    if L>aj:#比L小的就是L
        aj = L
    return aj


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    '''
    
    :param dataMatIn:     输入数据
    :param classLabels:   数据标签
    :param C:   约束条件 0<alpha<C 
    :param toler: 衡量alpha的变化的参数值
    :param maxIter:  迭代的次数
    :return: 
    '''
    dataMatrix = np.mat(dataMatIn); labelMat = np.mat(classLabels).transpose()#数据存为mat格式，标签一列向量的格式
    b = 0; m,n = np.shape(dataMatrix)#b初始值设为0，m,n分别为输入数据的行数和列数
    alphas = np.mat(np.zeros([m,1]))#为每个数据分配一个初始值为0的alpha
    iter = 0
    while(iter < maxIter):#，外层循环，迭代次数
        alphaPairsChanged = 0#一个用来记录两个alpha是否修改的变量
        for i in range(m):#对于m个数据，进行遍历
            fXi = float(np.multiply(alphas, labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b#用初始值计算fxi的值
            Ei = fXi - float(labelMat[i])#计算误差
            if((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and \
                                                                          (alphas[i] > 0 )):#判断满足约束条件的第一个alpha
                j = selectJrand(i,m)#找一个与alphai不一样的alphaj
                fXj = float(np.multiply(alphas, labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b#用alphaj计算fxj
                Ej = fXj - float(labelMat[j])#计算alphaj的误差
                alphaIold = alphas[i].copy()#将二者记录下来，因为后续要对alpha进行更新比较，二者要保留，作为alphaold
                alphaJold = alphas[j].copy()
                if(labelMat[i] != labelMat[j]):#yi不等于yj的情况，裁剪限制
                    L = max(0,alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:#yi等于yj的情况，裁剪限制
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L==H:print("L==H");continue
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T -dataMatrix[j,:]*dataMatrix[j,:].T
                #计算eta，是计算alpha更新的一部分
                if eta >= 0:print("eta>=0");continue
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta#alphaj的更新方法
                alphas[j] = clipAlpha(alphas[j], H, L)#alphaj的裁剪限制
                if(np.abs(alphas[j] - alphaJold) < 0.001):print("j not moving enough");continue#假如alphaj的变化小于阈值，就放弃
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])#计算alphai的更新，与alphaj反向
                b1 = b -Ei - labelMat[i]*(alphas[i] - alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - \
                    labelMat[j]*(alphas[j] - alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T#计算b1的更新
                b2 = b - Ej - labelMat[i]*(alphas[i] - alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - \
                    labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j, :] * dataMatrix[j, :].T#计算b2的更新
                if(0 < alphas[i]) and (C > alphas[i]):b = b1
                elif (0 < alphas[j]) and ( C > alphas[j]):b = b2
                else: b = (b1 + b2)/2.0#b更新的方法
                alphaPairsChanged += 1#假如alpha对更新了，计数
                print("iter: {} i:{} pairs changed {}".format(iter, i, alphaPairsChanged))
        if(alphaPairsChanged == 0):iter += 1#假如没修改，就增加迭代计数轮次，直到到达迭代轮次的限制，否则就重置iter
        else : iter = 0
        print("iteration number:{}".format(iter))
    return b,alphas

dataArr, labelArr = loadDataSet('testSet.txt')
b, alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
print(b)

