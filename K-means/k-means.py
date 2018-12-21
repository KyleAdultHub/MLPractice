# coding:utf-8
import numpy as np


def distEclud(vecA, vecB):  # 计算欧式距离
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))  # la.norm(vecA-vecB)


def randCent(dataSet, k):  # 初始化k个随机簇心
    n = np.array(dataSet).shape[1]  # 特征个数
    centroids = np.mat(np.zeros((k, n)))  # 簇心矩阵k*n
    for j in range(n):  # 特征逐个逐个地分配给这k个簇心。每个特征的取值需要设置在数据集的范围内
        minJ = min(dataSet[:, j])  # 数据集中该特征的最小值
        rangeJ = float(max(dataSet[:, j]) - minJ)  # 数据集中该特征的跨度
        centroids[:, j] = np.mat(minJ + rangeJ * np.random.rand(k, 1))  # 为k个簇心分配第j个特征，范围需限定在数据集内。
    return centroids  # 返回k个簇心


def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    """随机初始化k-means"""
    m = np.array(dataSet).shape[0]  # 数据个数
    clusterAssment = np.mat(np.zeros((m, 2)))  # 记录每个数据点被分配到的簇，以及到簇心的距离
    centroids = createCent(dataSet, k)  # 初始化k个随机簇心
    clusterChanged = True  # 记录一轮中是否有数据点的归属出现变化，如果没有则算法结束
    while clusterChanged:
        clusterChanged = False
        for i in range(m):  # 枚举每个数据点，重新分配其簇归属
            minDist = np.inf  # 先假设距离簇心的距离为无穷大，则先随机分配一个簇心给数据点
            minIndex = -1  # 记录最近簇心及其距离
            for j in range(k):  # 枚举每个簇心
                distJI = distMeas(centroids[j, :], dataSet[i, :])  # 计算数据点与簇心的距离
                if distJI < minDist:  # 更新最近簇心
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True  # 更新“变化”记录
                clusterAssment[i, :] = minIndex, minDist ** 2  # 更新数据点的簇归属
        print(centroids)
        for cent in range(k):  # 枚举每个簇心，更新其位置
            ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]  # 得到该簇所有的数据点
            centroids[cent, :] = np.mean(ptsInClust, axis=0)  # 将数据点的均值作为簇心的位置
    return centroids, clusterAssment  # 返回簇心及每个数据点的簇归属


def biKmeans(dataSet, k, distMeas=distEclud):
    """随机初始化二值化k-means"""
    m = np.array(dataSet).shape[0]
    centroid0 = np.mean(dataSet, axis=0).tolist()[0]  # 创建初始簇心，标号为0
    centList = [centroid0]  # 创建簇心列表
    clusterAssment = np.array(np.zeros((m, 2)))  # 初始化所有数据点的簇归属(为0)
    for j in range(m):  # 计算所有数据点与簇心0的距离
        clusterAssment[j, 1] = distMeas(np.array(centroid0), dataSet[j, :]) ** 2
    while len(centList) < k:  # 分裂k-1次，形成k个簇
        lowestSSE = np.inf  # 初始化最小sse为无限大
        for i in range(len(centList)):  # 枚举已有的簇，尝试将其一分为二
            ptsInCurrCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == i)[0], :]  # 将该簇的数据点提取出来
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)  # 利用普通k均值将其一分为二
            sseSplit = np.sum(splitClustAss[:, 1])  # 计算划分后该簇的SSE
            sseNotSplit = np.sum(clusterAssment[np.nonzero(clusterAssment[:, 0].A != i)[0], 1])  # 计算该簇之外的数据点的SSE
            print("sseSplit, and notSplit: ", sseSplit, sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE:  # 更新最小总SSE下的划分簇及相关信息
                bestCentToSplit = i  # 被划分的簇
                bestNewCents = centroidMat  # 划分后的两个簇心
                bestClustAss = splitClustAss.copy()  # 划分后簇内数据点的归属及到新簇心的距离
                lowestSSE = sseSplit + sseNotSplit  # 更新最小总SSE
        print('the bestCentToSplit is: ', bestCentToSplit)
        print('the len of bestClustAss is: ', len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]  # 一个新簇心的标号为旧簇心的标号，所以将其取代就簇心的位置
        centList.append(bestNewCents[1, :].tolist()[0])  # 另一个新簇心加入到簇心列表的尾部，标号重新起
        bestClustAss[np.nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)  # 更新旧簇内数据点的标号
        bestClustAss[np.nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit  # 同上
        clusterAssment[np.nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss  # 将更新的簇归属统计到总数据上
    return np.mat(centList), clusterAssment
