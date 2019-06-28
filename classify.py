import numpy as np
import operator
import os
import get_rssi


#最近邻
def classifynn(Inx,dataset,labels,k):
    '''
    :param Inx: 测试数据
    :param dataset: 训练集
    :param labels: 训练集标签即地址坐标
    :return: 距离最近的地址坐标
    '''
    # 计算距离
    datasize=dataset.shape[0]
    copyInx=np.tile(Inx,(datasize,1))-dataset
    Inxsquare=copyInx**2
    Inxsquaresum=Inxsquare.sum(axis=1)
    d=Inxsquaresum**0.5
    #获取最小距离并统计
    minindex=d.argsort()
    return (labels[minindex[0]])




def classifyknn(Inx,dataset,labels,k):
    labels = labels.copy()
    '''
    :param Inx: 测试数据
    :param dataset: 训练集
    :param labels: 训练集标签即地址坐标
    :param k: 取距离最近的前k个
    :return: 距离最近的前k个取平均数
    '''
    #计算距离
    datasize=dataset.shape[0]
    copyInx=np.tile(Inx,(datasize,1))-dataset
    Inxsquare=copyInx**2
    Inxsquaresum=Inxsquare.sum(axis=1)
    d=Inxsquaresum**0.5

    #获取最小距离并统计
    minindex=d.argsort()
    #k临近
    xtotal = 0
    ytotal = 0
    for i in range(k):
        xtotal += labels[minindex[i]][0]
        ytotal += labels[minindex[i]][1]
    esx = xtotal/k
    esy = ytotal/k
    return esx,esy


def classifywknn(Inx,dataset,labels,k=3):
    '''

    :param Inx: 测试数据
    :param dataset: 训练集
    :param labels: 训练集标签即地址坐标
    :param k: 取距离最小的前k个
    :return: 对1/d进行加权之后的地点标签
    '''
    labels = labels.copy()
    #计算距离
    datasize=dataset.shape[0]
    copyInx=np.tile(Inx,(datasize,1))-dataset
    Inxsquare=copyInx**2
    Inxsquaresum=Inxsquare.sum(axis=1)
    d=Inxsquaresum**0.5
    # print('dshape{}'.format(d.shape))
    minindex=d.argsort()  # 返回距离从小到大的标签
    #k临近
    esx = 0
    esy = 0
    dtotal = 0
    for i in range(k):
        if d[minindex[i]] !=0:
            dtotal += 1 / d[minindex[i]]
    for i in range(k):
        if d[minindex[i]] != 0:
            esx += labels[minindex[i]][0] * (1 / d[minindex[i]]) / dtotal
            esy += labels[minindex[i]][1] * (1 / d[minindex[i]]) / dtotal
    return esx,esy




