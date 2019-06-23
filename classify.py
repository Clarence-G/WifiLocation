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


# 在线获取wifi特征向量
def dataproconline(totaltime=2,second = 0.5,k = -1):
    '''
    :param totaltime:测试总时间，（由于人的运动，故每个点的测试总时间无法过长）
    :param second:测试频率
    :param k:wifi强度前k强
    :return:返回特征向量
    '''
    # 已有wifi列表
    wifilist = ['04:25:c5:b4:b2:80', '04:25:c5:b4:b2:81', '30:74:96:1a:eb:b6', '80:b5:75:77:4c:a0', '80:b5:75:77:31:c0', '80:b5:75:77:31:cf', '00:18:39:45:48:e9', '80:b5:75:77:5c:6f', '80:b5:75:77:5c:60', '80:b5:75:77:5c:61', '06:69:6c:86:d8:80', '80:b5:75:77:2d:c0', '80:b5:75:77:81:a0', '04:25:c5:b4:bc:a0', '04:25:c5:b4:cf:80', '04:25:c5:b4:cf:81', '60:0b:03:01:4c:d0', '60:0b:03:01:4c:d2', '60:0b:03:01:4c:d1', '80:b5:75:77:2e:e0', '80:b5:75:77:2e:ef', '04:25:c5:b4:b2:c0', '80:b5:75:77:81:40',
'80:b5:75:77:4c:a1', '06:69:6c:86:d8:11', '04:25:c5:b4:bc:a1', '80:b5:75:77:2d:c1', '00:12:5f:11:7d:10', '06:69:6c:86:d8:12', '80:b5:75:77:2e:e1', '80:b5:75:77:81:41', '80:b5:75:77:31:c1', '06:69:6c:86:d8:7f', '04:25:c5:b4:bf:a1', '00:12:5f:11:9c:2a', '04:25:c5:b4:bf:a0', '04:25:c5:b4:b2:c1', '80:b5:75:77:81:a1', '00:06:f4:e3:76:a0', '04:25:c5:b4:b3:20', '30:49:3b:08:2d:7e', 'd8:c8:e9:06:4c:68', '00:12:5f:11:8f:7c', '8c:ab:8e:ee:e0:40', '04:25:c5:b4:b3:21', '00:22:6b:59:72:1f', '80:b5:75:77:2f:a0', '80:b5:75:77:2f:a1', '04:25:c5:b4:cd:20', '00:16:b6:25:28:03', 'f4:83:cd:9f:e7:bc', '00:06:f4:e3:76:a1', '06:69:6c:86:db:1e', '00:12:5f:10:e7:cc', '04:25:c5:b4:cd:60', '04:25:c5:b4:cd:61', '00:12:5f:0b:b4:d8', '00:06:f4:e3:77:01', '00:06:f4:e3:77:00', '00:12:5f:10:e8:80', '52:8f:4c:6d:e2:9a', '04:25:c5:b4:b2:91', '04:25:c5:b4:b2:90', '04:25:c5:b4:cf:90', '04:25:c5:b4:cf:91', '80:b5:75:77:4c:b0', '80:b5:75:77:4c:b1', '80:b5:75:77:2d:d0', '80:b5:75:77:2d:d1', '80:b5:75:77:31:d0', '80:b5:75:77:31:d1', '80:b5:75:77:31:df', '04:25:c5:b4:bc:b0', '60:0b:03:01:4c:c0', '04:25:c5:b4:b2:d0', '60:0b:03:01:4c:c1', '60:0b:03:01:4c:c2', '80:b5:75:77:5c:71', '80:b5:75:77:5c:70', '80:b5:75:77:5c:7f', '04:25:c5:b4:bc:b1', '04:25:c5:b4:b2:d1', 'ec:88:8f:63:55:a0']
    # 线上获取 初始化
    onlinelist = [-999]*len(wifilist)
    onlineinfo = get_rssi.data_procedonline(totaltime,second,k)
    onlinewifilist = list(onlineinfo.keys())  # 实时获取wifi列表
    for onlinewifi in onlinewifilist:
        try:  # 将实时获取的WiFi强度对应添加到实时数据表钟
            index = wifilist.index(onlinewifi)
            onlinelist[index] = onlineinfo[onlinewifi]
        except:
            pass
    return onlinelist

def validate(method,truth,data,labellist,k,m):
    '''
    :param method: 判别方法，nn，knn，wknn
    :param truth: 测试数据编号
    :param data: 训练数据集
    :param labellist: 训练数据地址标签
    :param k:
    :param m:  扰动范围
    :return:
    '''
    disturbdata = data[truth]+m*np.random.random(len(data[truth]))  # 由于没有收集测试数据，在此加上扰动充当测试数据。
    predict = method(disturbdata,data,labellist,k)
    m,n = predict
    i,j = labellist[truth]
    print('use method {}, the predict result is {}'.format(method.__name__,predict))
    return np.linalg.norm((abs(m-i),abs(n-j)))




if __name__=='__main__':
    # 测试
    # 建立地点标签
    data = np.load('data/dataset.npy')
    labellist = []
    for i in range(len(data)):
        y, x  = divmod(i,6)  # 行x， 列 y
        labellist.append((x,y))
    # print(labellist)
    for truth in range(len(data)):
        print('the truth is {}'.format(labellist[truth]))
        validate(classifynn,truth,data,labellist,3,50)
        validate(classifyknn, truth, data, labellist,3,50)
        validate(classifywknn, truth, data,labellist, 3,50)