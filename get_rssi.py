import pywifi
import operator
import os
import time
import numpy as np

def getrssilist(k):
    # 选择网卡
    wifi = pywifi.PyWiFi()
    iface = wifi.interfaces()[0]
    iface.scan()
    result = iface.scan_results() # 扫描结果
    needresult = {}

    for i in range(len(result)):
        # needresult[(result[i].ssid, result[i].bssid)] = result[i].signal # 创建扫描结果字典（ssid，bssid）：信号强度
        needresult[result[i].bssid] = result[i].signal  # 创建扫描结果字典bssid：信号强度

    # 根据rssi强度由高到低排序
    finalresult = sorted(needresult.items(), key=lambda item: item[1], reverse=True)
    return finalresult[:k]
    # 返回信号前k强

#在totaltime内，每隔second秒获取信号前k强的数据并加入到字典中 格式：bssid:[rssi列表]
def get_rssi_time(totaltime,second,k=-1):
    print("start get")
    start=time.clock()
    result = {}
    while 1:
        time.sleep(second)
        data_now = getrssilist(k)
        for data in data_now:
            result[data[0]]=result.get(data[0],[])+[data[1]]
        end = time.clock()
        if int(end - start) == totaltime:
            break
    print('end')
    return result

# 输入某wifi的rssi序列对数据进行先高斯后均值处理
def gaussian(rssilist):
    array = np.array(rssilist)
    weight = 1.65
    premean = np.mean(rssilist)
    prestd = np.std(rssilist)
    gaumin = premean-weight
    gaumax = premean+weight
    aftergaussian = []
    for i in range(len(rssilist)):
        if(rssilist[i]<gaumin or rssilist[i]>gaumax):
            continue
        aftergaussian.append(rssilist[i])
    # print(max(aftergaussian))
    aftergaussianmean = np.mean(aftergaussian)
    return round(aftergaussianmean,2)

#返回数据预处理后的特征:测试总时间，每次测试时间间隔，取前k个信号最强的wifi。由于默认取所有可测得的数据故k默认取-1
# 离线阶段的数据采集
def data_proced(filename,totaltime,second,i,j,k=-1):
    result = {}
    for key,value in get_rssi_time(totaltime,second,k).items():
        result[key] = gaussian(value)
    file = open('{}.txt'.format(filename),'a+')
    file.write('({},{})：\n'.format(i,j))
    file.write(str(result))
    file.write('\n')

def data_proced_online(totaltime,second,k=-1):
    '''
    :param totaltime: 在线获取时间
    :param second: 频率
    :param k:
    :return: 返回当前所有wifi数据的字典格式：  mac：信号强度
    '''
    result = {}
    for key,value in get_rssi_time(totaltime,second,k).items():
        key = key.strip(':')
        result[key] = gaussian(value)
    return result


# 在线获取wifi特征向量
def get_data_online(totaltime=2,second = 0.5,k = -1):
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
    onlineinfo = data_proced_online(totaltime,second,k)
    onlinewifilist = list(onlineinfo.keys())  # 实时获取wifi列表
    for onlinewifi in onlinewifilist:
          # 将实时获取的WiFi强度对应添加到实时数据表钟
        if onlinewifi in wifilist:
            index = wifilist.index(onlinewifi)
            onlinelist[index] = onlineinfo[onlinewifi]

    return onlinelist

if __name__ == '__main__':
    print(data_procedonline(10,0.5,-1))

