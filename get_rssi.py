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
def get_rssi_time(totaltime,second,k):
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

#返回数据预处理后的特征:测试总时间，每次测试时间间隔，取前k个信号最强的wifi
def data_proced(totaltime,second,k,i,j):
    result = {}
    for key,value in get_rssi_time(totaltime,second,k).items():
        result[key] = gaussian(value)
    file = open('result.txt','a+')
    file.write('({},{}):\n'.format(i,j))
    file.write(str(result))
    file.write('\n')

if __name__ == '__main__':
    #data_proced(5,0.5,20,0,1)
    i = 0
    for key, value in get_rssi_time(30, 1, -1).items():
        print(i,"--", key," : ",value )
