# 本地测试文件
import get_rssi
import classify
import draw
import numpy as np

dataset = np.load('./data/dataset.npy') # 测试数据
labels = np.load('./data/labels.npy') # 地点坐标


coordinate = [[],[]] # 画图数据

for i in range(100):

    data_now = get_rssi.get_data_online()
    x, y = classify.classifywknn(data_now,dataset, labels, k=3)
    coordinate[0].append(x)
    coordinate[1].append(y)
    draw.plot_durations(coordinate)
