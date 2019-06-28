# 本地测试文件
import get_rssi
import classify
import numpy as np
import draw

dataset = np.load('./data/dataset.npy')
labels = np.load('./labels.npy')

#result = get_rssi.data_procedonline(10, 1)


coordinate = [[],[]]

for i in range(100):

    data_now = get_rssi.get_data_online()
    x, y = classify.classifywknn(data_now,dataset, labels, k=3)
    coordinate[0].append(x)
    coordinate[1].append(y)
    draw.plot_durations(coordinate)
