# 本地测试文件
import get_rssi
import classify
import numpy as np
import draw

labellist = []
dataset = np.load('./data/dataset.npy')
for i in range(len(dataset)):
    y, x  = divmod(i,6)  # 行x， 列 y
    labellist.append((x,y))

#result = get_rssi.data_procedonline(10, 1)


coordinate = [[],[]]

for i in range(100):

    data_now = classify.dataproconline()
    x, y = classify.classifywknn(data_now,dataset,labellist,k=3)
    coordinate[0].append(x)
    coordinate[1].append(y)
    draw.plot_durations(coordinate)
