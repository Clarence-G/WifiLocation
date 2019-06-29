import classify
import numpy as np
import time
import get_rssi

def sf_wknn(Inx, dataset, labels,n, k=3):
    sortindex = np.array(Inx).argsort().tolist()
    sortindex.reverse()  # 返回列表
    newInx = np.array(Inx)[sortindex[:n]]
    newdataset = dataset[:,sortindex[:n]]
    return classify.classifywknn(newInx, newdataset, labels, k)


def slide_get_data(t):
    beta = 0.125
    old = np.array(get_rssi.get_data_online(1,0.5))
    while 1:
        time.sleep(t)
        new = np.array(get_rssi.get_data_online(1, 0.5))
        old = beta*old+(1-beta)*new
        print(old)





if __name__ == '__main__':
    dataset = np.load('./data/dataset.npy')
    labels = np.load('./labels.npy')
    data = [-59.35, -59.0, -999, -74.82, -75.21, -75.29, -999, -76.44, -76.67, -76.44, -999, -78.8, -999, -78.0, -999, -999, -999, -81.0, -999, -999, -999, -999, -83.0, -75.21, -999, -999, -78.57, -999, -999, -999, -999, -76.5, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -58.56, -59.22, -63.5, -64.35, -66.88, -67.8, -78.0, -78.94, -74.7, -72.57, -74.5, -999, -999, -79.0, -999, -999, -999, -999, -999, -999, -999, -999]
    print(sf_wknn(data,dataset,labels,10,5))
    slide_get_data(2)