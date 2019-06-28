import classify
import numpy as np
def sf_wknn(Inx, dataset, labels, k=3):
    sortindex = np.array(Inx).argsort().tolist().reverse()  # 返回列表
    newInx = [Inx[i] for i in sortindex[:k]]
    newdataset = [dataset[i] for i in sortindex[:k]]
    newlabels = [labels[i] for i in sortindex[:k]]
    return classify.classifywknn(newInx, newdataset, newlabels, k)