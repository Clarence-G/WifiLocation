import numpy as np
import sf_wknn

dataset = np.arange(30).reshape((30,1))+np.zeros((30,20))
labels = np.arange(30).reshape((30,1))+np.zeros((30,2))

data_now = np.array([9]*20) + np.random.randn(20)

print(data_now)
print(sf_wknn.sf_wknn(data_now, dataset, labels, 7))