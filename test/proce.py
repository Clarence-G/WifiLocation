import numpy as np

a = np.arange(19).reshape(19,1) + np.zeros(shape=(19,2))
np.save('data/labels.npy', a)