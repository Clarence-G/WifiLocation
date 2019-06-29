import numpy as np

filename = "./data/origin_data.txt"
f = open(filename)
print('Open Sussessfuly')

nan = -999.
big = 1000
l = []
Macadd = []
num = 0
dictep = dict()
dataset = np.full(shape=(big, big), fill_value=nan)

for line in f.readlines():
    line = line.strip('\n:')
    num += 1
    if num % 2 == 1:
        l.append(line)
    else:
        dictep = eval(line)
        for key, value in dictep.items():
            key = key.strip(':')
            
            if key not in Macadd:
                Macadd.append(key)
            
            dataset[num//2-1][Macadd.index(key)] = value

dataset = dataset[:len(l), :len(Macadd)]
np.save('data/dataset.npy', dataset)
print(Macadd)
print(l)
print(dataset.shape)