import math
import random
import numpy as np
import matplotlib.pyplot as plt

plt.ion()

def plot_durations(coordinate):
    point_numbers = range(len(coordinate[0]))
    
    plt.figure(1)
    plt.clf()
    plt.xlim(0,5)
    plt.ylim(0,4)
    plt.title('FR\'s face')
    plt.scatter(coordinate[0], coordinate[1],\
        c=point_numbers, cmap=plt.cm.Blues, marker='x')
    
    plt.pause(0.5) # pause a bit so that plots are updated

coordinate = [[],[]]
x, y = 0, 0
for i in range(100):
    x = x + 4*np.random.rand()/50
    y = y + 5*np.random.rand()/50
    coordinate[0].append(x)
    coordinate[1].append(y)
    plot_durations(coordinate)