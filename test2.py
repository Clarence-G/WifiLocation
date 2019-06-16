from random import choice,randint
import matplotlib.pyplot as plt

class PersonMovement():
    def __init__(self,walk_nums = 10):
        self.walk_nums = walk_nums   #移动的次数
        self.x_values = [0]  #x方向轨迹坐标
        self.y_values = [0]  #y方向的轨迹坐标


    def move(self):
        while len(self.x_values) < self.walk_nums:
            x_direction = choice([-1,1])  # x轴运动方向
            x_distance = randint(0,20) #x轴运动距离
            x_step = x_direction *x_distance

            y_direction = choice([-1,1]) #y轴运动方向
            y_distance  = randint(0,20) #y轴运动距离
            y_step = y_direction *y_distance

            if x_step !=0 or y_step !=0:
                next_x = self.x_values[-1] + x_step
                next_y = self.y_values[-1] + y_step

                self.x_values.append(next_x)
                self.y_values.append(next_y)


for i in range(5):
    pm = PersonMovement(10000)
    pm.move()

    point_numbers = range(pm.walk_nums)
    #绘制运动的轨迹图，且颜色由浅入深
    plt.scatter(pm.x_values, pm.y_values, c=point_numbers, cmap=plt.cm.Blues, edgecolors='none', s=15)
    #将起点和终点高亮显示，s=100代表绘制的点的大小
    plt.scatter(pm.x_values[0], pm.y_values[0], c='green', s=100)
    plt.scatter(pm.x_values[-1], pm.y_values[-1], c='red', s=100)
    # 隐藏x、y轴
    plt.axes().get_xaxis().set_visible(True)
    plt.axes().get_yaxis().set_visible(True)
    #显示运动轨迹图
    plt.show()