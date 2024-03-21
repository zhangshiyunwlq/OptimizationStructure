import matplotlib.pyplot as plt
import numpy as np

y_axis_data = [1, 2, 3, 4, 5, 6]  # x
x_axis_data = [4.855753820255013,12.574656459910782,19.317326429638907,24.52220534438628,27.907332148767278,29.484081849850156]  # y
lim_data = [5,10,15,20,25,30]
lim_inter = [0.004,0.004,0.004,0.004,0.004,0.004]
plt.rcParams['font.sans-serif'] = ['SimHei']

Y_dis_pingyi=[2.5,
                 6.64,
                 10.18,
                 12.92,
                 14.72,
                 15.50]
Y_inter_pingyi=[0.0008,
                 0.0014,
                 0.0012,
                 0.0009,
                 0.0006,
                 0.00026]
Y_dis_gangjie=[4.800530189965177,
                 12.549076724772119,
                 19.280723708609788,
                 24.468428427613464,
                 27.839252522509334,
                 29.265145759894178]
Y_inter_gangjie=[0.0016001767299883925,
                 0.002654958721951585,
                 0.0022438823279458896,
                 0.0017292349063345587,
                 0.0011236080316319565,
                 0.0005608026855161938]
Y_dis_bolt=[2.5,
                 6.63,
                 10.22,
                 13.01,
                 14.85,
                 15.71]
Y_inter_bolt=[0.0008,
             0.0014,
             0.0011,
             0.00094,
             0.00061,
             0.00029]

plt.plot(Y_dis_pingyi, y_axis_data, '-', alpha=0.5, linewidth=3, label='平移弹簧模型')  # 'bo-'表示蓝色实线，数据点实心原点标注
## plot中参数的含义分别是横轴值，纵轴值，线的形状（'s'方块,'o'实心圆点，'*'五角星   ...，颜色，透明度,线的宽度和标签 ，
plt.plot(Y_dis_bolt, y_axis_data, '-', alpha=0.5, linewidth=3, label='螺栓连接')
plt.plot(lim_data, y_axis_data, '-', alpha=0.5, linewidth=3, label='限值')
plt.legend()  # 显示上面的label
plt.xlabel('最大位移',fontsize=10)  # x_label
plt.ylabel('楼层',fontsize=10)  # y_label

# plt.ylim(-1,1)#仅设置y轴坐标范围
plt.show()
