import matplotlib.pyplot as plt
import numpy as np

# 定义距离函数，计算点(x, y)到曲线y=1/x的距离
def distance_to_curve(x, y):
    curve_y = 1 / x
    return abs(y - curve_y)

# 生成随机的x坐标值
num_points = 10000
x_values = np.random.uniform(0.1, 10, num_points)

# 生成随机的y坐标值
y_values = np.random.uniform(0, 5, num_points)

# 选取满足条件的点，并增加根据距离曲线的远近调整显示概率
x_selected_blue = []
y_selected_blue = []
x_selected_grey = []
y_selected_grey = []
x_selected_green = []
y_selected_green = []
x_selected_yellow = []
y_selected_yellow = []

for x, y in zip(x_values, y_values):
    if y > 1 / x:
        distance = distance_to_curve(x, y)
        prob = np.exp(-distance)  # 使用指数函数作为概率，距离越近，概率越大
        if np.random.uniform() < prob:  # 使用均匀分布来决定是否显示该点
            if distance <= 0.1:  # 在一定距离范围内的点显示为红色
                if 1 <= x <= 2:  # 在指定的x范围内的红色点显示为绿色
                    x_selected_green.append(x)
                    y_selected_green.append(y)
                elif 8 <= x <= 9:
                    x_selected_blue.append(x)
                    y_selected_blue.append(y)
                elif 0.2 <= x <= 0.35:
                    x_selected_yellow.append(x)
                    y_selected_yellow.append(y)
                elif 2 <= x <= 8:
                    x_selected_grey.append(x)
                    y_selected_grey.append(y)
                elif 9 <= x <= 10:
                    x_selected_grey.append(x)
                    y_selected_grey.append(y)
            else:  # 超出距离范围的点显示为蓝色
                x_selected_grey.append(x)
                y_selected_grey.append(y)
fig =plt.figure()
ax = fig.add_subplot(111)
# 绘制散点图
ax.scatter(x_selected_blue, y_selected_blue, color='y', alpha=0.5,s=150)
ax.scatter(x_selected_grey, y_selected_grey, color='grey', alpha=0.5,s=30)
ax.scatter(x_selected_green, y_selected_green, color='green', alpha=0.5,s=150)
ax.scatter(x_selected_yellow, y_selected_yellow, color='b', alpha=1,s=150)
# 添加标题和坐标轴标签
# plt.title('Random Scatter Plot on y=1/x with Distance Color Coding and Probability')
plt.rcParams['font.sans-serif'] = ['SimHei']

plt.xlabel('建筑建造成本指数(1e6)',
           labelpad=6,  #调整x轴标签与x轴距离
           x=0.85,  #调整x轴标签的左右位置
           fontsize=40)
plt.ylabel('建筑安全性能指数(1e6)',
            fontsize=40,
           labelpad=-160,  #调整y轴标签与y轴的距离
           y=1.02,  #调整y轴标签的上下位置
           rotation=0,)
ax.spines['bottom'].set_linewidth(1.5);  ###设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
# 修改x轴和y轴的刻度标签
plt.xticks(np.arange(0, 11, 2),fontsize=33)  # 修改x轴的刻度
plt.yticks(np.arange(0, 6, 1),fontsize=33)   # 修改y轴的刻度

# 修改x轴和y轴的刻度标签
# custom_x_ticks = ['Zero', 'Two', 'Four', 'Six', 'Eight', 'Ten']  # 自定义的 x 轴刻度标签
# plt.xticks(np.arange(0, 11, 2), custom_x_ticks)  # 修改x轴的刻度
#
# plt.yticks(np.arange(0, 6, 1))   # 修改y轴的刻度
# 添加图例
# plt.legend()

# 显示图形
# plt.grid(True)
plt.show()