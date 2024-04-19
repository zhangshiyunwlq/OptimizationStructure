import matplotlib.pyplot as plt

node1 = [(0,0),(8,0),(0,3),(8,3),(11,0),(19,0),(11,3),(19,3)]
nodes_all = []
for i in range(6):
    for j in range(8):
        nodes_all.append((node1[j][0],node1[j][1]+3.4*i))
beam1 = [(0,1),(2,3),(4,5),(6,7)]
beams_all = []
for i in range(6):
    for j in range(4):
        beams_all.append((beam1[j][0]+8*i,beam1[j][1]+8*i))
colu1 = [(0,2),(1,3),(4,6),(5,7)]
colus_all = []
for i in range(6):
    for j in range(4):
        colus_all.append((colu1[j][0]+8*i,colu1[j][1]+8*i))

con1 = [(2,8),(3,9),(6,12),(7,13)]
cons_all = []
for i in range(5):
    for j in range(4):
        cons_all.append((con1[j][0]+8*i,con1[j][1]+8*i))

corr1 = [(1,4),(3,6)]
corrs_all = []
for i in range(6):
    for j in range(2):
        corrs_all.append((corr1[j][0]+8*i,corr1[j][1]+8*i))

# 定义框架节点坐标
nodes = [(0, 0), (1, 0), (0.5, 1), (1.5, 1), (1, 2)]

# 定义框架线段（连接节点的线段）
lines = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)]

# 绘制框架
for line in beams_all:
    x_values = [nodes_all[line[0]][0], nodes_all[line[1]][0]]
    y_values = [nodes_all[line[0]][1], nodes_all[line[1]][1]]
    plt.plot(x_values, y_values, 'b-')


for line in cons_all:
    x_values = [nodes_all[line[0]][0], nodes_all[line[1]][0]]
    y_values = [nodes_all[line[0]][1], nodes_all[line[1]][1]]
    plt.plot(x_values, y_values, 'b-')

for line in colus_all:
    x_values = [nodes_all[line[0]][0], nodes_all[line[1]][0]]
    y_values = [nodes_all[line[0]][1], nodes_all[line[1]][1]]
    plt.plot(x_values, y_values, 'b-')

for line in corrs_all:
    x_values = [nodes_all[line[0]][0], nodes_all[line[1]][0]]
    y_values = [nodes_all[line[0]][1], nodes_all[line[1]][1]]
    plt.plot(x_values, y_values, 'b-')

# 绘制节点
for node in nodes_all:
    plt.plot(node[0], node[1], 'ro')

# 设置坐标轴范围
plt.xlim(-3, 30)
plt.ylim(-5,30)

# 设置坐标轴标签
plt.xlabel('X')
plt.ylabel('Y')

# 显示图形
plt.gca().set_aspect('equal', adjustable='box')
plt.show()