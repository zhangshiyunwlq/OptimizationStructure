import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
def get_data(num,path):
    # 加载MNIST手写数字数据集
    # 从Excel文件读取数据
    x_train_df = pd.read_excel(io=path,sheet_name="pop2_all")  # 假设Excel文件中没有表头
    #删除表头
    jiangjiaqi= x_train_df.values.tolist()
    head = []
    for i in range(len(jiangjiaqi)):
        if type(jiangjiaqi[i][0])== str :
            head.append(i)
    for i in range(len(head)):
            jiangjiaqi.pop(int(head[i])-i)

    #
    y_train_df = pd.read_excel(io=path,sheet_name="pop_all_fitness")
    row_data = y_train_df.values.tolist()
    head = []
    for i in range(len(row_data)):
        if type(row_data[i][0])== str :
            head.append(i)
    for i in range(len(head)):
            row_data.pop(int(head[i])-i)
    y_train = []
    fitness = []
    for i in range(len(row_data)):
        y_train.extend(row_data[i])
        fitness.append(row_data[i])
    # 将数据转换为NumPy数组
    x_train = np.array(jiangjiaqi)
    y_train = np.array(y_train)

    x_train1 = x_train[0:num]
    y_train1 = y_train[0:num]
    return fitness

def draw_picture(data):
    fig2 = plt.figure(num=1, figsize=(23, 30))
    ax2 = fig2.add_subplot(111)
    ax2.tick_params(labelsize=40)
    ax2.set_xlabel("Iteration",fontsize=50)  # 添加x轴坐标标签，后面看来没必要会删除它，这里只是为了演示一下。
    ax2.set_ylabel('fitness', fontsize=50)  # 添加y轴标签，设置字体大小为16，这里也可以设字体样式与颜色
    ax2.spines['bottom'].set_linewidth(4);###设置底部坐标轴的粗细
    ax2.spines['left'].set_linewidth(4)
    ax2.spines['right'].set_color('none')
    ax2.spines['top'].set_color('none')
    # plt.ylim((150, 400))
    bbb = np.arange(0, len(data[0]))
    for i in range(len(data)):
        if i == 0:
            ax2.plot(bbb, data[i], linewidth=6, color='r')
        if i == 1:
            ax2.plot(bbb, data[i], linewidth=6, color='b')
        if i == 2:
            ax2.plot(bbb, data[i], linewidth=6, color='g')
    ax2.set(xlim=(0,  len(data[0])),
           xticks=np.arange(0,  len(data[0]), 20),
           )
    plt.show()

def drwa_loss(path_m):
    x_train_df = pd.read_excel(io=path_m, sheet_name="memorize_loss")
    all_data = x_train_df.values.tolist()
    data_x = []
    for i in range(len(all_data)):
        data_x.extend(all_data[i])
    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot()
    ax2.set_xlabel("time",fontsize=10)
    ax2.set_ylabel("loss", fontsize=10)
    dev_x = np.arange(0, len(data_x))
    dev_y = data_x
    ax2.plot(dev_x, dev_y)
    plt.show()


num = 2400
path = "D:\desktop\os\optimization of structure\optimization of structure\optimization of structure\out_all_infor_case4\\run_infor_9_6_0.xlsx"
path_memo = "D:\desktop\os\optimization of structure\optimization of structure\optimization of structure\out_all_memorize\memorize_infor_14_73.xls"
# drwa_loss(path_memo)
fitness_all = get_data(num,path)
fitness_max = []
fitness_min = []
fitness_ave = []
for i in range(len(fitness_all)):
    fitness_min.append(math.log(min(fitness_all[i]),100))
    fitness_max.append(math.log(max(fitness_all[i]),100))
    fitness_ave.append(math.log(np.mean(fitness_all[i]),100))

#取部分
fitness_max_local = []
fitness_min_local = []
fitness_ave_local = []
for i in range(len(fitness_all)):
    local_fit = []
    for j in range(len(fitness_all[i])-5,len(fitness_all[i])):
        local_fit.append(fitness_all[i][j])
    fitness_min_local.append(math.log(min(local_fit),100))
    fitness_max_local.append(math.log(max(local_fit),100))
    fitness_ave_local.append(math.log(np.mean(local_fit),100))

draw = [fitness_max,fitness_ave,fitness_min]
draw2 = [fitness_max_local,fitness_ave_local,fitness_min_local]

draw_picture(draw)
