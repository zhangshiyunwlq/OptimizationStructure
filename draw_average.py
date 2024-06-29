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
    for i in range(7):
        x_te = []
        for j in range(10):
            x_te.append(20*i-1)
        x_te = np.array(x_te)
        y_te = np.linspace(0, 5, 10)
        ax2.plot(x_te, y_te, linewidth=1, color='black')
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


def fit_data(num, path):
    fitness_all = get_data(num, path)
    fitness_max = []
    fitness_min = []
    fitness_ave = []
    for i in range(len(fitness_all)):
        fitness_min.append(math.log(min(fitness_all[i]), 100))
        fitness_max.append(math.log(max(fitness_all[i]), 100))
        fitness_ave.append(math.log(np.mean(fitness_all[i]), 100))

    # 取部分
    fitness_max_local = []
    fitness_min_local = []
    fitness_ave_local = []
    for i in range(len(fitness_all)):
        local_fit = []
        for j in range(len(fitness_all[i]) - 5, len(fitness_all[i])):
            local_fit.append(fitness_all[i][j])
        fitness_min_local.append(math.log(min(local_fit), 100))
        fitness_max_local.append(math.log(max(local_fit), 100))
        fitness_ave_local.append(math.log(np.mean(local_fit), 100))
    return fitness_max, fitness_ave, fitness_min, fitness_max_local, fitness_ave_local, fitness_min_local

def DNN_fit(path):

    fit_pred = pd.read_excel(io=path, sheet_name="fit_pred_all", header=None)
    fit_pred = fit_pred.values.tolist()

    fit_truth = pd.read_excel(io=path, sheet_name="fit_truth", header=None)
    fit_truth = fit_truth.values.tolist()
    fitness_pred_max = []
    fitness_pred_min = []
    fitness_pred_ave = []
    for i in range(len(fit_pred)):
        fitness_pred_min.append(math.log(min(fit_pred[i]), 10))
        fitness_pred_max.append(math.log(max(fit_pred[i]), 10))
        fitness_pred_ave.append(math.log(np.mean(fit_pred[i]), 10))
    fitness_truth_max = []
    fitness_truth_min = []
    fitness_truth_ave = []
    for i in range(len(fit_truth)):
        fitness_truth_min.append(math.log(min(fit_truth[i]), 10))
        fitness_truth_max.append(math.log(max(fit_truth[i]), 10))
        fitness_truth_ave.append(math.log(np.mean(fit_truth[i]), 10))

    return fitness_pred_max,fitness_pred_min,fitness_pred_ave,fitness_truth_max,fitness_truth_min,fitness_truth_ave

def DNN_fit_draw(data):
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
        # if i == 3:
        #     ax2.plot(bbb, data[i], linewidth=6, color='r',linestyle='dashed')
        # if i == 4:
        #     ax2.plot(bbb, data[i], linewidth=6, color='b',linestyle='dashed')
        # if i == 5:
        #     ax2.plot(bbb, data[i], linewidth=6, color='g',linestyle='dashed')
        if i == 0:
            ax2.plot(bbb, data[i], linewidth=6, color='r')
        if i == 1:
            ax2.plot(bbb, data[i], linewidth=6, color='b')
        if i == 2:
            ax2.plot(bbb, data[i], linewidth=6, color='g')
    ax2.set(xlim=(0, len(data[0])),
            xticks=np.arange(0, len(data[0]), 1),
            )
    for i in range(7):
        x_te = []
        for j in range(10):
            x_te.append(10*i)
        x_te = np.array(x_te)
        y_te = np.linspace(0, 10, 10)
        ax2.plot(x_te, y_te, linewidth=1, color='black')
    plt.show()

num = 2400
path = "D:\desktop\os\optimization of structure\optimization of structure\optimization of structure\out_all_infor_case4\\run_infor_5_3_19.xlsx"
path_memo = "D:\desktop\os\optimization of structure\optimization of structure\optimization of structure\out_all_memorize\memorize_infor_14_73.xls"
path_DNN = f"D:\desktop\os\optimization of structure\optimization of structure\optimization of structure\DNN_test_data\\all_data_5.xlsx"
# drwa_loss(path_memo)

#绘制优化后的数据
# fitness_max,fitness_ave,fitness_min,fitness_max_local,fitness_ave_local,fitness_min_local = fit_data(num,path)
# draw = [fitness_max,fitness_ave,fitness_min]
# draw2 = [fitness_max_local,fitness_ave_local,fitness_min_local]
# draw_picture(draw)

#绘制DNN预测后的数据
# fitness_pred_max,fitness_pred_min,fitness_pred_ave,fitness_truth_max,fitness_truth_min,fitness_truth_ave = DNN_fit(path_DNN)
# fit_pred = pd.read_excel(io=path_DNN, sheet_name="DNN_prediction_fitness", header=None)
fit_pred = pd.read_excel(io=path_DNN, sheet_name="fit_truth", header=None)
fit_pred = fit_pred.values.tolist()
fit_pred_truth = pd.read_excel(io=path_DNN, sheet_name="fit_pred_all", header=None)
fit_pred_truth = fit_pred_truth.values.tolist()

fitness_pred_max = []
fitness_pred_min = []
fitness_pred_ave = []
fitness_pred_truth_min = []
index = []
for i in range(len(fit_pred)):
    # fitness_pred_min.append(math.log(min(fit_pred[i]), 2))
    # fitness_pred_max.append(math.log(max(fit_pred[i]), 2))
    # fitness_pred_ave.append(math.log(np.mean(fit_pred[i]), 2))
    fitness_pred_min.append(min(fit_pred[i]))
    index.append(fit_pred[i].index(min(fit_pred[i])))
    fitness_pred_max.append(max(fit_pred[i]))
    fitness_pred_ave.append(np.mean(fit_pred[i]))
for i in range(len(fit_pred)):
    fitness_pred_truth_min.append(fit_pred_truth[i][index[i]])
data = [fitness_pred_truth_min,fitness_pred_min]
DNN_fit_draw(data)

