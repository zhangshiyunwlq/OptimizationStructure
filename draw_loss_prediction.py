import copy
import random
import xlrd
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
import math
import numpy as np

def gx_data_convert(gx_prediction_draw):
    fitness5 = copy.deepcopy(gx_prediction_draw)
    gx_prediction = []
    for j in range(len(gx_prediction_draw)):
        if fitness5[j][0]<=0:
            fitness5[j][0] =0
        if fitness5[j][1] <= 0:
            fitness5[j][1] = 0
        if fitness5[j][2]<=0.00167 and fitness5[j][2] >= -0.00167:
            fitness5[j][2] =0
        if fitness5[j][3] <= 0.004 and fitness5[j][3] >= -0.004:
            fitness5[j][3] = 0
        gx_prediction.append(fitness5[j][5]+10000*(fitness5[j][0]+fitness5[j][1]+fitness5[j][2]*100+fitness5[j][3]*100+fitness5[j][4]))
    return gx_prediction

def draw_gx(gx_nor_data, gx_prediction_data):
    #归一化还原
    # gx_prediction_nonnor = gx_nonNormalization(gx_prediction_data)
    # gx_nor_nonnor = gx_nonNormalization(gx_nor_data)

    gx_prediction_nonnor = gx_prediction_data
    gx_nor_nonnor = gx_nor_data


    gx_prediction_draw1 = copy.deepcopy(gx_prediction_nonnor)
    gx_nor_draw1 = copy.deepcopy(gx_nor_nonnor)

    gx_prediction_draw = []
    gx_nor_draw = []
    time_num = 0
    for i in range(len(gx_prediction_nonnor)):
        # gx_prediction_draw.append(sum(gx_prediction_draw1[i]))
        # gx_nor_draw.append(sum(gx_nor_draw1[i]))
        # gx_prediction_draw.append((gx_prediction_draw1[i][0]))
        # gx_nor_draw.append((gx_nor_draw1[i][0]))
        gx_prediction_draw = gx_data_convert(gx_prediction_draw1)
        gx_nor_draw2 = gx_data_convert(gx_nor_draw1)
        gx_nor_draw = gx_nor_draw2[0:len(gx_prediction_draw)]
    #     if gx_nor_draw[i]-gx_prediction_draw[i]<=0.3 and gx_nor_draw[i]-gx_prediction_draw[i]>=-0.3:
    #         time_num +=1
    # gx_cha_num.append(time_num/len(gx_prediction_draw))
    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot()
    ax2.tick_params(labelsize=30)
    ax2.set_xlabel("reality", fontsize=35)
    ax2.set_ylabel("prediction", fontsize=35)
    ax2.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax2.spines['left'].set_linewidth(2)
    ax2.spines['right'].set_color('none')
    ax2.spines['top'].set_color('none')
    dev_x = np.arange(0, len(gx_prediction_draw))
    dev_y = gx_prediction_draw
    dev_z = gx_nor_draw

    x_min = min(gx_nor_draw)
    x_max = max(gx_nor_draw)
    x_fun = np.linspace(x_min,x_max,20).tolist()
    # ax2.plot(dev_x, dev_y, linewidth=1, color='r', linestyle=':', marker='s', label='prediction')
    # ax2.plot(dev_x, dev_z, linewidth=1, color='b', marker='+', label='reality')
    ax2.plot(x_fun, x_fun, linewidth=1, color='black', linestyle=':', label='reality')
    ax2.scatter(dev_z, dev_y, linewidth=1, color='r')
    # ax2.legend(fontsize=30)
    plt.axis('equal')
    plt.show()

def draw_loss(loss_all):
    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot()
    ax2.tick_params(labelsize=30)
    ax2.set_xlabel("time", fontsize=35)
    ax2.set_ylabel("gx_sum", fontsize=35)
    ax2.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax2.spines['left'].set_linewidth(2)
    ax2.spines['right'].set_color('none')
    ax2.spines['top'].set_color('none')
    dev_x = np.arange(0, len(loss_all))
    dev_y = loss_all
    ax2.plot(dev_x, dev_y, linewidth=1, color='r', label='loss')
    ax2.legend(fontsize=30)
    plt.show()


def gx_nonNormalization(gx):
    gx_demo = copy.deepcopy(gx)
    for i in range(len(gx_demo)):
        gx_demo[i][0]=gx_demo[i][0]*6-1
        gx_demo[i][1] = gx_demo[i][1] * 3-1
        gx_demo[i][2] = gx_demo[i][2] * 0.05
        gx_demo[i][3] = gx_demo[i][3] * 0.05
    return gx_demo

def get_info():
    path_memo = f"D:\desktop\os\optimization of structure\optimization of structure\optimization of structure\out_all_memorize_case4\memorize_infor_{num_var}_{modular_num}_{time}.xlsx"
    path_pred = f"D:\desktop\os\optimization of structure\optimization of structure\optimization of structure\out_all_prediction_case4\prediction_infor_{num_var}_{modular_num}_{time}.xlsx"

    gx_prediction = pd.read_excel(io=path_pred, sheet_name=f"gx_prediction_{num}")
    gx_prediction_data = gx_prediction.values.tolist()

    gx_nor = pd.read_excel(io=path_memo, sheet_name="memorize_gx_nor")
    gx_nor_data = gx_nor.values.tolist()

    loss = pd.read_excel(io=path_memo, sheet_name="memorize_loss")
    loss_data = loss.values.tolist()

    loss_all = []
    for i in range(len(loss_data)):
        loss_all.extend(loss_data[i])
    # draw_loss(loss_all)
    draw_gx(gx_nor_data,gx_prediction_data)



num_var = 5
modular_num = 3
time = 9
num_pred = 7
gx_cha_num = []
for num in range(num_pred):
    get_info()
print(gx_cha_num)
