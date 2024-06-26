import pandas as pd
import copy
import numpy as np
from CNN import create_model
import data_to_json as dj
from random import randint
import model_to_sap as ms
import os
import modular_utils as md
import sap_run as sr
import threading
import queue
import configparser
import comtypes.client
import random

from keras.callbacks import EarlyStopping,LearningRateScheduler
import sys
import xlsxwriter
import matplotlib.pyplot as plt
#修改区间

# 定义余弦退火学习率调度函数
def cosine_annealing(epoch, T_max, eta_min=0, eta_max=0.001):
    lr = eta_min + (eta_max - eta_min) * (1 + np.cos(np.pi * epoch / T_max)) / 2
    return lr

#修改区间
def gx_Normalization(gx,gx_data_select):
    gx_demo = copy.deepcopy(gx)
    for i in range(len(gx_demo)):
        for j in range(len(gx_data_select)):
            if gx_data_select[j] == 0:
                if gx_demo[i][j]>=2:
                    gx_demo[i][j]=1
                elif gx_demo[i][j]<=-1:
                    gx_demo[i][j] = 0
                elif gx_demo[i][j]<=2 and gx_demo[i][j]>=-1:
                    gx_demo[i][j]=(gx_demo[i][j]+1)/3
            elif gx_data_select[j] == 1:
                if gx_demo[i][j]>=3:
                    gx_demo[i][j]=1
                elif gx_demo[i][j]<=-1:
                    gx_demo[i][j] = 0
                elif gx_demo[i][j]<=3 and gx_demo[i][j]>=-1:
                    gx_demo[i][j]=(gx_demo[i][j]+1)/4
            elif gx_data_select[j] == 2:
                if gx_demo[i][j] >= 0.007:
                    gx_demo[i][j] = 1
                else:
                    gx_demo[i][j] = gx_demo[i][j] / 0.007
            elif gx_data_select[j] == 3:
                if gx_demo[i][j] >= 0.01:
                    gx_demo[i][j] = 1
                else:
                    gx_demo[i][j] = gx_demo[i][j] / 0.01
            elif gx_data_select[j] == 5:
                if gx_demo[i][j] >= 600:
                    gx_demo[i][j] = 1
                elif gx_demo[i][j] <= 0:
                    gx_demo[i][j] = 0
                elif gx_demo[i][j] <= 600 and gx_demo[i][j] >= 0:
                    gx_demo[i][j] = gx_demo[i][j] / 600
    return gx_demo

def gx_Normalization_1(gx):
    gx_demo = copy.deepcopy(gx)
    for i in range(len(gx_demo)):
        if gx_demo[i][0]>=2:
            gx_demo[i][0]=1
        elif gx_demo[i][0]<=-1:
            gx_demo[i][0] = 0
        elif gx_demo[i][0]<=2 and gx_demo[i][0]>=-1:
            gx_demo[i][0]=(gx_demo[i][0]+1)/3
        if gx_demo[i][1]>=3:
            gx_demo[i][1]=1
        elif gx_demo[i][1]<=-1:
            gx_demo[i][1] = 0
        elif gx_demo[i][1]<=3 and gx_demo[i][1]>=-1:
            gx_demo[i][1]=(gx_demo[i][1]+1)/4
        if gx_demo[i][2] >= 0.007:
            gx_demo[i][2] = 1
        else:
            gx_demo[i][2] = gx_demo[i][2] / 0.007
        if gx_demo[i][3] >= 0.01:
            gx_demo[i][3] = 1
        else:
            gx_demo[i][3] = gx_demo[i][3] / 0.01
        if gx_demo[i][5] >= 600:
            gx_demo[i][5] = 1
        elif gx_demo[i][5] <= 0:
            gx_demo[i][5] = 0
        elif gx_demo[i][5] <= 600 and gx_demo[i][5] >= 0:
            gx_demo[i][5] = gx_demo[i][5] / 600
    return gx_demo


#输入输出修改,

# 验证测试集
def DNN_GA_test(memorize_pool_local,memorize_gx_local,memorize_pool,memorize_gx,num_var,num_room_type,num_ind,best_indivi,run_time,model):
    global gx_test_data1,pop_test_data
    # 早停法训练深度深度网络
    early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    # 定义学习率调度回调
    T_max = 50  # 余弦退火周期
    lr_scheduler = LearningRateScheduler(lambda epoch: cosine_annealing(epoch, T_max, eta_min=0.0001, eta_max=0.001))

    #局部训练
    if len(memorize_pool_local)!=0:
        pool_local = copy.deepcopy(memorize_pool_local)
        x_train1_local = np.array(pool_local)
        x_train_local = x_train1_local#提取用于训练的x_train部分
        gx_local = copy.deepcopy(memorize_gx_local)
        y_train_local = np.array(gx_local)
        y_train_local= gx_Normalization(y_train_local,gx_data_select)#归一化
        # model= create_model(len(x_train_local[0]), len(y_train_local[0]))#创建模型

        #verbose取消打印损失
        his = model.fit(x_train_local, y_train_local, epochs=600, batch_size=32,verbose=0,callbacks=[early_stopping,lr_scheduler])#训练模型

    #全局训练
    pool_global = copy.deepcopy(memorize_pool)
    gx_global = copy.deepcopy(memorize_gx)
    x_train1 = np.array(pool_global)
    x_train = x_train1#提取用于训练的x_train部分
    y_train = np.array(gx_global)
    y_train = gx_Normalization(y_train,gx_data_select)#归一化
    # model = create_model(len(x_train[0]),len(y_train[0]))#创建模型
    # early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history=model.fit(x_train, y_train, epochs=600, batch_size=32,verbose=0,callbacks=[early_stopping,lr_scheduler])#训练模型
    # history_loss.extend(history.history['loss'])
    # history_mae.extend(history.history['mae'])
    # history_loss.append(history.history['loss'][len(history.history['loss'])-1])
    # history_mae.append(history.history['mae'][len(history.history['loss'])-1])
    history_loss.append(history.history['loss'])
    history_mae.append(history.history['mae'])

    pop_test_data_temp = []
    gx_test_data_temp = []
    num_test = range(0, len(pop_all_read))
    num_test_data = random.sample(num_test, 20)
    # 生成测试集
    for i in num_test_data:
        pop_test_data_temp.append(pop_all_read[i])
        gx_test_data_temp.append(gx_all_read[i])
    pop_test_data.extend(pop_test_data_temp)
    gx_test_data1.extend(gx_test_data_temp)
    #测试机验证
    pop_test = copy.deepcopy(pop_test_data_temp)
    pop_test=np.array(pop_test)
    fitness_test = model.predict(pop_test, verbose=0)
    gx_test_data_all.extend(fitness_test.tolist())

    # pop_best = []
    # fitness_best=[]#新增内容
    # for i in range(num_ind):
    #     pop1 = generation_population_modular_section(best_indivi, 0.15)#根据最好个体生成种群
    #     pop2 = copy.deepcopy(pop1)
    #     pop2,fitness_best = GA_for_DNN(run_time, pop2, model,fitness_best)
    #     pop_best.append(pop2[0].tolist())
    # pop_best = np.array(pop_best)
    # return pop_best,model,

#按照局部记忆池划分全局记忆池
def get_local_global_data(file_time):
    path_infor = f"D:\desktop\os\optimization of structure\optimization of structure\optimization of structure\out_all_infor_case4\\run_infor_{num_var}_{modular_num}_{file_time}.xlsx"
    path_memo = f"D:\desktop\os\optimization of structure\optimization of structure\optimization of structure\out_all_memorize_case4\memorize_infor_{num_var}_{modular_num}_{file_time}.xlsx"
    global_get_pop = pd.read_excel(io=path_memo, sheet_name="memorize_pool",header=None)
    global_get_pop = global_get_pop.values.tolist()
    local_pop = pd.read_excel(io=path_memo, sheet_name="memorize_num",header=None)
    local_pop = local_pop.values.tolist()
    global_get_gx = pd.read_excel(io=path_memo, sheet_name="memorize_gx",header=None)
    global_get_gx = global_get_gx.values.tolist()
    # global_get_gx = gx_Normalization(global_get_gx,gx_data_select)

    local_num = []
    for i in range(len(local_pop)):
        local_num.append(local_pop[i][0])
    local_each_num = [local_num[0]]
    for i in range(1,len(local_num)):
        local_each_num.append(local_num[i]-local_num[i-1])

    pop2_remove = []
    pop2_best1 = pd.read_excel(io=path_infor, sheet_name="pop2_all", header=None)
    pop2_pool_all = pop2_best1.values.tolist()
    for i in range(len(pop2_pool_all)):
        if i <= len(pop2_pool_all):
            if type(pop2_pool_all[i][0]) == str:
                pop2_remove.append(i)

    for i in range(len(pop2_remove)):
        pop2_pool_all.remove(pop2_pool_all[int(pop2_remove[len(pop2_remove) - 1 - i])])

    pop_best = []
    for i in range(len(local_num)):
        pop_best.append(pop2_pool_all[i*20*30+19*30])



    local_memorize_pop = []
    local_memorize_gx = []
    for i in range(len(local_num)):
        if i ==0:
            local_memorize_pop.append(global_get_pop[0:local_num[i]])
            local_memorize_gx.append(global_get_gx[0:local_num[i]])
        else:
            local_memorize_pop.append(global_get_pop[local_num[i-1]:local_num[i]])
            local_memorize_gx.append(global_get_gx[local_num[i-1]:local_num[i]])

    return local_memorize_pop,local_memorize_gx,pop_best,global_get_gx,global_get_pop


#验证测试集
def run_DNN_GA_test(local_pop1,local_gx1,pop_best1):
    local_gx = copy.deepcopy(local_gx1)
    local_pop = copy.deepcopy(local_pop1)
    pop_best = copy.deepcopy(pop_best1)
    global_pop = []
    global_gx = []
    all_pop2=[]
    fit_pred_all =[]
    pop_best=np.array(pop_best)
    for i in range(len(local_pop)):
        global_pop.extend(local_pop[i])
        global_gx.extend(local_gx[i])
        global_pop_train = copy.deepcopy(global_pop)
        global_gx_train = copy.deepcopy(global_gx)
        local_pop_train = copy.deepcopy(local_pop[i])
        local_gx_train = copy.deepcopy(local_gx[i])
        model = create_model(len(local_pop[0][0]), len(local_gx[0][0]))  # 创建模型
        DNN_GA_test(local_pop_train, local_gx_train, global_pop_train, global_gx_train, num_var, num_room_type, 10,
               pop_best[i], 100,model)

        print(f'完成进度{i+1}/{len(local_pop)}')
    return all_pop2,fit_pred_all,DNN_prediction_fitness,gx_pred_best,all_fit_pred_GA,pop_last


def draw_test_data(truth_data1, pred_data1, draw_time):
 truth_data = copy.deepcopy(truth_data1)
 pred_data = copy.deepcopy(pred_data1)

 truth_draw_data = []
 pred_draw_data = []

 for i in range(len(pred_data[0])):
  temp1 = []
  temp2 = []
  for j in range(len(truth_data)):
   temp1.append(truth_data[j][i])
   temp2.append(pred_data[j][i])
  truth_draw_data.append(temp1)
  pred_draw_data.append(temp2)

 draw_x = np.arange(0, len(truth_draw_data[0]))
 color_data = ['b', 'g', 'r', 'c', 'k', 'm']
 # 绘图
 plt.clf()
 # plt.title('test data')  # 折线图标题
 plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示汉字
 plt.xlabel('number', fontsize=20)  # x轴标题
 plt.ylabel('Nor', fontsize=20)  # y轴标题
 plt.xlim((0, 140))
 plt.ylim((0, 1))
 my_x_ticks = np.arange(0, 140, 20)
 my_y_ticks = np.arange(0, 1, 0.1)
 plt.xticks(my_x_ticks, size=15)
 plt.yticks(my_y_ticks, size=15)

 for i in draw_time:
  plt.plot(draw_x, truth_draw_data[i], marker='o', markersize=4, color=color_data[i], linewidth=1.5, linestyle="-")
  plt.plot(draw_x, pred_draw_data[i], marker='x', markersize=4, color=color_data[i], linewidth=1.5, linestyle="--")

 # for a, b in zip(x, y1):
 #     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)  # 设置数据标签位置及大小
 # for a, b in zip(x, y2):
 #     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
 # for a, b in zip(x, y3):
 #     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
 # for a, b in zip(x, y4):
 #     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
 # for a, b in zip(x, y5):
 #     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
 # la = ['truth_data']
 # for i in range(len(truth_draw_data)):
 #     la.append(f'pred_data_{i}')
 # plt.legend(la)  # 设置折线名称

 plt.show()
 # plt.close()
 # return pred_draw_data





# generate model

POP_SIZE = 30
CROSSOVER_RATE = 0.6
MUTATION_RATE = 0.1
N_GENERATIONS = 140
num_thread = 5

pop_last = []
num_room_type=1

POP_SIZE=30
num_var = 5
modular_num= 3
file_time = 15

history_loss = []
history_mae = []


sap_run_time00 = 0

#新增
gx_all_truth = []

gx_pred_best=[]#每次预测得到的

gx_test_data_all =[]#测试集每次预测得到的结果
#局部记忆池

DNN_prediction_fitness = []

gx_data_select = [0,1,2,3,4,5]
all_fit_pred_GA=[]
pop_test_data = []
gx_test_data1 = []


#获得每隔N代的新种群finess，与真实fitness排序
# fit_pred,fit_truth,pop_min_pred = get_DNN_GA(file_time,27,20)
# sort_pred,sort_truth=fit_sort(fit_pred,fit_truth)
#静态训练以及sap计算对比
# pop2_best,memorize_pool,memorize_fit,memorize_weight,memorize_gx,gx_prediction,memorize_loss,memorize_mae,memorize_gx_nor,memorize_num=get_continue_data(file_time,num_continue)
# fitness_prediction2,fitness,DNN_prediction_fitness,gx_truth_all,gx_pred_all=get_pred_fit(pop2_best,10,400)
#在线训练神经网络生成最优个体
local_memorize_pop,local_memorize_gx,pop_best,gx_all_read,pop_all_read=get_local_global_data(file_time)



all_pop2,fit_pred_all,DNN_prediction_fitness,gx_pred_best,all_fit_pred_GA,pop_last=run_DNN_GA_test(local_memorize_pop,local_memorize_gx,pop_best)

gx_demo1 = copy.deepcopy(gx_test_data1)
# for i in range(len(gx_test_data1)):
for i in range(50):
    if gx_demo1[i][0] >= 2:
        gx_demo1[i][0] = 1
    elif gx_demo1[i][0] <= -1:
        gx_demo1[i][0] = 0
    elif gx_demo1[i][0] <= 2 and gx_demo1[i][0] >= -1:
        gx_demo1[i][0] = (gx_demo1[i][0] + 1) / 3
    if gx_demo1[i][1] >= 3:
        gx_demo1[i][1] = 1
    elif gx_demo1[i][1] <= -1:
        gx_demo1[i][1] = 0
    elif gx_demo1[i][1] <= 3 and gx_demo1[i][1] >= -1:
        gx_demo1[i][1] = (gx_demo1[i][1] + 1) / 4
    if gx_demo1[i][2] >= 0.007:
        gx_demo1[i][2] = 1
    else:
        gx_demo1[i][2] = gx_demo1[i][2] / 0.007
    if gx_demo1[i][3] >= 0.01:
        gx_demo1[i][3] = 1
    else:
        gx_demo1[i][3] = gx_demo1[i][3] / 0.01
    if gx_demo1[i][5] >= 600:
        gx_demo1[i][5] = 1
    elif gx_demo1[i][5] <= 0:
        gx_demo1[i][5] = 0
    elif gx_demo1[i][5] <= 600 and gx_demo1[i][5] >= 0:
        gx_demo1[i][5] = gx_demo1[i][5] / 600
    print(f'{i}次——{gx_demo1[i]}')

gounvyangtingting = range(len(gx_test_data1))




gx_test_truth_data = gx_Normalization_1(gx_test_data1)
gx_tzhanjiaqi =gx_Normalization_1(gx_test_data1)
draw_time = [5]
draw_test_data(gx_demo1,gx_test_data_all,draw_time)
