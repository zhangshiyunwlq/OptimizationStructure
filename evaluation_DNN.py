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
import sys
import xlsxwriter
import matplotlib.pyplot as plt
#修改区间

def gx_nonNormalization(gx,gx_data_select):
    gx_demo = copy.deepcopy(gx)
    for i in range(len(gx_demo)):
        for j in range(len(gx_data_select)):
            if gx_data_select[j] == 0:
                gx_demo[i][j] = gx_demo[i][j] * 3 - 1
            elif gx_data_select[j] == 1:
                gx_demo[i][j] = gx_demo[i][j] * 4 - 1
            elif gx_data_select[j] == 2:
                gx_demo[i][j] = gx_demo[i][j] * 0.01
            elif gx_data_select[j] == 3:
                gx_demo[i][j] = gx_demo[i][j] * 0.01
            elif gx_data_select[j] == 5:
                gx_demo[i][j] = gx_demo[i][j] * 600
        # gx_demo[i][1] = gx_demo[i][1] * 1.5-1
        # gx_demo[i][2] = gx_demo[i][2] * 0.1
        # gx_demo[i][3] = gx_demo[i][3] * 0.1
        # gx_demo[i][5] = gx_demo[i][5] * 200+400
        # gx_demo[i][0] = gx_demo[i][0] * 600
    return gx_demo
#修改区间
def Gx_convert(fitness1,gx_data_select):
    fitness3 = copy.deepcopy(fitness1)
    fitness4 = []  # 储存所有gx
    fitness2 = []  # 所有gx的和
    for j in range(len(fitness3)):
        fitness4.append(fitness3[j].tolist())
    fitness4=gx_nonNormalization(fitness4,gx_data_select)
    fitness5 = copy.deepcopy(fitness4)
    for j in range(len(fitness3)):
        # fitness2.append(sum(fitness4[j]))
        for z in range(len(gx_data_select)):
            if gx_data_select[z] == 0:
                if fitness5[j][z]<=0:
                    fitness5[j][z] =0
            elif gx_data_select[z] == 1:
                if fitness5[j][z] <= 0:
                    fitness5[j][z] = 0
            elif gx_data_select[z] == 2:
                if fitness5[j][z]<=0.00167 and fitness5[j][z] >= -0.00167:
                    fitness5[j][z] =0
                else:
                    fitness5[j][z] = 100*abs(fitness5[j][z])
            elif gx_data_select[z] == 3:
                if fitness5[j][z] <= 0.004 and fitness5[j][z] >= -0.004:
                    fitness5[j][z] = 0
                else:
                    fitness5[j][z] = 100*abs(fitness5[j][z])
            elif gx_data_select[z] == 4:
                if fitness5[j][z] <= 0.4:
                    fitness5[j][z] = 0
                else:
                    fitness5[j][z] = 100 * abs(fitness5[j][z])
        value = 0
        for z in range(len(gx_data_select)):
            if gx_data_select[z] != 5:
                value += 10000*fitness5[j][z]
            elif gx_data_select[z] == 5:
                value += fitness5[j][z]
        fitness2.append(value)
        # if fitness5[j][0]<=0:
        #     fitness5[j][0] =0
        # if fitness5[j][1] <= 0:
        #     fitness5[j][1] = 0
        # if fitness5[j][2]<=0.00167 and fitness5[j][2] >= -0.00167:
        #     fitness5[j][2] =0
        # else:
        #     fitness5[j][2] = abs(fitness5[j][2])
        # if fitness5[j][3] <= 0.004 and fitness5[j][3] >= -0.004:
        #     fitness5[j][3] = 0
        # else:
        #     fitness5[j][3] = abs(fitness5[j][3])
        # if fitness5[j][4] <= 0.4:
        #     fitness5[j][4] = 0
        # fitness2.append(fitness5[j][5]+10000*(fitness5[j][0]+fitness5[j][1]+fitness5[j][2]*100+fitness5[j][3]*100+100*abs(fitness5[j][4])))
        # if fitness5[j][0]<=0:
        #     fitness5[j][0] = abs(fitness5[j][0])*100
        # fitness2.append(fitness5[j][0])
    return fitness2
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
                if gx_demo[i][j] >= 0.01:
                    gx_demo[i][j] = 1
                else:
                    gx_demo[i][j] = gx_demo[i][j] / 0.01
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
def select_2(pop, fitness):  # nature selection wrt pop's fitness

    fit_ini = copy.deepcopy(fitness)
    luyi = copy.deepcopy(fitness)
    luyi.sort(reverse=True)
    sort_num = []
    for i in range(len(fit_ini)):
        sort_num.append(luyi.index(fit_ini[i]))
    # print(sort_num)
    # print(f'{len(sort_num)}_{len(pop)}')
    for i in range(len(sort_num)):
        if sort_num[i]==0:
            sort_num[i]+=0.01
    pop_last.append(pop)



    # for i in range(len(list_new)):
    #     list_new[i] = m.e ** (list_new[i] * 1.5)
    idx = np.random.choice(np.arange(len(pop)), size=len(pop), replace=True,
                           p=np.array(sort_num) / (sum(sort_num)))
    pop2 = np.zeros((len(pop), len(pop[0])))
    for i in range(len(pop2)):
        pop2[i] = pop[int(idx[i])]
    return pop2
#处理交叉后出现相同截面的问题
def crossover_and_mutation_GA_for_DNN(pop2,num_var,CROSSOVER_RATE,MUTATION_RATE):
    pop = pop2

    new_pop = np.zeros((len(pop),len(pop[0])))
    for i in range(len(pop)):
        father = pop[i]
        child = father
        if np.random.rand() < CROSSOVER_RATE:
            mother = pop[np.random.randint(len(pop2))]
            cross_points1 = np.random.randint(low=0, high=len(pop[0]))
            cross_points2 = np.random.randint(low=0, high=len(pop[0]))
            while cross_points2==cross_points1:
                cross_points2 = np.random.randint(low=0, high=len(pop[0]))
            exchan = []
            exchan.append(cross_points2)
            exchan.append(cross_points1)
            for j in range(min(exchan),max(exchan)):
                child[j] = mother[j]
        mutation_1_stort_modular_section(num_room_type,num_var,child,MUTATION_RATE)
        new_pop[i] = child

    for i in range(len(new_pop)):

        sec_sort = []
        room_sort = []
        for j in range(num_var):
            sec_sort.append(new_pop[i][j])
        sec_sort.sort()
        for j in range(num_var):
            new_pop[i][j] = sec_sort[j]

        for mutate_point in range(num_var):
            x_var = list(map(int, x.tolist()))
            for mutate_point_1 in range(num_var):
                if pop[i][mutate_point_1] in x_var:
                    x_var.remove(pop[i][mutate_point_1])
            if pop[i][mutate_point] == pop[i][
                mutate_point + 1] and mutate_point <= num_var - 2:  # 以MUTATION_RATE的概率进行变异
                pop[i][mutate_point] = np.random.choice(x_var)

        sec_sort = []
        room_sort = []
        for j in range(num_var):
            sec_sort.append(new_pop[i][j])
        sec_sort.sort()
        for j in range(num_var):
            new_pop[i][j] = sec_sort[j]

    return new_pop

def mutation_1_stort_modular_section(num_room_type,num_var,child,MUTATION_RATE):

    num_var = int(num_var)
    room_nu = np.linspace(1, 12, 12)
    num_room_type = int(num_room_type)
    if num_var!=len(x):
        for mutate_point in range(num_var):
            x_var = list(map(int, x.tolist()))
            for mutate_point_1 in range(num_var):
                if child[mutate_point_1] in x_var:
                    x_var.remove(child[mutate_point_1])
            if np.random.rand() < MUTATION_RATE:  # 以MUTATION_RATE的概率进行变异
                child[mutate_point] = np.random.choice(x_var)

    for j in range(num_var,num_var+num_room_type):
        if np.random.rand() < MUTATION_RATE:
            child[j] = randint(1,3)
    for j in range(num_var+num_room_type,num_var+num_room_type+section_num):
        if np.random.rand() < MUTATION_RATE:
            child[j] = randint(0,num_var-1)
    for j in range(num_var+num_room_type+section_num,num_var+num_room_type+section_num+brace_num):
        if np.random.rand() < MUTATION_RATE:
            child[j] = randint(0,1)
    for j in range(num_var + num_room_type + section_num+ brace_num, num_var + num_room_type + section_num + brace_num+zone_num):
        if np.random.rand() < MUTATION_RATE:
            child[j] = randint(0, modular_num-1)
            # child[j] = 0

#修改输入输出接口
def GA_for_DNN(run_time,pop2,model,fitness_best):
    fitness_pred = []
    for i in range(run_time):
        temp = []
        fitness1 = model.predict(pop2,verbose=0)
        fitness2 = Gx_convert(fitness1,gx_data_select)#归一化还原，并将每个染色体对应的gx累加
        all_fit_pred_GA.append(fitness2)
        mm = fitness2.index(min(fitness2))
        min1 = min(fitness2)
        temp.append(fitness1[mm])
        fitness_pred.append(min1)
        mm2_all = pop2[mm]
        #选择
        pop2 = select_2(pop2, fitness2)
        # 交叉变异
        pop2 = crossover_and_mutation_GA_for_DNN(pop2, num_var,CROSSOVER_RATE,MUTATION_RATE)
        fit_pred = model.predict(pop2,verbose=0)
        fit_pred2=Gx_convert(fit_pred,gx_data_select)
        if min1 <= fit_pred2[0]:
            pop2[0] = mm2_all
    gx_pred_best.append(temp)
    DNN_prediction_fitness.append(fitness_pred)
    fitness_best.append(min(min1,fit_pred2[0]))
    return pop2,fitness_best

def generation_population_modular_section(best_indivi,rate):

    best_in = copy.deepcopy(best_indivi)
    best_in.tolist()
    pop = np.zeros((50,len(best_in)))
    for i in range(len(pop)):
        if num_var != len(x):
            for mutate_point in range(num_var):
                x_var = list(map(int, x.tolist()))
                for mutate_point_1 in range(num_var):
                    if pop[i][mutate_point_1] in x_var:
                        x_var.remove(pop[i][mutate_point_1])
                if np.random.rand() <0.25:  # 以MUTATION_RATE的概率进行变异
                    pop[i][mutate_point] = np.random.choice(x_var)
                else:
                    pop[i][mutate_point] = best_in[mutate_point]
        for j in range(num_var, num_var + num_room_type):
            if np.random.rand() < 0.25:
                pop[i][j] = randint(1, 3)
            else:
                pop[i][j] = best_in[j]
        for j in range(num_var + num_room_type, num_var + num_room_type + section_num):
            if np.random.rand() < 0.25:
                pop[i][j] = randint(0, num_var - 1)
            else:
                pop[i][j] = best_in[j]
        for j in range(num_var + num_room_type + section_num, num_var + num_room_type + section_num + brace_num):
            if np.random.rand() < 0.25:
                pop[i][j] = randint(0, 1)
            else:
                pop[i][j] = best_in[j]
        for j in range(num_var + num_room_type + section_num + brace_num,
                       num_var + num_room_type + section_num + brace_num + zone_num):
            if np.random.rand() < 0.25:
                pop[i][j] = randint(0, modular_num - 1)
            else:
                pop[i][j] = best_in[j]

    new_pop = copy.deepcopy(pop)
    for i in range(len(new_pop)):

        sec_sort = []
        room_sort = []
        for j in range(num_var):
            sec_sort.append(new_pop[i][j])
        sec_sort.sort()
        for j in range(num_var):
            new_pop[i][j] = sec_sort[j]

        for mutate_point in range(num_var):
            x_var = list(map(int, x.tolist()))
            for mutate_point_1 in range(num_var):
                if pop[i][mutate_point_1] in x_var:
                    x_var.remove(pop[i][mutate_point_1])
            if pop[i][mutate_point] ==pop[i][mutate_point+1] and mutate_point <= num_var-2 :  # 以MUTATION_RATE的概率进行变异
                pop[i][mutate_point] = np.random.choice(x_var)

        sec_sort = []
        room_sort = []
        for j in range(num_var):
            sec_sort.append(new_pop[i][j])
        sec_sort.sort()
        for j in range(num_var):
            new_pop[i][j] = sec_sort[j]


    return new_pop


def mulit_get_sap(num_thread):
    case_name = []
    APIPath_name = []
    mySapObject_name = []
    SapModel_name = []
    ModelPath_name = []
    for i in range(num_thread):
        case_name.append(f"cases{i}")
        APIPath_name.append(os.path.join(os.getcwd(), f"cases{i}"))
        mySapObject_name.append(f"mySapObject{i}")
        SapModel_name.append(f"SapModel{i}")
        ModelPath_name.append(f"ModelPath{i}")
        mySapObject_name[i], ModelPath_name[i], SapModel_name[i] = SAPanalysis_GA_run2(APIPath_name[i])
    return mySapObject_name,ModelPath_name,SapModel_name

def get_continue_data(file_time,num_continue):
    path_memo = f"D:\desktop\os\optimization of structure\optimization of structure\optimization of structure\out_all_memorize_case4\memorize_infor_{num_var}_{modular_num}_{file_time}.xlsx"
    path_infor = f"D:\desktop\os\optimization of structure\optimization of structure\optimization of structure\out_all_infor_case4\\run_infor_{num_var}_{modular_num}_{file_time}.xlsx"
    gx_nor = pd.read_excel(io=path_memo, sheet_name="memorize_gx_nor",header=None)
    gx_nor_data = gx_nor.values.tolist()

    memorize_pool_pop1 = pd.read_excel(io=path_memo, sheet_name="memorize_pool",header=None)
    memorize_pool = memorize_pool_pop1.values.tolist()

    memorize_fit1 = pd.read_excel(io=path_memo, sheet_name="memorize_fit",header=None)
    memorize_fit2 = memorize_fit1.values.tolist()
    memorize_fit = []
    for i in range(len(memorize_fit2)):
        memorize_fit.append(memorize_fit2[i][0])

    memorize_weight1 = pd.read_excel(io=path_memo, sheet_name="memorize_weight",header=None)
    memorize_weight2 = memorize_weight1.values.tolist()
    memorize_weight = []
    for i in range(len(memorize_weight2)):
        memorize_weight.append(memorize_weight2[i][0])

    memorize_gx1 = pd.read_excel(io=path_memo, sheet_name="memorize_gx",header=None)
    memorize_gx = memorize_gx1.values.tolist()

    gx_prediction1 = pd.read_excel(io=path_memo, sheet_name="gx_prediction",header=None)
    gx_prediction = gx_prediction1.values.tolist()

    memorize_loss1 = pd.read_excel(io=path_memo, sheet_name="memorize_loss",header=None)
    memorize_loss = memorize_loss1.values.tolist()

    memorize_mae1 = pd.read_excel(io=path_memo, sheet_name="memorize_mae",header=None)
    memorize_mae = memorize_mae1.values.tolist()

    memorize_gx_nor1 = pd.read_excel(io=path_memo, sheet_name="memorize_gx_nor",header=None)
    memorize_gx_nor = memorize_gx_nor1.values.tolist()

    memorize_num1 = pd.read_excel(io=path_memo, sheet_name="memorize_num",header=None)
    memorize_num2 = memorize_num1.values.tolist()
    memorize_num = []
    for i in range(len(memorize_num2)):
        memorize_num.append(memorize_num2[i][0])

    pop2_best1 = pd.read_excel(io=path_infor, sheet_name="pop2_all",header=None)
    pop2_fitness1 = pd.read_excel(io=path_infor, sheet_name="pop_all_fitness",header=None)
    pop2_pool_all = pop2_best1.values.tolist()
    fitness_pool_all = pop2_fitness1.values.tolist()
    pop2_remove = []
    fitness_remove = []
    for i in range(len(pop2_pool_all)):
        if i <= len(pop2_pool_all):
            if type(pop2_pool_all[i][0]) == str:
                pop2_remove.append(i)

    for i in range(len(fitness_pool_all)):
        if i <= len(pop2_pool_all):
            if type(fitness_pool_all[i][0]) == str:
                fitness_remove.append(i)

    for i in range(len(pop2_remove)):
        pop2_pool_all.remove(pop2_pool_all[int(pop2_remove[len(pop2_remove) - 1 - i])])

    for i in range(len(fitness_remove)):
        fitness_pool_all.remove(fitness_pool_all[int(fitness_remove[len(fitness_remove) - 1 - i])])

    pop2_best = pop2_pool_all[(num_continue - 1) * POP_SIZE]
    fitness_best = fitness_pool_all[num_continue - 1][0]
    return pop2_best,memorize_pool,memorize_fit,memorize_weight,memorize_gx,gx_prediction,memorize_loss,memorize_mae,memorize_gx_nor,memorize_num

#输入输出修改,
def DNN_GA(memorize_pool_local,memorize_gx_local,memorize_pool,memorize_gx,num_var,num_room_type,num_ind,best_indivi,run_time):
    #局部训练
    if len(memorize_pool_local)!=0:
        pool_local = copy.deepcopy(memorize_pool_local)
        x_train1_local = np.array(pool_local)
        x_train_local = x_train1_local#提取用于训练的x_train部分
        gx_local = copy.deepcopy(memorize_gx_local)
        y_train_local = np.array(gx_local)
        y_train_local= gx_Normalization(y_train_local,gx_data_select)#归一化
        model= create_model(len(x_train_local[0]), len(y_train_local[0]))#创建模型
        #verbose取消打印损失
        model.fit(x_train_local, y_train_local, epochs=200, batch_size=32,verbose=0)#训练模型

    #全局训练
    pool_global = copy.deepcopy(memorize_pool)
    gx_global = copy.deepcopy(memorize_gx)
    x_train1 = np.array(pool_global)
    x_train = x_train1#提取用于训练的x_train部分
    y_train = np.array(gx_global)
    y_train = gx_Normalization(y_train,gx_data_select)#归一化
    model = create_model(len(x_train[0]),len(y_train[0]))#创建模型
    history=model.fit(x_train, y_train, epochs=200, batch_size=32,verbose=0)#训练模型
    # history_loss.extend(history.history['loss'])
    # history_mae.extend(history.history['mae'])
    # history_loss.append(history.history['loss'][len(history.history['loss'])-1])
    # history_mae.append(history.history['mae'][len(history.history['loss'])-1])
    history_loss.append(history.history['loss'])
    history_mae.append(history.history['mae'])
    pop_best = []
    fitness_best=[]#新增内容
    for i in range(num_ind):
        pop1 = generation_population_modular_section(best_indivi, 0.15)#根据最好个体生成种群
        pop2 = copy.deepcopy(pop1)
        pop2,fitness_best = GA_for_DNN(run_time, pop2, model,fitness_best)
        pop_best.append(pop2[0].tolist())
    pop_best = np.array(pop_best)
    return pop_best,model,fitness_best

def decoding_modular_section(pop2):

    pop_all = copy.deepcopy(pop2)
    modular_type1 = [i for i in range(3)]
    #生成对每个模块的截面编号索引
    modular_type_all= []
    for i in range(modular_num):
        modular_type_temp = []
        for j in range(len(modular_type1)):
            modular_type_temp.append(num_var+num_room_type+modular_type1[j]+3*i)
        modular_type_all.append(modular_type_temp)


    #提取截面表
    pop1_all = []
    for i in range(len(pop_all)):
        pop1_section = []
        for j in range(num_var+num_room_type+section_num+brace_num,num_var+num_room_type+section_num+brace_num+zone_num):
            for z in range(3):
                sec = int(pop_all[i][j])
                pop1_section.append(pop_all[i][int(modular_type_all[sec][z])])
        pop1_all.append(pop1_section)

    #解码pop1_1all
    pop1_decoding = []
    for i in range(len(pop1_all)):
        pop1_temp = []
        for j in range(len(pop1_all[i])):
            pop1_temp.append(pop_all[i][int(pop1_all[i][j])])
        pop1_decoding.append(pop1_temp)


    #生成支撑表
    brace_sort = [i for i in range(num_var+num_room_type+section_num,num_var+num_room_type+section_num+brace_num)]
    pop3_all = []
    for i in range(len(pop_all)):
        pop3_brace = []
        for j in range(num_var+num_room_type+section_num+brace_num,num_var+num_room_type+section_num+brace_num+zone_num):
            bra = int(pop_all[i][j])
            if pop_all[i][int(brace_sort[bra])] ==0:
                pop3_brace.append(0)
            else:
                pop3_brace.append(pop_all[i][num_var])
        pop3_all.append(pop3_brace)
    pop_3 =[]
    for j in range(len(pop_all)):
        temp2 = pop3_all[j]
        brace_all = []
        for i in range(len(labels)):
            temp1 = int(labels[i])
            brace_all.append(temp2[temp1])
        pop_3.append(brace_all)

    return pop1_decoding,pop_3

def mulit_Sap_analy_allroom(ModelPath,mySapObject, SapModel,pop_room,pop_room_label):
    # 建立房间信息
    sections_data_c1, type_keys_c1, sections_c1 = ms.get_section_info(section_type='c0',
                                                                      cfg_file_name="Steel_section_data_I_cube.ini")
    modular_building = md.ModularBuilding(nodes, room_indx, edges_all, labels, joint_hor, joint_ver, cor_edges)
    # 按房间分好节点
    modulars_of_building = modular_building.building_modulars

    modular_infos = {}
    # 每个房间定义梁柱截面信息
    sr.run_column_room_modular_section(labels,pop_room_label, modular_length_num * 2 * story_num, sections_data_c1, modular_infos, pop_room,story_num,zone_num)
    #
    for i in range(len(modulars_of_building)):
        modulars_of_building[i].Add_Info_And_Update_Modular(
            # modular_infos[modulars_of_building[i].modular_label - 1])
            modular_infos[i])
    modular_building.Building_Assembling(modulars_of_building)

    all_data = ms.Run_GA_sap_2(mySapObject, ModelPath, SapModel, modular_building,pop_room_label,200,modular_length_num,story_num)
    aa, bb, cc, dd, ee, ff, gg, hh, ii = all_data

    ret = SapModel.SetModelIsLocked(False)
    return aa,bb,cc,dd,ee,ff,gg,hh,ii

#修改gx输出增加一个
def Fun_1(weight,g_col,g_beam,dis_all,all_force,u,rate):

    g_col_max= max(g_col)
    g_beam_max = max(g_beam)

    dis_all5_abs = copy.deepcopy(dis_all[5])
    for i in range(len(dis_all5_abs)):
        dis_all5_abs[i] = abs(dis_all5_abs[i])

    dis_all7_abs = copy.deepcopy(dis_all[7])
    for i in range(len(dis_all7_abs)):
        dis_all7_abs[i] = abs(dis_all7_abs[i])

    dis_all_max = max(dis_all5_abs)
    interdis_max = max(dis_all7_abs)

    rate_nonzero = copy.deepcopy(rate)
    if rate_nonzero<=0.4:
        rate_nonzero =0
    else:
        rate_nonzero=rate_nonzero

    if g_col_max<=0:
        g_col_fit = 0
    else:
        g_col_fit = g_col_max

    if g_beam_max<=0:
        g_beam_fit =0
    else:
        g_beam_fit = g_beam_max

    if dis_all_max<= 0.00167 and dis_all_max >= -0.00167:
        dis_all_fit = 0
    else:
        dis_all_fit = dis_all_max

    if interdis_max<= 0.004 and interdis_max >= -0.004:
        interdis_all_fit = 0
    else:
        interdis_all_fit = interdis_max

    G_value=u * (g_col_fit + g_beam_fit + 100*dis_all_fit + 100*interdis_all_fit +rate_nonzero)
    value_jisuan = [g_col_fit,g_beam_fit,100*dis_all_fit,100*interdis_all_fit,rate_nonzero,weight]
    # gx = [g_col_max,g_beam_max,dis_all_max,interdis_max,rate,weight]
    gx = []
    for z in range(len(gx_data_select)):
        if gx_data_select[z] ==0:
            gx.append(g_col_max)
        elif gx_data_select[z] ==1:
            gx.append(g_beam_max)
        elif gx_data_select[z] ==2:
            gx.append(dis_all_max)
        elif gx_data_select[z] ==3:
            gx.append(interdis_max)
        elif gx_data_select[z] ==4:
            gx.append(rate)
        elif gx_data_select[z] ==5:
            gx.append(weight)
    # gx_Normalization = [g_col_all,g_beam_all,Y_dis_radio_all,Y_interdis_all]
    # result = weight + G_value

    value = 0
    for z in range(len(gx_data_select)):
        if gx_data_select[z] != 5:
            value += 10000 * value_jisuan[int(gx_data_select[z])]
        elif gx_data_select[z] == 5:
            value += weight

    result = value

    gx_demo = copy.deepcopy(gx)
    for j in range(len(gx_demo)):
        if gx_data_select[j] == 0:
            if gx_demo[j] >= 2:
                gx_demo[j] = 1
            elif gx_demo[j] <= -1:
                gx_demo[j] = 0
            elif gx_demo[j] <= 2 and gx_demo[j] >= -1:
                gx_demo[j] = (gx_demo[j] + 1) / 3
        elif gx_data_select[j] == 1:
            if gx_demo[j] >= 3:
                gx_demo[j] = 1
            elif gx_demo[j] <= -1:
                gx_demo[j] = 0
            elif gx_demo[j] <= 3 and gx_demo[j] >= -1:
                gx_demo[j] = (gx_demo[j] + 1) / 4
        elif gx_data_select[j] == 2:
            if gx_demo[j] >= 0.01:
                gx_demo[j] = 1
            else:
                gx_demo[j] = gx_demo[j] / 0.01
        elif gx_data_select[j] == 3:
            if gx_demo[j] >= 0.01:
                gx_demo[j] = 1
            else:
                gx_demo[j] = gx_demo[j] / 0.01
        elif gx_data_select[j] == 5:
            if gx_demo[j] >= 600:
                gx_demo[j] = 1
            elif gx_demo[j] <= 0:
                gx_demo[j] = 0
            elif gx_demo[j] <= 600 and gx_demo[j] >= 0:
                gx_demo[j] = (gx_demo[j]) / 600
    # if gx_demo[0]>=2:
    #     gx_demo[0]=1
    # elif gx_demo[0]<=-1:
    #     gx_demo[0] = 0
    # elif gx_demo[0]<=2 and gx_demo[0]>=-1:
    #     gx_demo[0]=(gx_demo[0]+1)/3
    # if gx_demo[1]>=0.5:
    #     gx_demo[1]=1
    # elif gx_demo[1]<=-1:
    #     gx_demo[1] = 0
    # elif gx_demo[1]<=0.5 and gx_demo[1]>=-1:
    #     gx_demo[1]=(gx_demo[1]+1)/1.5
    # if gx_demo[2] >= 0.1:
    #     gx_demo[2] = 1
    # else:
    #     gx_demo[2] = gx_demo[2] / 0.1
    # if gx_demo[3] >= 0.1:
    #     gx_demo[3] = 1
    # else:
    #     gx_demo[3] = gx_demo[3] / 0.1
    # if gx_demo[0] >= 600:
    #     gx_demo[0] = 1
    # elif gx_demo[0] <= 0:
    #     gx_demo[0] = 0
    # elif gx_demo[0] <= 600 and gx_demo[0] >= 0:
    #     gx_demo[0] = (gx_demo[0])/600
    return result,weight,gx,gx_demo
def get_analysis(ModelPath_name,mySapObject_name,SapModel_name,pop1,pop2,pop3):

    fit = [0 for i in range(len(pop2))]
    weight = [0 for i in range(len(pop2))]
    clo_val = [0 for i in range(len(pop2))]
    beam_val = [0 for i in range(len(pop2))]
    result1,weight_pop,clo_up_1,beam_up_1=thread_sap(ModelPath_name,mySapObject_name,SapModel_name,num_thread, pop1, pop2, pop3, fit, weight, clo_val, beam_val)
    return result1,weight_pop
def SAPanalysis_GA_run2(APIPath):


    cfg = configparser.ConfigParser()
    cfg.read("Configuration.ini", encoding='utf-8')
    ProgramPath = cfg['SAP2000PATH']['dirpath']
    if not os.path.exists(APIPath):
        try:
            os.makedirs(APIPath)
        except OSError:
            pass

    AttachToInstance = False
    SpecifyPath = True

    # ModelPath = os.path.join(APIPath, 'API_1-001.sdb')
    helper = comtypes.client.CreateObject('SAP2000v1.Helper')
    helper = helper.QueryInterface(comtypes.gen.SAP2000v1.cHelper)
    if AttachToInstance:
        # attach to a running instance of SAP2000
        try:
            # get the active SapObject
            mySapObject = helper.Getobject("CSI.SAP2000.API.SapObject")
        except (OSError, comtypes.COMError):
            print("No running instance of the program found or failed to attach.")
            sys.exit(-1)
    else:
        if SpecifyPath:
            try:
                # 'create an instance of the SAPObject from the specified path
                mySapObject = helper.CreateObject(ProgramPath)
            except (OSError, comtypes.COMError):
                print("Cannot start a new instance of the program from" + ProgramPath)
                sys.exit(-1)
        else:
            try:
                # create an instance of the SapObject from the latest installed SAP2000
                mySapObject = helper.CreateObjectProgID("CSI.SAP2000.API.SapObject")
            except (OSError, comtypes.COMError):
                print("Cannot start a new instance of the program")
                sys.exit(-1)

        # start SAP2000 application
        mySapObject.ApplicationStart()

    # create SapModel object
    SapModel = mySapObject.SapModel
    # initialize model
    SapModel.InitializeNewModel()
    ModelPath = os.path.join(APIPath, 'API_1-001.sdb')
    # create new blank model
    return mySapObject,ModelPath, SapModel

#改但不要同步
def mulitrun_GA_1(ModelPath,mySapObject, SapModel,pop1,pop_all,pop3,q,result,weight_1,col_up,beam_up,sap_run_time00):
    while True:
        if q.empty():
            break
        time = q.get()
        pop = pop1[time]
        pop_room_label = pop3[time]
        pop2= pop_all[time]

        we, co, be, r1, r2, r3, r4, dis_all, force_all = mulit_Sap_analy_allroom(ModelPath, mySapObject, SapModel,
                                                                                         pop,
                                                                                         pop_room_label)
        num_zero = pop_room_label.count(0)
        nonzero_rate = (len(pop_room_label)-num_zero)/len(pop_room_label)
        res1, res2,gx,gx_demo = Fun_1(we, co, be, dis_all, force_all, 10000,nonzero_rate)

        # num3 += 1
        weight_1[time] = res2
        col_up[time] = co
        beam_up[time] = be
        result[time] = res1
        #全局记忆池
        memorize_sum.append(sum(pop2))
        memorize_pool.append(pop2)
        memorize_fit.append(res1)
        memorize_weight.append(res2)
        memorize_col.append(col_up[time])
        memorize_beam.append(beam_up[time])
        memorize_gx.append(gx)
        memorize_gx_nor.append(gx_demo)
        # 局部记忆池
        memorize_sum_local.append(sum(pop2))
        memorize_pool_local.append(pop2)
        memorize_fit_local.append(res1)
        memorize_weight_local.append(res2)
        memorize_col_local.append(col_up[time])
        memorize_beam_local.append(beam_up[time])
        gx_all_truth.append(gx)
        memorize_gx_local.append(gx)

def thread_sap(ModelPath_name,mySapObject_name,SapModel_name,num,pop1,pop2,pop3,result,weight_1,col_up,beam_up):


    pop_n = [0 for i in range(len(pop2[0]))]

    q = queue.Queue()
    threads = []
    for i in range(len(pop1)):
        q.put(i)
    for i in range(num_thread):
        if len(ModelPath_name)!=num_thread:
            for j in range(len(ModelPath_name),num_thread):
                mySapObject_name.append(f"mySapObject{j}")
                SapModel_name.append(f"SapModel{j}")
                ModelPath_name.append(f"ModelPath{j}")
                mySapObject_name[j], ModelPath_name[j], SapModel_name[j] = SAPanalysis_GA_run2(os.path.join(os.getcwd(), f"cases{j}"))
        t = threading.Thread(target=mulitrun_GA_1, args=(ModelPath_name[i],mySapObject_name[i],SapModel_name[i],pop1,pop2,pop3,q,result,weight_1,col_up,beam_up,sap_run_time00))
        t.start()
        threads.append(t)
    for i in threads:
        i.join()
    return result,weight_1,col_up,beam_up
#增加支撑百分比

def get_pred_fit(pop2_best,num_indi,num_iter):
    mySapObject_name, ModelPath_name, SapModel_name = mulit_get_sap(num_thread)

    pop2_best = np.array(pop2_best)
    pop_best, model = DNN_GA(num_var, num_room_type, num_indi, pop2_best, num_iter)
    pop1, pop3 = decoding_modular_section(pop_best)
    pop_best = pop_best.tolist()
    fitness, weight = get_analysis(ModelPath_name, mySapObject_name, SapModel_name, pop1, pop_best, pop3)
    pop2_pred = copy.deepcopy(pop_best)
    pop2_pred = np.array(pop2_pred)
    fitness_prediction = model.predict(pop2_pred, verbose=0)
    gx_pre_nor = gx_nonNormalization(fitness_prediction,gx_data_select)

    fitness_prediction2 = Gx_convert(fitness_prediction,gx_data_select)

    for i in range(len(mySapObject_name)):
        ret = mySapObject_name[i].ApplicationExit(False)
        SapModel_name[i] = None
        mySapObject_name[i] = None

    return fitness_prediction2,fitness,DNN_prediction_fitness,np.array(gx_all_truth),np.array(gx_pre_nor)


def get_test_pop2(file_time, num_continue):
    path_infor = f"D:\desktop\os\optimization of structure\optimization of structure\optimization of structure\out_all_infor_case4\\run_infor_{num_var}_{modular_num}_{file_time}.xlsx"
    pop2_best1 = pd.read_excel(io=path_infor, sheet_name="pop2_all",header=None)
    pop2_pool_all = pop2_best1.values.tolist()
    pop2_remove = []
    fitness_remove = []
    for i in range(len(pop2_pool_all)):
        if i <= len(pop2_pool_all):
            if type(pop2_pool_all[i][0]) == str:
                pop2_remove.append(i)

    for i in range(len(pop2_remove)):
        pop2_pool_all.remove(pop2_pool_all[int(pop2_remove[len(pop2_remove) - 1 - i])])

    pop_test_all = []
    for i in range(5):
        pop_test_all.append(pop2_pool_all[(num_continue - 1) * POP_SIZE + i])
    return pop_test_all
#每次DNN+GA推荐得到的个体
def get_DNN_GA(file_time,num_pred_fit,num_run_DNN):
    path_infor = f"D:\desktop\os\optimization of structure\optimization of structure\optimization of structure\out_all_infor_case4\\run_infor_{num_var}_{modular_num}_{file_time}.xlsx"
    path_memo = f"D:\desktop\os\optimization of structure\optimization of structure\optimization of structure\out_all_DNN_fitness_case4\prediction_fitness_{num_var}_{modular_num}_{file_time}.xlsx"
    pop2_best1 = pd.read_excel(io=path_infor, sheet_name="pop2_all",header=None)
    pop2_fitness1 = pd.read_excel(io=path_infor, sheet_name="pop_all_fitness",header=None)
    pop2_pool_all = pop2_best1.values.tolist()
    fitness_pool_all = pop2_fitness1.values.tolist()
    pop2_remove = []
    fitness_remove = []
    for i in range(len(pop2_pool_all)):
        if i <= len(pop2_pool_all):
            if type(pop2_pool_all[i][0]) == str:
                pop2_remove.append(i)

    for i in range(len(fitness_pool_all)):
        if i <= len(pop2_pool_all):
            if type(fitness_pool_all[i][0]) == str:
                fitness_remove.append(i)

    for i in range(len(pop2_remove)):
        pop2_pool_all.remove(pop2_pool_all[int(pop2_remove[len(pop2_remove) - 1 - i])])

    for i in range(len(fitness_remove)):
        fitness_pool_all.remove(fitness_pool_all[int(fitness_remove[len(fitness_remove) - 1 - i])])

    pop_pred_fitness = pd.read_excel(io=path_memo, sheet_name="DNN_fitness",header=None)
    pop_pred_fitness=pop_pred_fitness.values.tolist()
    pop2_DNN=[]
    run_time = 0
    for i in range(len(fitness_pool_all)):
        run_time+=1
        if run_time%num_run_DNN ==0:
            pop2_DNN.append(fitness_pool_all[i])

    pop_DNN_divided = []
    for i in range(len(pop2_DNN)):
        temp = pop2_DNN[i][3:]
        pop_DNN_divided.append(temp)

    num_time = 0
    pop2_DNN_fit = []
    fit_temp = []
    for i in range(len(pop_pred_fitness)):
        num_time+=1
        fit_temp.append(pop_pred_fitness[i][-1])
        if num_time%num_pred_fit==0:
            fit_temp.reverse()
            temp2 = fit_temp
            pop2_DNN_fit.append(temp2)
            fit_temp = []

    min_fit_index =[]
    for i in range(len(pop2_DNN_fit)):
        min_fit_index.append(pop2_DNN_fit[i].index(min(pop2_DNN_fit[i])))

    pop_min_pred = []
    for i in range(len(min_fit_index)-1):
        pop_min_pred.append(pop2_pool_all[i*20*30+19*30+3+min_fit_index[i]])
    return pop2_DNN_fit,pop_DNN_divided,pop_min_pred

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

    return local_memorize_pop,local_memorize_gx,pop_best,global_get_gx

def run_DNN_GA(local_pop1,local_gx1,pop_best1):
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
        pop2,model,fit_best=DNN_GA(local_pop_train, local_gx_train, global_pop_train, global_gx_train, num_var, num_room_type, 10,
               pop_best[i], 100)
        all_pop2.append(pop2)
        fit_pred_all.append(fit_best)
        print(f'完成进度{i+1}/{len(local_pop)}')
    return all_pop2,fit_pred_all,DNN_prediction_fitness,gx_pred_best,all_fit_pred_GA,pop_last

def get_fitness(all_pop2):
    fit_truth = []
    mySapObject_name, ModelPath_name, SapModel_name = mulit_get_sap(num_thread)
    for i in range(len(all_pop2)):
        pop_temp = copy.deepcopy(all_pop2[i])
        pop2=pop_temp.tolist()
        pop1,pop3 = decoding_modular_section(pop2)
        fitness, weight = get_analysis(ModelPath_name, mySapObject_name, SapModel_name, pop1, pop2, pop3)
        fit_truth.append(fitness)
        print(f'完成sap计算进度{i+1}/{len(all_pop2)}')
    for i in range(len(mySapObject_name)):
        ret = mySapObject_name[i].ApplicationExit(False)
        SapModel_name[i] = None
        mySapObject_name[i] = None

    return fit_truth

def fit_sort(fit_pred,fit_truth):
    fitness_pred = copy.deepcopy(fit_pred)
    fitness_truth = copy.deepcopy(fit_truth)
    fitness_pred_sort = []
    fitness_truth_sort = []
    for i in range(len(fitness_pred)):
        fitness_pred_sort.append((np.argsort(fitness_pred[i])).tolist())
        fitness_truth_sort.append((np.argsort(fitness_truth[i])).tolist())
    return fitness_pred_sort,fitness_truth_sort

def draw_min_pop(all_pop2,fit_pred,fit_truth,memorize_gx_no,memorize_gx):
    pop_pred_best = []
    pop_truth_best = []
    index = []
    for i in range(len(fit_pred)):
        ind = fit_pred[i].index(min(fit_pred[i]))
        ind2 = fit_truth[i].index(min(fit_truth[i]))
        pop_pred_best.append(all_pop2[i][ind].tolist())
        pop_truth_best.append(all_pop2[i][ind2].tolist())
        index.append(ind2)
    # pop1, pop3 = decoding_modular_section(pop_pred_best)

    gx_nor_divided = []
    gx_divided = []
    num = int(len(memorize_gx_no)/len(pop_pred_best))
    for i in range(len(pop_pred_best)):
        temp = []
        temp1 = []
        for j in range(num):
            temp.append(memorize_gx_no[i*num+j])
            temp1.append(memorize_gx[i * num + j])
        gx_nor_divided.append(temp)
        gx_divided.append(temp1)

    gx_truth_min = []
    gx_min = []
    for i in range(len(index)):
        gx_truth_min.append(gx_nor_divided[i][index[i]])
        gx_min.append(gx_divided[i][index[i]])
    return pop_pred_best,pop_truth_best,gx_truth_min,gx_min



def output_data(pop2_all,fit_truth,fit_pred_all,all_pop2,DNN_prediction_fitness,time):
    APIPath = os.path.join(os.getcwd(), 'DNN_test_data')
    SpecifyPath = True
    if not os.path.exists(APIPath):
        try:
            os.makedirs(APIPath)
        except OSError:
            pass

    path1 = os.path.join(APIPath, f'all_data_{time}')


    wb1 = xlsxwriter.Workbook(f'{path1}.xlsx')

    out_fit_pred_all = wb1.add_worksheet('fit_pred_all')
    loc = 0
    for i in range(len(fit_pred_all)):
        for j in range(len(fit_pred_all[i])):
            out_fit_pred_all.write(loc, j, fit_pred_all[i][j])
        loc += 1

    out_pop2 = wb1.add_worksheet('pred_pop2_all')
    loc = 0
    for i in range(len(all_pop2)):
        out_pop2.write(loc, 0, f'{[i]}')
        for j in range(len(all_pop2[i])):
            loc += 1
            for z in range(len(all_pop2[i][j])):
                out_pop2.write(loc, z, all_pop2[i][j][z])
        loc += 1


    out_fit_truth_all = wb1.add_worksheet('fit_truth')
    loc = 0
    for i in range(len(fit_truth)):
        for j in range(len(fit_truth[i])):
            out_fit_truth_all.write(loc, j, fit_truth[i][j])
        loc += 1

    out_pop2_all = wb1.add_worksheet('pop2_all')
    loc = 0
    for i in range(len(pop2_all)):
        for j in range(len(pop2_all[i])):
            out_pop2_all.write(loc, j, pop2_all[i][j])
        loc += 1

    DNN_fit_all = wb1.add_worksheet('DNN_prediction_fitness')
    loc = 0
    for i in range(len(DNN_prediction_fitness)):
        for j in range(len(DNN_prediction_fitness[i])):
            DNN_fit_all.write(loc, j, DNN_prediction_fitness[i][j])
        loc += 1
    wb1.close()

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



def draw_fit_truth(data1):
    data = copy.deepcopy(data1)
    data.pop(16)
    fig2 = plt.figure(num=1, figsize=(23, 30))
    ax2 = fig2.add_subplot(111)
    ax2.tick_params(labelsize=40)
    ax2.set_xlabel("Iteration", fontsize=50)  # 添加x轴坐标标签，后面看来没必要会删除它，这里只是为了演示一下。
    ax2.set_ylabel('fitness', fontsize=50)  # 添加y轴标签，设置字体大小为16，这里也可以设字体样式与颜色
    ax2.spines['bottom'].set_linewidth(4);  ###设置底部坐标轴的粗细
    ax2.spines['left'].set_linewidth(4)
    ax2.spines['right'].set_color('none')
    ax2.spines['top'].set_color('none')
    # plt.ylim((150, 400))
    bbb = np.arange(0, len(data))

    ax2.plot(bbb, data, linewidth=6, color='r')

    ax2.set(xlim=(0, len(data)),
            xticks=np.arange(0, len(data), 10),
            )
    for i in range(7):
        x_te = []
        for j in range(10):
            x_te.append(20 * i - 1)
        x_te = np.array(x_te)
        y_te = np.linspace(0, 5, 10)
        ax2.plot(x_te, y_te, linewidth=1, color='black')
    plt.show()


def draw_gx_chayi(gx_truth,gx_pred):
    gx_truth_div = []
    gx_pred_div = []
    for i in range(len(gx_truth[0])):
        temp1=[]
        temp2=[]
        for j in range(len(gx_truth)):
            temp1.append(gx_truth[j][i])
            temp2.append(gx_pred[j][i])
        gx_truth_div.append(temp1)
        gx_pred_div.append(temp2)
    return gx_truth_div,gx_pred_div
def draw_gx_chayi2(gx_truth_div,gx_pred_div,time):
    fig2 = plt.figure(num=1, figsize=(23, 30))
    ax2 = fig2.add_subplot(111)
    ax2.tick_params(labelsize=40)
    ax2.set_xlabel("Iteration", fontsize=50)  # 添加x轴坐标标签，后面看来没必要会删除它，这里只是为了演示一下。
    ax2.set_ylabel('fitness', fontsize=50)  # 添加y轴标签，设置字体大小为16，这里也可以设字体样式与颜色
    ax2.spines['bottom'].set_linewidth(4);  ###设置底部坐标轴的粗细
    ax2.spines['left'].set_linewidth(4)
    ax2.spines['right'].set_color('none')
    ax2.spines['top'].set_color('none')
    # plt.ylim((150, 400))




    fig2 = plt.figure(num=1, figsize=(23, 30))
    ax2 = fig2.add_subplot(111)
    ax2.tick_params(labelsize=40)
    ax2.set_xlabel("Iteration", fontsize=50)  # 添加x轴坐标标签，后面看来没必要会删除它，这里只是为了演示一下。
    ax2.set_ylabel('fitness', fontsize=50)  # 添加y轴标签，设置字体大小为16，这里也可以设字体样式与颜色
    ax2.spines['bottom'].set_linewidth(4);  ###设置底部坐标轴的粗细
    ax2.spines['left'].set_linewidth(4)
    ax2.spines['right'].set_color('none')
    ax2.spines['top'].set_color('none')
    # plt.ylim((150, 400))


    bbb = np.arange(0, len(gx_pred_div[0]))
    ax2.plot(bbb, gx_pred_div[time], linewidth=6, color='r')
    ax2.plot(bbb, gx_truth_div[time], linewidth=6, color='blue')
    ax2.set(xlim=(0, len(gx_pred_div[0])),ylim=(0, 1),
            xticks=np.arange(0, len(gx_pred_div[0]), 10),yticks=np.arange(0, 1, 0.1),
                )
        # for i in range(7):
        #     x_te = []
        #     for j in range(10):
        #         x_te.append(20 * i - 1)
        #     x_te = np.array(x_te)
        #     y_te = np.linspace(0, 1.5, 10)
        #     ax2.plot(x_te, y_te, linewidth=1, color='black')
    plt.show()
    # plt.clf()

#查看gx记忆池中的分布范围
def gx_dietribute(gx_all):
    gx_num = []
    gx_all_memo =copy.deepcopy(gx_all)
    start_num = 0
    end_num = 1
    Jg = np.round(np.arange(start_num, end_num, 0.1),1).tolist()
    Jg.append(end_num)
    Jg_1 = []
    for i in range(len(Jg)-1):
        Jg_1.append([Jg[i],Jg[i+1]])

    for i in range(len(gx_all[0])):
        temp = [[] for nu in range(10)]
        for j in range(len(gx_all)):
            for z in range(len(Jg_1)):
                if gx_all[j][i]<=Jg_1[z][1] and gx_all[j][i]>=Jg_1[z][0]:
                    temp[z].append(gx_all[j][i])
        gx_num.append(temp)

    gx_eve_num = []
    for i in range(len(gx_num)):
        temp = []
        for j in range(len(gx_num[0])):
            temp.append(len(gx_num[i][j]))
        gx_eve_num.append(temp)
    return gx_num,gx_eve_num
#绘制统计gx个数分布柱状图
def gx_column(gx_num,time):

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['figure.figsize'] = (5, 3)
    data = gx_num[time]
    x= range(len(gx_num[time]))
    countries = ['0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5','0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1']
    plt.bar(countries, data)
    plt.tick_params(labelsize=30)
    plt.show()

modular_length = 8000
modular_width = [4000,4000,5400,3600,3600,4400,4400,4000]
modular_heigth = 3000
modular_length_num = 8
modular_dis = 400
corridor_width = 4000
story_num = 12
story_zone = 4#每组模块的分区数量
story_group = 3#每组模块的楼层数
modular_num = 3#整个建筑的模块种类

zone_num = int(story_num / story_group * story_zone)
section_num = 3 * modular_num
brace_num = modular_num
group_num = int(story_num / story_group)
modular_all = modular_length_num * 2 *story_num

pop_last = []
fit_last =[]

# steel section information
sections_data_c1, type_keys_c1, sections_c1 = ms.get_section_info(section_type='c0',
                                                                  cfg_file_name="Steel_section_data_I_cube.ini")

# generate model
model_data = dj.generate_model_data(modular_length,modular_width,modular_heigth,modular_length_num,modular_dis,story_num,corridor_width)
nodes = model_data[0]
edges_all = model_data[1]
labels = model_data[2]
cor_edges = model_data[3]
joint_hor = model_data[4]
joint_ver = model_data[5]
room_indx = model_data[6]
#优化参数
DNA_SIZE = 4*story_num+modular_length_num*2*story_num
POP_SIZE = 30
CROSSOVER_RATE = 0.6
MUTATION_RATE = 0.1
N_GENERATIONS = 140
num_thread = 5

min_genera = []

num_room_type=1




x = np.linspace(0, 11, 12)

sap_run_time00 =0
num_room_type=1
#全局记忆池
memorize_pool = []
memorize_fit = []
memorize_weight = []
memorize_col = []
memorize_beam = []
memorize_sum = []
memorize_gx = []
memorize_gx_nor = []
memorize_num = []
sap_run_time00 = 0

#新增
gx_all_truth = []

gx_pred_best=[]#每次预测得到的

#局部记忆池


all_fit_pred_GA = []
memorize_sum_local=[]
memorize_pool_local=[]
memorize_fit_local=[]
memorize_weight_local=[]
memorize_col_local=[]
memorize_beam_local=[]
memorize_gx_local=[]
history_loss = []
history_mae = []
DNN_prediction_fitness= []
POP_SIZE=30
num_var = 5
modular_num= 3
file_time = 15
num_continue = 140
labels = []
labels1 = []
for i in range(group_num):
    temp = []
    for j in range(story_zone):
        for z in range(int(modular_length_num/story_zone)):
            temp.append(i*story_zone+j)
    for j in range(2*story_group):
        labels.extend(temp)
        labels1.append(temp)

gx_data_select = [0,1,2,3,4,5]
#获得每隔N代的新种群finess，与真实fitness排序
# fit_pred,fit_truth,pop_min_pred = get_DNN_GA(file_time,27,20)
# sort_pred,sort_truth=fit_sort(fit_pred,fit_truth)
#静态训练以及sap计算对比
# pop2_best,memorize_pool,memorize_fit,memorize_weight,memorize_gx,gx_prediction,memorize_loss,memorize_mae,memorize_gx_nor,memorize_num=get_continue_data(file_time,num_continue)
# fitness_prediction2,fitness,DNN_prediction_fitness,gx_truth_all,gx_pred_all=get_pred_fit(pop2_best,10,400)
#在线训练神经网络生成最优个体
local_memorize_pop,local_memorize_gx,pop_best,gx_all_read=get_local_global_data(file_time)

for i in range(len(local_memorize_gx)):
    for j in range(len(local_memorize_gx[i])):
        temp =[]
        for z in gx_data_select:
            temp.append(local_memorize_gx[i][j][z])
        local_memorize_gx[i][j] = temp
for i in range(len(gx_all_read)):
    temp = []
    for j in gx_data_select:
        temp.append(gx_all_read[i][j])
    gx_all_read[i]=temp


all_pop2,fit_pred_all,DNN_prediction_fitness,gx_pred_best,all_fit_pred_GA,pop_last=run_DNN_GA(local_memorize_pop,local_memorize_gx,pop_best)

for i in range(len(gx_pred_best)):
    gx_pred_best[i] = gx_pred_best[i][0].tolist()
fit_truth = get_fitness(all_pop2)

sort_pred,sort_truth=fit_sort(fit_pred_all,fit_truth)
pop_pred_best,pop_truth_best,gx_truth_min,gx_min = draw_min_pop(all_pop2,fit_pred_all,fit_truth,memorize_gx_nor,memorize_gx)
output_data(pop_pred_best,fit_truth,fit_pred_all,all_pop2,DNN_prediction_fitness,6)

# #绘制gx差异值0
gx_truth_div,gx_pred_div=draw_gx_chayi(memorize_gx_nor,gx_pred_best)

draw_gx_chayi2(gx_truth_div,gx_pred_div,4)
# #统计gx分布并绘制
# gx_dis,gx_num=gx_dietribute(gx_all_read)
# gx_column(gx_num,0)
# #绘制所有最优个体fit曲线
# draw_fit_truth(memorize_fit)
#
# #绘制loss曲线
# all_loss_data = []
# for i in range(len(history_loss)):
#     all_loss_data.extend(history_loss[i])
# draw_loss(all_loss_data)

# fit_all_memorize = []
# for i in range(len(local_memorize_gx)):
#     temp= Gx_convert(np.array(local_memorize_gx[i]))
#     fit_all_memorize.append(temp)
#
# fit_xiaoyu = []
# for i in range(len(fit_all_memorize)):
#     temp = []
#     for j in range(len(fit_all_memorize[i])):
#         if fit_all_memorize[i][j]<700:
#             temp.append(fit_all_memorize[i][j])
#     fit_xiaoyu.append(temp)
