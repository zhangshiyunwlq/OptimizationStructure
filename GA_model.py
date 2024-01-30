import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from importlib import reload
import configparser
import copy
from matplotlib import cm
import modular_utils as md
import model_to_sap as ms
from mpl_toolkits.mplot3d import Axes3D
import xlwt
import math as m
import comtypes.client
import time
import model_to_sap as ms
from matplotlib import font_manager
import sap_run as sr
import data_to_json as dj
import  main2 as mm
def Fun(weight,g_col,g_beam,dis_all,all_force,u):
    g_col_all = 0
    g_beam_all = 0
    Y_dis_radio_all = 0
    Y_interdis_all = 0
    Y_interdis_radio_all = 0
    floor_radio = 0
    for i in range(len(g_col)):
        if g_col[i]<= 0:
            g_col[i] = 0
        else:
            g_col[i] = g_col[i]
        g_col_all += g_col[i]
    for i in range(len(g_beam)):
        if g_beam[i]<= 0:
            g_beam[i] = 0
        else:
            g_beam[i] = g_beam[i]
        g_beam_all += g_beam[i]
    #y dis ratio
    for i in range(len(dis_all[5])):
        if dis_all[5][i] <= 1.5 and dis_all[5][i] >= -1.5:
            dis_all[5][i] = 0
        else:
            dis_all[5][i] = dis_all[5][i]
        Y_dis_radio_all += dis_all[5][i]
    # y interdis max
    for i in range(len(dis_all[7])):
        if dis_all[7][i] <= 0.004 and dis_all[7][i] >= -0.004:
            dis_all[7][i] = 0
        else:
            dis_all[7][i] = dis_all[7][i]
        Y_interdis_all += dis_all[7][i]
    # y interdis radio
    for i in range(len(dis_all[11])):
        if dis_all[11][i] <= 1.5 and dis_all[11][i] >= -1.5:
            dis_all[11][i] = 0
        else:
            dis_all[11][i] = dis_all[11][i]
        Y_interdis_radio_all += dis_all[11][i]
    # x interdis ratio
    for i in range(len(all_force[10])):
        if all_force[10][i] <= 1.5 and all_force[10][i] >= -1.5:
            all_force[10][i] = 0
        else:
            all_force[10][i] = all_force[10][i]
        floor_radio += all_force[10][i]
    result = weight + u*(g_col_all + g_beam_all + Y_dis_radio_all + Y_interdis_all + Y_interdis_radio_all+floor_radio)
    return result,weight

def Get_fitness(result):
    fitness1 = []
    fitness2 = []
    for i in range(len(result)):
        if result[i]>=800:
            fitness1.append((result[i]/100000)+800)
        else :
            fitness1.append(result[i])
        if fitness1[i]>=1100:
            fitness2.append(1)
        else:
            fitness2.append(1100-fitness1[i])
    # fitness1.append(m.exp(-result[i]))
    # fitness2.append((m.exp(-result[i])+0.001))
    return fitness1, fitness2

def select(pop, fitness):  # nature selection wrt pop's fitness
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=(fitness) / (sum(fitness)))
    pop2 = np.zeros((POP_SIZE, DNA_SIZE))
    for i in range(len(pop2)):
        pop2[i] = pop[int(idx[i])]
    return pop2

def crossover_and_mutation(pop, CROSSOVER_RATE=0.8):
    new_pop = np.zeros((POP_SIZE,DNA_SIZE))
    for i in range(len(pop)):
        father = pop[i]
        child = father
        if np.random.rand() < CROSSOVER_RATE:
            mother = pop[np.random.randint(POP_SIZE)]
            cross_points1 = np.random.randint(low=0, high=DNA_SIZE)
            cross_points2 = np.random.randint(low=0, high=DNA_SIZE)
            while cross_points2==cross_points1:
                cross_points2 = np.random.randint(low=0, high=DNA_SIZE)
            exchan = []
            exchan.append(cross_points2)
            exchan.append(cross_points1)
            for j in range(min(exchan),max(exchan)):
                child[j] = mother[j]
        mutation(child)
        new_pop[i] = child
    return new_pop

def mutation(child, MUTATION_RATE=0.3):
    for mutate_point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:  # 以MUTATION_RATE的概率进行变异
            # mutate_point = np.random.randint(0, DNA_SIZE)  # 随机产生一个实数，代表要变异基因的位置
            # if mutate_point == 2:
            #     child[mutate_point] = np.random.choice(z)
            # else:
            child[mutate_point] = np.random.choice(x)

def Sap_analy_allroom(pop_room):

    sections_data_c1, type_keys_c1, sections_c1 = ms.get_section_info(section_type='c0',
                                                                      cfg_file_name="Steel_section_data33.ini")
    modular_building = md.ModularBuilding(nodes, room_indx, edges_all, labels, joint_hor, joint_ver, cor_edges)
    modulars_of_building = modular_building.building_modulars
    modular_nums = len(labels)
    modular_infos = {}
    APIPath = os.path.join(os.getcwd(), 'cases')
    mySapObject, ModelPath, SapModel = ms.SAPanalysis_GA_run(APIPath)
    sr.run_column_room(labels, modular_length_num * 2 * story_num, sections_data_c1, modular_infos, pop_room)
    for i in range(len(modulars_of_building)):
        modulars_of_building[i].Add_Info_And_Update_Modular(
            modular_infos[modulars_of_building[i].modular_label - 1])
    modular_building.Building_Assembling(modulars_of_building)

    all_data = ms.Run_GA_sap(mySapObject, ModelPath, SapModel, modular_building,200,modular_length_num,story_num)
    aa, bb, cc, dd, ee, ff, gg, hh, ii = all_data

    ret = SapModel.SetModelIsLocked(False)
    ret = mySapObject.ApplicationExit(False)
    SapModel = None
    mySapObject = None
    return aa,bb,cc,dd,ee,ff,gg,hh,ii

def GA(pop1):
    result = []
    num1 = 0
    num2 = 0
    num3 = 0
    weight_1 = []
    col_up = []
    beam_up = []
    # if len(pop_all) ==0:
    #     for time in range(len(pop1)):
    #         pop = pop1[time]
    #         pop_all.append(pop)
    #         we,co,be,r1,r2,r3,r4 =Sap_analy(pop)
    #         res1,res2 = Fun(we, co, be, 10000)
    #         num3 += 1
    #         weight_1.append(res2)
    #         result.append(res1)
    #         pop_fun_all.append(res1)
    #         pop_weight_all.append(res2)
    # else:
    #     for time in range(len(pop1)):
    #         pop = pop1[time]
    #         for i in range(len(pop_all)):
    #             if all(pop == pop_all[i]):
    #                 num1 = 1
    #                 num2 = i
    #                 # result.append(pop_fun_all[i])
    #         if num1 == 0:
    #             pop_all.append(pop)
    #             we, co, be, r1, r2, r3, r4 = Sap_analy(pop)
    #             res1, res2 = Fun(we, co, be, 10000)
    #             num3 += 1
    #             weight_1.append(res2)
    #             pop_weight_all.append(res2)
    #             result.append(res1)
    #             pop_fun_all.append(res1)
    #         elif num1 == 1:
    #             result.append(pop_fun_all[num2])
    #             weight_1.append(pop_weight_all[num2])
    for time in range(len(pop1)):
        pop = pop1[time]
        # pop_all.append(pop)
        we,co,be,r1,r2,r3,r4,dis_all,force_all =Sap_analy_allroom(pop)
        res1,res2 = Fun(we, co, be,dis_all,force_all, 10000)
        # num3 += 1
        weight_1.append(res2)
        col_up.append(co)
        beam_up.append(be)
        result.append(res1)
        # pop_fun_all.append(res1)
        # pop_weight_all.append(res2)
    return result,weight_1,col_up,beam_up

def Run_GA_allstory(POP_SIZE_1,DNA_SIZE_1,CROSSOVER_RATE_1,MUTATION_RATE_1,N_GENERATIONS_1,xx1,mySapObject):

    # run GA
    start = time.perf_counter()
    POP_SIZE = POP_SIZE_1
    DNA_SIZE = DNA_SIZE_1
    x = xx1
    CROSSOVER_RATE = CROSSOVER_RATE_1
    MUTATION_RATE = MUTATION_RATE_1
    N_GENERATIONS = N_GENERATIONS_1
    max_ru = []  # 记录每代fitness 最值
    pop_all = []  # 记录所有计算过的种群（不重复）
    pop_fun_all = []  # 记录所有计算过的种群fitness（不重复）
    pop_zhongqun_all = []  # 记录每代种群（不重复）
    pop_weight_all = []  # 记录所有计算过的种群重量（不重复）
    sap_run_time = 0  # 记录sap运行次数
    weight_min = []  # 记录每代最小重量
    pop_all_fitness = []  # 记录每代种群fitness
    pop_all_weight = []  # 记录每代种群重量
    col_up_all = []  # 记录每代每层柱最大约束
    beam_up_all = []  # 记录每代每层梁最大约束
    pop1 = np.zeros((POP_SIZE, DNA_SIZE))
    for i in range(len(pop1)):
        for j in range(len(pop1[0])):
            pop1[i][j] = np.random.choice(x)
            # pop1[i][j] = 15
    for _ in range(N_GENERATIONS):
        pop_zhongqun_all.append(pop1)
        result1, weight_pop, clo_up_1, beam_up_1 = GA(pop1)
        col_up_all.append(clo_up_1)
        beam_up_all.append(beam_up_1)
        pop_all_weight.append(weight_pop)
        # sap_run_time += run_time
        fitness1, fitness2 = Get_fitness(result1)
        pop_all_fitness.append(fitness2)
        mm = fitness2.index(max(fitness2))
        weight_min.append(weight_pop[mm])
        max1 = max(fitness2)
        mm2 = pop1[mm]
        max_ru.append(1100 - max(fitness2))
        pop1 = select(pop1, fitness2)
        pop1 = np.array(crossover_and_mutation(pop1, CROSSOVER_RATE))
        aaa = []
        aaa.append(pop1[0])
        pop200 = pop_all
        if max1 >= Get_fitness(GA(aaa)[0])[1][0]:
            sap_run_time += 1
            pop1[0] = mm2

    print(f"最小值位置为", pop1[0])
    print(f"最小值为", max_ru[len(max_ru) - 1])
    end = time.perf_counter()
    runTime = end - start
    print("运行时间：", runTime)


    all_infor = [pop1[0],max_ru[len(max_ru) - 1],max_ru,pop_all,pop_zhongqun_all,weight_min,pop_all_fitness,pop_all_weight,
                 col_up_all,beam_up_all]
    return all_infor

modular_length = mm.modular_length
modular_width = mm.modular_width
modular_heigth = mm.modular_heigth
modular_length_num = mm.modular_length_num
modular_dis = mm.modular_dis
story_num = mm.story_num
corridor_width = mm.corridor_width
# model_data = dj.generate_model_data(modular_length,modular_width,modular_heigth,modular_length_num,modular_dis,story_num,corridor_width)
nodes = mm.nodes
edges_all = mm.edges_all
labels = mm.labels
cor_edges = mm.cor_edges
joint_hor = mm.joint_hor
joint_ver = mm.joint_ver
room_indx = mm.room_indx
#
# APIPath = os.path.join(os.getcwd(), 'cases')
# mySapObject, ModelPath, SapModel = ms.SAPanalysis_GA_run(APIPath)
#
#
# ''' 按楼层(1层)+房间进行优化 '''
POP_SIZE = mm.POP_SIZE
DNA_SIZE = mm.DNA_SIZE
CROSSOVER_RATE = mm.CROSSOVER_RATE
MUTATION_RATE = mm.MUTATION_RATE
N_GENERATIONS = mm.N_GENERATIONS
x = mm.x
#
# all_GA_infor = Run_GA_allstory(POP_SIZE,DNA_SIZE,CROSSOVER_RATE,MUTATION_RATE,N_GENERATIONS,x,mySapObject)
#
# weight_fin = all_GA_infor[0]
# pop_fin = all_GA_infor[1]
# max_ru = all_GA_infor[2]
# pop_all= all_GA_infor[3]
# pop_zhongqun_all= all_GA_infor[4]
# weight_min= all_GA_infor[5]
# pop_all_fitness= all_GA_infor[6]
# pop_all_weight= all_GA_infor[7]
# col_up_all= all_GA_infor[8]
# beam_up_all= all_GA_infor[9]
#
# #查看根据room分类最优模型
# pop_room = weight_fin
# # for i in range(18):
# #     pop2[i] = 14
# pop_room = np.array(pop_room)
# APIPath = os.path.join(os.getcwd(), 'cases')
# mySapObject, ModelPath, SapModel = ms.SAPanalysis_GA_run(APIPath)
# # ret = SapModel.File.Save("D:\图片文件夹\结构分析模型.sdb")
# weight1,g_col,g_beam,reaction_all,section_all,all_up_name,all_up_data,Joint_dis,all_force = Sap_analy_allroom(pop_room)
