import openpyxl
import pandas as pd
import xlwt
import xlrd
import copy
import numpy as np
import math as m
import configparser
import time
import random
from random import randint
import os
import sys
import xlsxwriter
def out_put_result(pop1_all,pop2_all,pop3_all,fitness_all,weight_all,pop_all_fitness,pop_all_weight,time):
    APIPath = os.path.join(os.getcwd(), 'out_all_infor_case4')
    SpecifyPath = True
    if not os.path.exists(APIPath):
        try:
            os.makedirs(APIPath)
        except OSError:
            pass

    path1 = os.path.join(APIPath, f'run_infor_{num_var}_{modular_num}_{1001}')


    wb1 = xlsxwriter.Workbook(f'{path1}.xlsx')
    out_pop1_all = wb1.add_worksheet('pop1_all')

    loc = 0
    for i in range(len(pop1_all)):
        out_pop1_all.write(loc, 0, f'{[i]}')
        for j in range(len(pop1_all[i])):
            loc += 1
            for z in range(len(pop1_all[i][j])):
                out_pop1_all.write(loc, z, pop1_all[i][j][z])
        loc += 1

    out_pop2_all = wb1.add_worksheet('pop2_all')
    loc = 0
    for i in range(len(pop2_all)):
        out_pop2_all.write(loc, 0, f'{[i]}')
        for j in range(len(pop2_all[i])):
            loc += 1
            for z in range(len(pop2_all[i][j])):
                out_pop2_all.write(loc, z, pop2_all[i][j][z])
        loc += 1

    out_pop3_all = wb1.add_worksheet('pop3_all')
    loc = 0
    for i in range(len(pop3_all)):
        out_pop3_all.write(loc, 0, f'{[i]}')
        for j in range(len(pop3_all[i])):
            loc += 1
            for z in range(len(pop3_all[i][j])):
                out_pop3_all.write(loc, z, pop3_all[i][j][z])
        loc += 1

    pop_all_fit = wb1.add_worksheet('pop_all_fitness')
    loc = 0
    for i in range(len(pop_all_fitness)):
        pop_all_fit.write(loc, 0, f'{[i]}')
        loc += 1
        for j in range(len(pop_all_fitness[i])):
            pop_all_fit.write(loc, j, pop_all_fitness[i][j])
        loc += 1


    pop_all_wei = wb1.add_worksheet('pop_all_weight')
    loc = 0
    for i in range(len(pop_all_weight)):
        pop_all_wei.write(loc, 0, f'{[i]}')
        loc += 1
        for j in range(len(pop_all_weight[0])):
            pop_all_wei.write(loc, j, pop_all_weight[i][j])
        loc += 1


    outmaxfitness = wb1.add_worksheet('max_fitness')
    loc = 0
    outmaxfitness.write(loc, 0, 'max_fitness_all')
    loc += 1
    for i in range(len(fitness_all)):
        outmaxfitness.write(loc, i, fitness_all[i])
    loc += 1
    outmaxfitness.write(loc, 0, 'min_weight_all')
    loc += 1
    for i in range(len(weight_all)):
        outmaxfitness.write(loc, i, weight_all[i])




    wb1.close()

def remove_pop(pop2_all_read):
    pop2_remove = []
    fitness_remove = []
    for i in range(len(pop2_all_read)):
        if i <= len(pop2_all_read):
            if type(pop2_all_read[i][0]) == str:
                pop2_remove.append(i)

    for i in range(len(pop2_remove)):
        pop2_all_read.remove(pop2_all_read[int(pop2_remove[len(pop2_remove) - 1 - i])])



    # 将原始列表分成 9 个列表，每个列表包含 10 个子列表
    chunk_size = 30
    chunked_list = [pop2_all_read[i:i + chunk_size] for i in range(0, len(pop2_all_read), chunk_size)]

    return chunked_list

def remove_fit(all_data):

    # 保留偶数位置的子列表（索引从 0 开始，所以索引为 1, 3, 5... 的位置是偶数位置）
    filtered_list = [sublist for index, sublist in enumerate(all_data) if index % 2 != 0]
    return filtered_list

modular_length_num = 8

story_num = 12
story_zone = 4#每组模块的分区数量
story_group = 3#每组模块的楼层数
modular_num = 4#整个建筑的模块种类

zone_num = int(story_num / story_group * story_zone)
section_num = 3 * modular_num
brace_num = modular_num
group_num = int(story_num / story_group)
modular_all = modular_length_num * 2 *story_num

'''model visvalization'''

'''GA to model'''
# case 1 按照区域分组优化
POP_SIZE = 30
DNA_SIZE = 2*story_num*3
CROSSOVER_RATE = 0.4
MUTATION_RATE = 0.15
N_GENERATIONS = 100
num_thread = 10
min_genera = []
x = np.linspace(0, 12, 13)

num_var= 4
num_room_type=1
file_time= 101


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


path_infor = f"D:\desktop\os\optimization of structure\out_all_infor_case4\\run_infor_{num_var}_{modular_num}_{file_time}.xlsx"
pop1_best1 = pd.read_excel(io=path_infor, sheet_name="pop1_all",header=None)
pop1_all_read = pop1_best1.values.tolist()
pop2_best1 = pd.read_excel(io=path_infor, sheet_name="pop2_all",header=None)
pop2_all_read = pop2_best1.values.tolist()
pop3_best1 = pd.read_excel(io=path_infor, sheet_name="pop3_all",header=None)
pop3_all_read = pop3_best1.values.tolist()
pop_fit_best = pd.read_excel(io=path_infor, sheet_name="pop_all_fitness",header=None)
pop_fit_read = pop_fit_best.values.tolist()
pop_weight_best = pd.read_excel(io=path_infor, sheet_name="pop_all_weight",header=None)
pop_weight_read = pop_weight_best.values.tolist()
pop_max_best = pd.read_excel(io=path_infor, sheet_name="max_fitness",header=None)
pop2_max_read = pop_max_best.values.tolist()

pop1_all = remove_pop(pop1_all_read)
pop2_all = remove_pop(pop2_all_read)
pop3_all = remove_pop(pop3_all_read)
pop_all_fitness=remove_fit(pop_fit_read)
max_fitness=remove_fit(pop2_max_read)
pop_all_weight=remove_fit(pop_weight_read)

out_put_result(pop1_all,pop2_all,pop3_all,max_fitness[0],max_fitness[1],pop_all_fitness,pop_all_weight,time)