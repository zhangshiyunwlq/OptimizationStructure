import copy

import numpy as np
import random
from random import randint
import xlwt
import openpyxl
import configparser
import xlrd
import matplotlib.pyplot as plt
from matplotlib import font_manager
import math as m

story_num = 12
modular_length_num = 8
modular_num = 4
num_var = 6
num_room_type=1
DNA_SIZE = 4*story_num+modular_length_num*2*story_num
POP_SIZE = 50
story_zone = 4
story_group = 3
x = np.linspace(0, 13, 14)



zone_num = int(story_num / story_group * story_zone)
section_num = 3 * modular_num
brace_num = modular_num
group_num = int(story_num / story_group)
modular_all = modular_length_num * 2 *story_num

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

def generate_coding_modular_section(x):

    room_nu = np.linspace(1, 3, 3)

    pop = np.zeros((POP_SIZE,num_var+num_room_type+section_num+brace_num+zone_num))
    for i in range(len(pop)):
        sec = list(map(int, random.sample(x.tolist(), num_var)))
        sec.sort()
        for j in range(num_var):
            pop[i][j] = sec[j]

        for j in range(num_var,num_var+num_room_type):
            pop[i][j] = random.randint(1,3)

        for j in range(num_var+num_room_type,num_var+num_room_type+section_num):
            pop[i][j] = random.randint(0,num_var-1)

        for j in range(num_var+num_room_type+section_num,num_var+num_room_type+section_num+brace_num):
            pop[i][j] = randint(0,1)
        for j in range(num_var+num_room_type+section_num+brace_num,num_var+num_room_type+section_num+brace_num+zone_num):
            pop[i][j] = randint(0,modular_num-1)

    return pop

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



pop2= generate_coding_modular_section(x)
pop1,pop3 = decoding_modular_section(pop2)

ytt = pop1[0]
zjq = pop3[0]
section_all_modular = []
brace_all_modular = []

modular_type1 = [i for i in range(3)]

zone_type_all = [] #所有区域的编号索引
for i in range(zone_num):
    modular_type_temp = []
    for j in range(len(modular_type1)):
        modular_type_temp.append(modular_type1[j] + 3 * i)
    zone_type_all.append(modular_type_temp)

section_all = []
for i in range(len(labels)):
    for j in range(3):
        temp1 = zone_type_all[int(labels[i])][j]
        section_all.append(ytt[temp1])

brace_all = []
for i in range(len(labels)):
        temp1 = int(labels[i])
        brace_all.append(zjq[temp1])
