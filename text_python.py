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


story_num =6
POP_SIZE =30
DNA_SIZE = story_num*3
CROSSOVER_RATE = 0.6
MUTATION_RATE = 0.1
N_GENERATIONS = 30
num_thread =10
min_genera = []

num_room_type = 1

modular_length_num = 8

modular_num = 4

zone_num = 6
section_num = 3 * modular_num
brace_num = modular_num
modular_all = modular_length_num * 2 *story_num
num_var=4
x = np.linspace(0, 11, 12)

labels = []
for i in range(0,story_num):
    for j in range(modular_length_num*2):
        labels.append(i)

xingnuzhanjiaqi = generate_coding_modular_section(x)
xingnuyangtingting,xingnujiangjiaqi = decoding_modular_section(xingnuzhanjiaqi)
