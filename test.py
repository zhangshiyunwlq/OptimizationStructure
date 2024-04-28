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
all_value_str = []
story_num = 6
modular_length_num = 8
modular_num = 6
POP_SIZE = 4*story_num+modular_length_num*2*story_num
POP_SIZE = 50
def generate_coding_modular():
    section_num = 3*modular_num
    brace_num = modular_num

    pop = np.zeros((POP_SIZE,section_num+brace_num+modular_length_num*2*story_num))
    for i in range(len(pop)):
        for j in range(section_num):
            pop[i][j] = random.randint(0,13)
        for j in range(section_num,section_num+brace_num):
            pop[i][j] = randint(1,3)
        for j in range(section_num+brace_num,section_num+brace_num+modular_length_num*2*story_num):
            pop[i][j] = randint(0,modular_num-1)

    return pop

def decoding_modular(pop2):
    section_num = 3*modular_num
    brace_num = modular_num
    pop_all = copy.deepcopy(pop2)
    modular_type1 = [i for i in range(3)]
    #生成对每个模块的截面编号索引
    modular_type_all= []
    for i in range(modular_num):
        modular_type_temp = []
        for j in range(len(modular_type1)):
            modular_type_temp.append(modular_type1[j]+3*i)
        modular_type_all.append(modular_type_temp)
    #生成截面表
    pop1_all = []
    for i in range(len(pop_all)):
        pop1_section = []
        for j in range(section_num+brace_num,section_num+brace_num+modular_length_num*2*story_num):
            for z in range(3):
                sec = int(pop_all[i][j])
                pop1_section.append(pop_all[i][int(modular_type_all[sec][z])])
        pop1_all.append(pop1_section)
    #生成支撑表
    brace_sort = [i for i in range(section_num,section_num+brace_num)]
    pop3_all = []
    for i in range(len(pop_all)):
        pop3_brace = []
        for j in range(section_num+brace_num,section_num+brace_num+modular_length_num*2*story_num):
            bra = int(pop_all[i][j])
            pop3_brace.append(pop_all[i][int(brace_sort[bra])])
        pop3_all.append(pop3_brace)
    return pop1_all,pop3_all


pop2= generate_coding_modular()
pop1,pop3 = decoding_modular(pop2)

