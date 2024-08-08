import copy
import time
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
story_zone = 1
story_group = 2
x = np.linspace(0, 11, 12)



zone_num = int(story_num / story_group * story_zone)
section_code = 3 * zone_num
brace_code = zone_num
group_num = int(story_num / story_group)
modular_all = modular_length_num * 2 *story_num

labels = []
for i in range(int(story_num/2)):
    for j in range(modular_length_num*4):
        labels.append(i)

start = time.time()
for i in range(10):
    time.sleep(1)
    if i%3==0:
        end = time.time()
        print(f'{end-start}')