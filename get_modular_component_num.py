import numpy as np
import pandas as pd
import openpyxl
from collections import Counter

modular_num = 3
num_var = 4
file_name = 1

story_num = 12
story_zone = 4#每组模块的分区数量
story_group = 3#每组模块的楼层数
modular_length_num = 8
zone_num = int(story_num / story_group * story_zone)
section_num = 3 * modular_num
brace_num = modular_num
group_num = int(story_num / story_group)
modular_all = modular_length_num * 2 *story_num
num_room_type = 1

all_mod = 192
all_sect = 4*4*3

wb = openpyxl.load_workbook(
    filename=f"D:\desktop\os\optimization of structure\out_all_infor_case4\\run_infor_{num_var}_{modular_num}_{file_name}.xlsx",
    )
# sheet1 = wb['pop1_all']
# for z in range(48):
#     rows = sheet1.cell(4311,z+1).value
#     pop_room.append(rows)
# sheet1 = wb['pop3_all']
# for z in range(modular_length_num*2*story_num):
#     rows = sheet1.cell(4311,z+1).value
#     pop_room_label.append(rows)

pop1 = []
sheet1 = wb['pop1_all']
for z in range(all_sect):
    rows = sheet1.cell(4311,z+1).value
    pop1.append(rows)

pop2 = []
sheet1 = wb['pop2_all']
for z in range(num_var+num_room_type+section_num+brace_num+zone_num):
    rows = sheet1.cell(4311,z+1).value
    pop2.append(rows)

pop2 = pop2[-16:]

pop1_counter = Counter(pop1)
pop3_counter = Counter(pop2)

k = sorted(pop1_counter.items(), key=lambda x: x[0])


for i in range(len(k)):
    k[i] = list(k[i])
    k[i][1] = k[i][1]*24

pop2_temp = sorted(pop3_counter.items(), key=lambda x: x[0])


for i in range(len(pop2_temp)):
    pop2_temp[i] = list(pop2_temp[i])
    pop2_temp[i][1] = pop2_temp[i][1]*6



print(k)
print(pop2_temp)