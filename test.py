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
for num_var in [4,8,10]:
    for num_room_type in [5,9]:
        value_str = []
        for z in range(50):
            wb = xlrd.open_workbook(filename=f'out_sap_run_{num_var}_{num_room_type}.xls', formatting_info=True)
            sheet1 = wb.sheet_by_index(6)
            rows = sheet1.row_values(3)[z]
            value_str.append(rows)
        all_value_str.append(value_str)

value_all_int = []
for i in range(len(all_value_str)):
    value_int = []
    for j in range(len(all_value_str[0])):
        value= int(all_value_str[i][j][0])*100+int(all_value_str[i][j][1])*10+int(all_value_str[i][j][2])
        value_int.append(value)
    value_all_int.append(value_int)

num1 = 0.8
num2 = 0.75
num3 = 3
num4 = 0
fig2 = plt.figure(num=1, figsize=(23, 30))
ax2 = fig2.add_subplot(111)
ax2.tick_params(labelsize=40)
ax2.set_xlabel("Iteration",fontsize=50)  # 添加x轴坐标标签，后面看来没必要会删除它，这里只是为了演示一下。
ax2.set_ylabel("Total weight", fontsize=50)  # 添加y轴标签，设置字体大小为16，这里也可以设字体样式与颜色
ax2.spines['bottom'].set_linewidth(4);###设置底部坐标轴的粗细
ax2.spines['left'].set_linewidth(4)
ax2.spines['right'].set_color('none')
ax2.spines['top'].set_color('none')
bbb = np.arange(0, 50)
label_var = [[4,5],[4,9],[8,5],[8,9],[10,5],[10,9]]
for i in range(6):
    ccc = value_all_int[i]
    ax2.plot(bbb, ccc, label = f"sec{label_var[i][0]}_ro{label_var[i][1]}",linewidth=6)
    ax2.legend(bbox_to_anchor=(num1, num2), loc=num3, borderaxespad=num4,  handlelength=1.5, fontsize=30, shadow=False)

    # ax2.set_xticks(np.linspace(1, 5, 5))
    # ax2.set_yticks(np.linspace(0, 18, 10))
    # ax2.set_xticklabels(["1th time", "2nd time", "3rd time", "4th time", "5th time"], fontproperties="SimHei", \
    #                    fontsize=12, rotation=10)
plt.show()

conf = configparser.ConfigParser()
list = []
for i in range(5):
    list1 = []
    for j in range(5):
        list1.append(randint(0,5))
    list.append(list1)



yangtingting = []
zhanjiqi = []
luyiwen = []
for i in range(50):
    yangtingting.append(i+1)
    zhanjiqi.append(m.e**(yangtingting[i]*0.25))
    luyiwen.append(zhanjiqi[i])



zhanhuang = 0
for i in range(len(luyiwen)):
    zhanhuang+=luyiwen[i]

zhangqingyang = []
for i in range(len(luyiwen)):
    zhangqingyang.append(luyiwen[i]/zhanhuang)
