import xlrd
import matplotlib.pyplot as plt
import copy
import numpy as np

def get_info(name1):
    all_value_str = []
    for i in range(len(name1)):
        value_str = []
        wb = xlrd.open_workbook(
            filename=f'D:\desktop\os\optimization of structure\optimization of structure\optimization of structure\out_all_infor\\run_infor_{name1[i][1]}_{name1[i][2]}.xls',
            formatting_info=True)
        # wb = xlrd.open_workbook(
        #     filename=f'D:\desktop\os\optimization of structure\optimization of structure\optimization of structure\分析数据\不限定范围\\run_infor_{name1[i][1]}_{name1[i][2]}.xls',
        #     formatting_info=True)
        # wb = xlrd.open_workbook(
        #     filename=f'E:\C盘默认文件\\run\OptimizationStructure\分析数据\限定范围\\run_infor_{name1[i][1]}_{name1[i][2]}.xls',
        #     formatting_info=True)
        sheet1 = wb.sheet_by_index(5)
        for z in range(name1[i][0]):
            rows = sheet1.row_values(1)[z]
            value_str.append(rows)
        all_value_str.append(value_str)
    return all_value_str

def draw_picture(info2,name,title_name):
    num1 = 0.8
    num2 = 0.60
    num3 = 3
    num4 = 0
    fig2 = plt.figure(num=1, figsize=(23, 30))
    ax2 = fig2.add_subplot(111)
    ax2.tick_params(labelsize=40)
    ax2.set_xlabel("Iteration",fontsize=50)  # 添加x轴坐标标签，后面看来没必要会删除它，这里只是为了演示一下。
    ax2.set_ylabel(title_name, fontsize=50)  # 添加y轴标签，设置字体大小为16，这里也可以设字体样式与颜色
    ax2.spines['bottom'].set_linewidth(4);###设置底部坐标轴的粗细
    ax2.spines['left'].set_linewidth(4)
    ax2.spines['right'].set_color('none')
    ax2.spines['top'].set_color('none')
    plt.ylim((150, 400))
    info = copy.deepcopy(info2)
    for i in range(len(info)):
        for j in range(len(info[i])):
            if info[i][j]>=500:
            # info[i][j] = 500+100*(m.log(info[i][j]))
                info[i][j] = 500 + info[i][j]/1000

    for i in range(len(info)):
        bbb = np.arange(0, len(info[i]))
        ccc = info[i]
        ax2.plot(bbb, ccc, label = name[i],linewidth=6)
        ax2.legend(bbox_to_anchor=(num1, num2), loc=num3, borderaxespad=num4,  handlelength=1.5, fontsize=30, shadow=False)

    plt.show()

def draw_plot_picture(info2,data_infor):
    num_var =[]
    fit = []
    for i in range(len(data_infor)):
        num_var.append(data_infor[i][1])
        fit.append(info2[i][len(info2[i])-1])

    plt.figure(figsize=(10, 10), dpi=100)
    plt.xticks(range(0, 12, 1))
    # plt.yticks(range(100, 500, 50))
    plt.ylim((150, 400))
    plt.scatter(num_var, fit, s=13, label='fitness')
    plt.xlabel("The number of variable", fontdict={'size': 16})
    plt.ylabel("Fitness", fontdict={'size': 16})
    # plt.title("历年天猫双11总成交额", fontdict={'size': 20})
    plt.show()

def static_braced(name1):
    braced_st = []
    for i in range(len(name1)):
        pop_room = []
        pop_room_label = []
        wb = xlrd.open_workbook(
            filename=f'D:\desktop\os\optimization of structure\optimization of structure\optimization of structure\out_all_infor\\run_infor_{name1[i][1]}_{name1[i][2]}.xls',
            formatting_info=True)
        sheet1 = wb.sheet_by_index(2)
        for z in range(6):
            rows = sheet1.row_values(5050)[z*16]
            pop_room.append(rows)
        braced_st.append(pop_room)
    return  braced_st
# data_info = [[100,2,0],[100,3,0],[100,3,1],[100,4,0],[100,4,1],[100,5,0],[100,5,1],[100,6,0],[100,6,1],[100,6,2],[100,6,3],[100,6,4],[100,7,0],[100,8,0],[100,8,1],[100,8,2],[100,9,0],[100,9,1],[100,9,2],[100,10,0],[100,10,1]]
# infor_all = get_info(data_info)
# infor_name = ['2_0','3_0','3_1','4_0','4_1','5_0','5_1','6_0','6_1','6_2','6_3','6_4','7_0','8_0','8_1','8_2','9_0','9_1','9_2','10_0','10_1']
# title_name = 'Fitness'

data_info = [[100,5,0],[100,5,1],[100,6,0],[100,6,1],[100,6,2],[100,6,3],[100,6,4],[100,7,0]]
infor_all = get_info(data_info)
infor_name = ['5_0','5_1','6_0','6_1','6_2','6_3','6_4','7_0']
title_name = 'Fitness'


draw_picture(infor_all,infor_name,title_name)
# draw_plot_picture(infor_all,data_info)
# static_braced(data_info)