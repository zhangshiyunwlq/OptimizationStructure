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
        sheet1 = wb.sheet_by_index(5)
        for z in range(name1[i][0]):
            rows = sheet1.row_values(1)[z]
            value_str.append(rows)
        all_value_str.append(value_str)
    return all_value_str

def draw_picture(info2,name,title_name):
    num1 = 0.8
    num2 = 0.75
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

data_info = [[50,3,1],[50,5,0],[50,6,0],[50,8,0],[50,9,0]]
infor_all = get_info(data_info)
infor_name = ['3','5','6','8','9']
title_name = 'Fitness'

draw_picture(infor_all,infor_name,title_name)
draw_plot_picture(infor_all,data_info)
