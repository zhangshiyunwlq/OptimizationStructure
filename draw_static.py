import xlrd
import matplotlib.pyplot as plt
import copy
import numpy as np
import openpyxl
def get_info(name1):
    all_value_str = []
    for i in range(len(name1)):
        value_str = []
        wb = openpyxl.load_workbook(
            filename=f'D:\desktop\os\optimization of structure\optimization of structure\optimization of structure\out_all_infor_case4\\run_infor_{name1[i][1]}_{modular_num}_{name1[i][2]}.xlsx'
            )
        # wb = xlrd.open_workbook(
        #     filename=f'D:\desktop\os\optimization of structure\optimization of structure\optimization of structure\分析数据\不限定范围\\run_infor_{name1[i][1]}_{name1[i][2]}.xls',
        #     formatting_info=True)
        # wb = xlrd.open_workbook(
        #     filename=f'E:\C盘默认文件\\run\OptimizationStructure\分析数据\限定范围\\run_infor_{name1[i][1]}_{name1[i][2]}.xls',
        #     formatting_info=True)
        sheet1 = wb['max_fitness']
        for z in range(name1[i][0]):
            rows = sheet1.cell(2,z+1).value
            value_str.append(rows)
        all_value_str.append(value_str)

        # value_str = []
        # wb =  xlrd.open_workbook(
        #     filename=f'D:\desktop\os\optimization of structure\optimization of structure\optimization of structure\out_all_infor\\run_infor_{name1[i][1]}_{name1[i][2]}.xlsx'
        #     formatting_info=True)
        # # wb = xlrd.open_workbook(
        # #     filename=f'D:\desktop\os\optimization of structure\optimization of structure\optimization of structure\分析数据\不限定范围\\run_infor_{name1[i][1]}_{name1[i][2]}.xls',
        # #     formatting_info=True)
        # sheet1 = wb.sheet_by_index(5)
        # for z in range(name1[i][0]):
        #     rows = sheet1.row_values(1)[z]
        #     value_str.append(rows)
        # all_value_str.append(value_str)

    return all_value_str

def draw_picture(info2,name,title_name):
    num1 = 0.8
    num2 = 0.8
    num3 = 3
    num4 = 0
    fig2 = plt.figure(num=1, figsize=(23, 30))
    ax2 = fig2.add_subplot(111)
    ax2.tick_params(labelsize=40)
    ax2.set_xlabel("Iteration",fontsize=50)  # 添加x轴坐标标签，后面看来没必要会删除它，这里只是为了演示一下。
    ax2.set_ylabel(title_name, fontsize=50)  # 添加y轴标签，设置字体大小为16，这里也可以设字体样式与颜色
    ax2.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax2.spines['left'].set_linewidth(2)
    ax2.spines['right'].set_color('none')
    ax2.spines['top'].set_color('none')
    plt.ylim((150, 400))
    info = copy.deepcopy(info2)

    for i in range(len(info)):
        if name[i] =='200*200':
            co = 'r'
        elif name[i] =='9-6':
            co = 'black'
        elif name[i] == '100*100':
            co = 'blue'

        bbb = np.arange(0, len(info[i]))
        ccc = info[i]
        ax2.plot(bbb, ccc, linewidth=6, color=co)
        # legend_handles = [plt.Line2D([0], [0], color='red', lw=2,linewidth=20),
        #                   plt.Line2D([0], [0], color='black', lw=2,linewidth=20),
        #                   plt.Line2D([0], [0], color='blue', lw=2, linewidth=20)
        #                   ]
        # legend_labels = ['200*200', '50*50', '100*100']
        # plt.legend(legend_handles, legend_labels,fontsize=20)

    ax2.set(xlim=(0, 150), ylim=(0, 1000),
           xticks=np.arange(0, 150, 20),
           yticks=np.arange(0, 1000, 100))
    plt.show()

def draw_picture0(info2,name,title_name):
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
    # plt.ylim((150, 400))
    # plt.xlim((0, len(info2[0])))
    ax2.set(xlim=(0, len(info2[1])), ylim=(0, 6000),
           xticks=np.arange(0, len(info2[1]), 20),
           yticks=np.arange(0, 6000, 100))
    info = copy.deepcopy(info2)
    for i in range(len(info)):
        for j in range(len(info[i])):
            if info[i][j]>=500:
            # info[i][j] = 500+100*(m.log(info[i][j]))
                info[i][j] = 500 + info[i][j]/1000

    for i in range(len(info)):
        bbb = np.arange(0, len(info[i]))
        ccc = info[i]
        ax2.plot(bbb, ccc,label=f'{name[i]}', linewidth=6)
        ax2.legend(bbox_to_anchor=(num1, num2), loc=num3, borderaxespad=num4,  handlelength=1.5, fontsize=30, shadow=False)


    plt.show()

def draw_plot_picture(info2,data_infor):
    num_var =[]
    fit = []
    for i in range(len(data_infor)):
        num_var.append(data_infor[i][1])
        fit.append(info2[i][len(info2[i])-1])

    plt.figure(figsize=(10, 10), dpi=100)
    plt.xticks(range(0, 14, 1))
    # plt.yticks(range(100, 500, 50))
    plt.ylim((0, 700))
    plt.xlim((0, 14))
    z = [433.72,446.06,443.39,436.44]
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
            filename=f'D:\desktop\os\optimization of structure\optimization of structure\optimization of structure\out_all_infor\\run_infor_{name1[i][1]}_{modular_num}_{name1[i][2]}.xls',
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

# data_info = [[100,14,18],[100,14,19],[100,14,20],[100,14,21],[100,14,22],[100,14,25],[100,14,30],[100,14,31],[100,14,33],[100,14,36]]
# data_info = [[150,14,44],[150,14,45],[150,14,46],[150,14,47],[150,14,48],[150,14,49],[150,14,471],[150,14,491],[200,14,60],[200,14,65],[200,14,67],[200,14,68]]
#
# infor_name = ['HIGA1','HIGA1','HIGA1','HIGA1','HIGA1','HIGA1','HIGA1','HIGA1','HIGA2','HIGA2','HIGA2','HIGA2']

# data_info = [[140,14,74],[140,14,75],[140,14,76],[140,14,77],[140,14,78],[140,14,79],[140,14,81]]
#
# infor_name = ['GA','GA','GA','GA','GA','HIGA','HIGA']
modular_num = 3
data_info = [[140,2,0],[140,2,1],[140,3,0],[140,3,1],[140,5,0],[140,5,1],[140,5,2],[140,5,3],[140,6,0],[140,6,1],[250,5,4],[250,5,5],[250,5,6],[250,5,7]]
data_info = [[140,3,0]]
infor_name = ['9-6','9-6','9-6','9-6','9-6','9-6','9-6','9-6','9-6','9-6','9-6','9-6','9-6','9-6','9-6','9-6','9-6','9-6','9-6','9-6','9-6','9-6','9-6','9-6','9-6','9-6','9-6','9-6','50*50','50*50','50*50','100*100','100*100']


# data_info = [[200,14,65]]
# infor_name = ['GA']
infor_all = get_info(data_info)
#普通遗传算法
# infor_name = ['14_8','14_13','14_14','14_15','14_16']
title_name = 'Fitness'


draw_picture(infor_all,infor_name,title_name)
# draw_plot_picture(infor_all,data_info)
