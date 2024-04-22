import copy
import random
import xlrd
import matplotlib.pyplot as plt

def generate_data():
    node1 = [(0, 0), (8, 0), (0, 3), (8, 3), (11, 0), (19, 0), (11, 3), (19, 3)]
    nodes_all = []
    for i in range(6):
        for j in range(8):
            nodes_all.append((node1[j][0], node1[j][1] + 3.8 * i))
    beam1 = [(0, 1), (2, 3), (4, 5), (6, 7)]
    beams_all = []
    for i in range(6):
        for j in range(4):
            beams_all.append((beam1[j][0] + 8 * i, beam1[j][1] + 8 * i))
    colu1 = [(0, 2), (1, 3), (4, 6), (5, 7)]
    colus_all = []
    for i in range(6):
        for j in range(4):
            colus_all.append((colu1[j][0] + 8 * i, colu1[j][1] + 8 * i))

    con1 = [(2, 8), (3, 9), (6, 12), (7, 13)]
    cons_all = []
    for i in range(5):
        for j in range(4):
            cons_all.append((con1[j][0] + 8 * i, con1[j][1] + 8 * i))

    corr1 = [(1, 4), (3, 6)]
    corrs_all = []
    for i in range(6):
        for j in range(2):
            corrs_all.append((corr1[j][0] + 8 * i, corr1[j][1] + 8 * i))


    # 一层梁编号
    beam_bo_1 = [0, 2]
    beam_bo_all = []
    for i in range(6):
        beam_bo_all.append([beam_bo_1[0] + 4 * i, beam_bo_1[1] + 4 * i])

    beam_ce_1 = [1, 3]
    beam_ce_all = []
    for i in range(6):
        beam_ce_all.append([beam_ce_1[0] + 4 * i, beam_ce_1[1] + 4 * i])

    col_ce_1 = [0, 1, 2, 3]
    col_ce_all = []
    for i in range(6):
        col_ce_all.append([col_ce_1[0] + 4 * i, col_ce_1[1] + 4 * i, col_ce_1[2] + 4 * i, col_ce_1[3] + 4 * i])

    member_all = []
    for i in range(6):
        member_all.append(beam_ce_all[i])
        member_all.append(beam_bo_all[i])
        member_all.append(col_ce_all[i])
    return member_all,nodes_all,beams_all,colus_all,cons_all,corrs_all

def draw_clo_beam(member_all,member_section,nodes_all,beams_all,colus_all):
    for i in range(len(member_all)):
        if len(member_all[i]) == 2:
            for j in range(len(member_all[i])):
                if member_section[i] <= 6:
                    c = 'red'
                    x_values = [nodes_all[beams_all[int(member_all[i][j])][0]][0],
                                nodes_all[beams_all[int(member_all[i][j])][1]][0]]
                    y_values = [nodes_all[beams_all[int(member_all[i][j])][0]][1],
                                nodes_all[beams_all[int(member_all[i][j])][1]][1]]
                    plt.plot(x_values, y_values, 'b-', color=c, linewidth=int(list_new[int(member_section[i])]*1.3 + 1))
                elif member_section[i] >= 6:
                    c = 'b'
                    x_values = [nodes_all[beams_all[int(member_all[i][j])][0]][0],
                                nodes_all[beams_all[int(member_all[i][j])][1]][0]]
                    y_values = [nodes_all[beams_all[int(member_all[i][j])][0]][1],
                                nodes_all[beams_all[int(member_all[i][j])][1]][1]]
                    plt.plot(x_values, y_values, 'b-', color=c, linewidth=int(list_new[int(member_section[i])]*1.3 + 1))
        if len(member_all[i]) == 4:
            for j in range(len(member_all[i])):
                if member_section[i] <= 6:
                    c = 'red'
                    x_values = [nodes_all[colus_all[int(member_all[i][j])][0]][0],
                                nodes_all[colus_all[int(member_all[i][j])][1]][0]]
                    y_values = [nodes_all[colus_all[int(member_all[i][j])][0]][1],
                                nodes_all[colus_all[int(member_all[i][j])][1]][1]]
                    plt.plot(x_values, y_values, 'b-', color=c, linewidth=int(list_new[int(member_section[i])]*1.3 + 1))
                elif member_section[i] >= 6:
                    c = 'b'
                    x_values = [nodes_all[colus_all[int(member_all[i][j])][0]][0],
                                nodes_all[colus_all[int(member_all[i][j])][1]][0]]
                    y_values = [nodes_all[colus_all[int(member_all[i][j])][0]][1],
                                nodes_all[colus_all[int(member_all[i][j])][1]][1]]
                    plt.plot(x_values, y_values, 'b-', color=c, linewidth=int(list_new[int(member_section[i])]*1.3 + 1))

def draw_corr_conn(cons_all,nodes_all,corrs_all):
    for line in cons_all:
        x_values = [nodes_all[line[0]][0], nodes_all[line[1]][0]]
        y_values = [nodes_all[line[0]][1], nodes_all[line[1]][1]]
        plt.plot(x_values, y_values, 'b-', linewidth=4, color='grey')

    for line in corrs_all:
        x_values = [nodes_all[line[0]][0], nodes_all[line[1]][0]]
        y_values = [nodes_all[line[0]][1], nodes_all[line[1]][1]]
        plt.plot(x_values, y_values, 'b-', linewidth=4, color='grey')
    for node in nodes_all:
        plt.plot(node[0], node[1], 'ro', markersize=4, color='grey')

area = [3090,3814,3568,4356,5041,4644,6096,6800,7600,8400,9600,10800,11600,13600]
fit_ini = copy.deepcopy(area)
luyi = copy.deepcopy(area)
luyi.sort()
sort_num = []
lst = list(range(1, len(fit_ini) + 1))
list_new = []
for i in range(len(fit_ini)):
    sort_num.append(fit_ini.index(luyi[i]))
for i in range(len(fit_ini)):
    list_new.append(lst[sort_num[i]])

num_var = 14
num_room = 1
member_section = []
wb = xlrd.open_workbook(
    filename=f'D:\desktop\os\optimization of structure\optimization of structure\optimization of structure\out_all_infor\\run_infor_14_67.xls',
    formatting_info=True)
sheet1 = wb.sheet_by_index(0)
for z in range(18):
    rows = sheet1.row_values(1)[z]
    member_section.append(rows)



member_all,nodes_all,beams_all,colus_all,cons_all,corrs_all = generate_data()
draw_clo_beam(member_all,member_section,nodes_all,beams_all,colus_all)
draw_corr_conn(cons_all,nodes_all,corrs_all)

# 设置坐标轴范围
plt.xlim(-3, 30)
plt.ylim(-5,30)

# 设置坐标轴标签
plt.xlabel('X')
plt.ylabel('Y')

# 显示图形
plt.gca().set_aspect('equal', adjustable='box')
plt.show()