import copy
import random
import xlrd
import matplotlib.pyplot as plt
import os
import sys

'''该部分代码用于绘制每层使用同一种模块侧视图，对应文件New_gaijin.py生成的文件'''
def generate_data():
    node1 = [(0, 0), (8, 0), (0, 3), (8, 3), (11, 0), (19, 0), (11, 3), (19, 3)]
    nodes_all = []
    for i in range(story_num):
        for j in range(8):
            nodes_all.append((node1[j][0], node1[j][1] + 3.8 * i))
    beam1 = [(0, 1), (2, 3), (4, 5), (6, 7)]
    beams_all = []
    for i in range(story_num):
        for j in range(4):
            beams_all.append((beam1[j][0] + 8 * i, beam1[j][1] + 8 * i))
    colu1 = [(0, 2), (1, 3), (4, 6), (5, 7)]
    colus_all = []
    for i in range(story_num):
        for j in range(4):
            colus_all.append((colu1[j][0] + 8 * i, colu1[j][1] + 8 * i))

    con1 = [(2, 8), (3, 9), (6, 12), (7, 13)]
    cons_all = []
    for i in range(story_num-1):
        for j in range(4):
            cons_all.append((con1[j][0] + 8 * i, con1[j][1] + 8 * i))

    corr1 = [(1, 4), (3, 6)]
    corrs_all = []
    for i in range(story_num):
        for j in range(2):
            corrs_all.append((corr1[j][0] + 8 * i, corr1[j][1] + 8 * i))

    #底梁中点坐标
    mid_point_bo = [(4, 0), (15, 0)]
    mid_trpoint_bo = [(3, 0), (5, 0),(14, 0), (16, 0)]
    mid_point_bo_all = []
    mid_trpoint_bo_all = []
    for i in range(story_num):
        mid_temp = []
        for j in range(len(mid_point_bo)):
            mid_temp.append((mid_point_bo[j][0], mid_point_bo[j][1] + 3.8 * i))
        mid_point_bo_all.append(mid_temp)
    for i in range(story_num):
        tri_temp = []
        for j in range(len(mid_trpoint_bo)):
            tri_temp.append((mid_trpoint_bo[j][0], mid_trpoint_bo[j][1] + 3.8 * i))
        mid_trpoint_bo_all.append(tri_temp)
    #顶梁中点编号
    mid_point_to = [(4, 3), (15, 3)]
    mid_trpoint_to = [(3, 3), (5, 3),(14, 3), (16, 3)]
    mid_point_to_all = []
    mid_trpoint_to_all = []
    for i in range(story_num):
        mid_temp = []
        for j in range(len(mid_point_to)):
            mid_temp.append((mid_point_to[j][0], mid_point_to[j][1] + 3.8 * i))
        mid_point_to_all.append(mid_temp)
    for i in range(story_num):
        tri_temp = []
        for j in range(len(mid_trpoint_to)):
            tri_temp.append((mid_trpoint_to[j][0], mid_trpoint_to[j][1] + 3.8 * i))
        mid_trpoint_to_all.append(tri_temp)
    #顶梁底梁一起坐标
    mid_beam = [(4, 0), (15, 0),(4, 3), (15, 3),(3, 0), (5, 0),(14, 0), (16, 0),(3, 3), (5, 3),(14, 3), (16, 3),(0,0),(8,0),(11,0),(19,0),(0,3),(8,3),(11,3),(19,3)]
    mid_beam_all = []
    for i in range(story_num):
        tri_temp = []
        for j in range(len(mid_beam)):
            tri_temp.append((mid_beam[j][0], mid_beam[j][1] + 3.8 * i))
        mid_beam_all.append(tri_temp)
    #人字支撑
    person_brace = [[2,12],[2,13],[3,14],[3,15]]
    #交叉支撑
    exchange_brace = [[13, 16], [12, 17], [19, 14], [18, 15]]
    #双交叉支撑
    dou_exchange_brace = [[8, 12], [4, 16], [9, 13], [5, 17],[10, 14], [6, 18], [7, 19], [11, 15]]
    brace_data = [mid_beam_all,person_brace,exchange_brace,dou_exchange_brace]
    # 一层梁编号
    beam_bo_1 = [0, 2]
    beam_bo_all = []
    for i in range(story_num):
        beam_bo_all.append([beam_bo_1[0] + 4 * i, beam_bo_1[1] + 4 * i])

    beam_ce_1 = [1, 3]
    beam_ce_all = []
    for i in range(story_num):
        beam_ce_all.append([beam_ce_1[0] + 4 * i, beam_ce_1[1] + 4 * i])

    col_ce_1 = [0, 1, 2, 3]
    col_ce_all = []
    for i in range(story_num):
        col_ce_all.append([col_ce_1[0] + 4 * i, col_ce_1[1] + 4 * i, col_ce_1[2] + 4 * i, col_ce_1[3] + 4 * i])

    member_all = []
    for i in range(story_num):
        member_all.append(beam_ce_all[i])
        member_all.append(beam_bo_all[i])
        member_all.append(col_ce_all[i])
    story_1_point = [0,1,2,3,4,5,6,7]
    story_all_point = []
    for i in range(story_num):
        story_temp_point = []
        for j in range(len(story_1_point)):
            story_temp_point.append(story_1_point[j]+i*8)
        story_all_point.append(story_temp_point)
    return member_all,nodes_all,beams_all,colus_all,cons_all,corrs_all,brace_data

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


def draw_side_all(num_pop,iter,wb):
    sheet1 = wb.sheet_by_index(0)
    for it in range(len(iter)):
        iter_num = iter[it]*num_pop+1
        for z in range(18):
            rows = sheet1.row_values(iter_num)[z]
            member_section.append(rows)

        member_all, nodes_all, beams_all, colus_all, cons_all, corrs_all,brace_data = generate_data()
        draw_clo_beam(member_all, member_section, nodes_all, beams_all, colus_all)
        draw_corr_conn(cons_all, nodes_all, corrs_all)

        # 设置坐标轴范围
        plt.xlim(-3, 30)
        plt.ylim(-5, 30)

        # 设置坐标轴标签
        plt.xlabel('X')
        plt.ylabel('Y')

        # 显示图形
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
        # APIPath = os.path.join(os.getcwd(), f'frame_side')
        # SpecifyPath = True
        # if not os.path.exists(APIPath):
        #     try:
        #         os.makedirs(APIPath)
        #     except OSError:
        #         pass
        # path1 = os.path.join(APIPath, f'loss{iter[it]}')
        # plt.savefig(path1, dpi=300)
        # plt.close()

def draw_brace(brace_data,brace_type,brace_dis):
    brace = brace_data[int(brace_type)]
    for i in range(len(brace_dis)):
        if brace_dis[i] == 1:
            for line in brace:
                x_values = [brace_data[0][i][line[0]][0], brace_data[0][i][line[1]][0]]
                y_values = [brace_data[0][i][line[0]][1], brace_data[0][i][line[1]][1]]
                plt.plot(x_values, y_values, 'b-', linewidth=4, color='grey')


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
story_num = 6
pop_num = 139#第n代种群
pop_size = 30#种群数量
member_section = []
wb = xlrd.open_workbook(
    filename=f'D:\desktop\os\optimization of structure\optimization of structure\optimization of structure\out_all_infor\\run_infor_14_100.xls',
    formatting_info=True)

iter=[0,2,10,14,77,153,186,199]
# draw_side_all(30,iter,wb)

#获得梁柱截面编号
sheet1 = wb.sheet_by_index(0)
for z in range(3*story_num):
    rows = sheet1.row_values(pop_num*(pop_size+1)+1)[z]
    member_section.append(rows)
#获得支撑类型
sheet2 = wb.sheet_by_index(1)
brace_type = sheet2.row_values(pop_num*(pop_size+1)+1)[num_var]
#获得支撑分布情况
brace_dis = []
for z in range(num_var+num_room+3*story_num,num_var+num_room+4*story_num):
    rows = sheet2.row_values(pop_num*(pop_size+1)+1)[z]
    brace_dis.append(rows)

member_all,nodes_all,beams_all,colus_all,cons_all,corrs_all,brace_data = generate_data()

draw_clo_beam(member_all,member_section,nodes_all,beams_all,colus_all)
draw_corr_conn(cons_all,nodes_all,corrs_all)
draw_brace(brace_data,brace_type,brace_dis)

# 设置坐标轴范围
plt.xlim(-3, 30)
plt.ylim(-5,60)

# 设置坐标轴标签
plt.xlabel('X')
plt.ylabel('Y')

# 显示图形
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
