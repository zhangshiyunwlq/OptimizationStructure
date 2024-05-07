import numpy as np
import model_to_sap as ms

def run_column(modular_all_num,sections_data_c1,modular_infos2,f1,f2,f3,f4,f5,f6,b1,b2,b3,b4,b5,b6,t1,t2,t3,t4,t5,t6):
    for i in range(int(modular_all_num / 6)):
        modular_infos2[i] = ms.Modular_Info_Initialization(type='regular', top_edge=sections_data_c1[t1],
                                                           bottom_edge=sections_data_c1[b1],
                                                           column_edge=sections_data_c1[f1])
    for i in range(int((modular_all_num / 6)), int(2 * (modular_all_num / 6))):
        modular_infos2[i] = ms.Modular_Info_Initialization(type='regular', top_edge=sections_data_c1[t2],
                                                           bottom_edge=sections_data_c1[b2],
                                                           column_edge=sections_data_c1[f2])
    for i in range(int(2 * (modular_all_num / 6)), int(3 * (modular_all_num / 6))):
        modular_infos2[i] = ms.Modular_Info_Initialization(type='regular', top_edge=sections_data_c1[t3],
                                                           bottom_edge=sections_data_c1[b3],
                                                           column_edge=sections_data_c1[f3])
    for i in range(int(3 * (modular_all_num / 6)), int(4 * (modular_all_num / 6))):
        modular_infos2[i] = ms.Modular_Info_Initialization(type='regular', top_edge=sections_data_c1[t4],
                                                           bottom_edge=sections_data_c1[b4],
                                                           column_edge=sections_data_c1[f4])
    for i in range(int(4 * (modular_all_num / 6)), int(5 * (modular_all_num / 6))):
        modular_infos2[i] = ms.Modular_Info_Initialization(type='regular', top_edge=sections_data_c1[t5],
                                                           bottom_edge=sections_data_c1[b5],
                                                           column_edge=sections_data_c1[f5])
    for i in range(int(5 * (modular_all_num / 6)), int(6 * (modular_all_num / 6))):
        modular_infos2[i] = ms.Modular_Info_Initialization(type='regular', top_edge=sections_data_c1[t6],
                                                           bottom_edge=sections_data_c1[b6],
                                                           column_edge=sections_data_c1[f6])

#storey(1)+room
def run_column_room(building_room_labels,edges_all_labels,pop_room_label,modular_all_num,sections_data_c1,modular_infos2,pop_num):
    edges_labels = [['top_edge', 'bottom_edge', 'column_edge'], ['left_member', 'right_member', 'short_member'],
                    ['front_member', 'back_member', 'long_member'], ['column', 'short_member', 'long_member']]
    for z in range(6):
        for i in range(int(z*(modular_all_num / 6)), int((z+1) * (modular_all_num / 6))):
            if building_room_labels[i] == 1:
                modular_infos2[i] = ms.Modular_Info_Initialization(type='regular', top_edge=sections_data_c1[pop_num[z*5*3+3*0+0]],
                                                                   bottom_edge=sections_data_c1[pop_num[z*5*3+3*0+1]],
                                                                   column_edge=sections_data_c1[pop_num[z*5*3+3*0+2]])
            elif building_room_labels[i] == 2:
                modular_infos2[i] = ms.Modular_Info_Initialization(type='regular', top_edge=sections_data_c1[pop_num[z*5*3+3*1+0]],
                                                                   bottom_edge=sections_data_c1[pop_num[z*5*3+3*1+1]],
                                                                   column_edge=sections_data_c1[pop_num[z*5*3+3*1+2]])
            elif building_room_labels[i] == 3:
                modular_infos2[i] = ms.Modular_Info_Initialization(type='regular', top_edge=sections_data_c1[pop_num[z*5*3+3*2+0]],
                                                                   bottom_edge=sections_data_c1[pop_num[z*5*3+3*2+1]],
                                                                   column_edge=sections_data_c1[pop_num[z*6+3*2+2]])
            elif building_room_labels[i] == 4:
                modular_infos2[i] = ms.Modular_Info_Initialization(type='regular', top_edge=sections_data_c1[pop_num[z*5*3+3*3+0]],
                                                                   bottom_edge=sections_data_c1[pop_num[z*5*3+3*3+1]],
                                                                   column_edge=sections_data_c1[pop_num[z*5*3+3*3+2]])
            elif building_room_labels[i] == 5:
                modular_infos2[i] = ms.Modular_Info_Initialization(type='regular', top_edge=sections_data_c1[pop_num[z*5*3+3*4+0]],
                                                                   bottom_edge=sections_data_c1[pop_num[z*5*3+3*4+1]],
                                                                   column_edge=sections_data_c1[pop_num[z*5*3+3*4+2]])
            elif building_room_labels[i] == 6:
                modular_infos2[i] = ms.Modular_Info_Initialization(type='regular', top_edge=sections_data_c1[pop_num[z*5*3+3*5+0]],
                                                                   bottom_edge=sections_data_c1[pop_num[z*5*3+3*5+1]],
                                                                   column_edge=sections_data_c1[pop_num[z*5*3+3*5+2]])


def run_column_room_v1(building_room_labels, pop_room_label, modular_all_num, sections_data_c1,
                       modular_infos2, pop_num):
    edges_labels = [['top_edge', 'bottom_edge', 'column_edge'], ['left_member', 'right_member', 'short_member'],
                    ['front_member', 'back_member', 'long_member'], ['column', 'short_member', 'long_member']]

    for z in range(6):
        for i in range(int(z * (modular_all_num / 6)), int((z + 1) * (modular_all_num / 6))):
            #房间标签
            pop_room_label = list(map(int, pop_room_label))
            j = int(building_room_labels[i])-1
            if pop_room_label[i] == 1 or pop_room_label[i] == 4 or pop_room_label[i] == 7 or pop_room_label[i] == 10:
                modular_infos2[i] = ms.Modular_Info_Initialization(type='regular', top_edge=sections_data_c1[
                    pop_num[z * 5 * 3 + 3 * j + 0]],bottom_edge=sections_data_c1[pop_num[z * 5 * 3 + 3 * j + 1]],
                    column_edge=sections_data_c1[pop_num[z * 5 * 3 + 3 * j + 2]])
            elif pop_room_label[i] == 2 or pop_room_label[i] == 5 or pop_room_label[i] == 8 or pop_room_label[i] == 11:
                modular_infos2[i] = ms.Modular_Info_Initialization(type='regular', all_member=sections_data_c1[
                    pop_num[z * 5 * 3 + 3 * j + 0]],right_member=sections_data_c1[pop_num[z * 5 * 3 + 3 * j + 1]],
                    short_member=sections_data_c1[pop_num[z * 4 * 3 + 3 * j + 2]])
            elif pop_room_label[i] == 3 or pop_room_label[i] == 6 or pop_room_label[i] == 9 or pop_room_label[i] == 12:
                modular_infos2[i] = ms.Modular_Info_Initialization(type='regular', short_member=sections_data_c1[
                    pop_num[z * 5 * 3 + 3 * j + 0]],long_member=sections_data_c1[pop_num[z * 5 * 3 + 3 * j + 1]],
                    column=sections_data_c1[pop_num[z * 4 * 3 + 3 * j + 2]])
            # elif pop_room_label[i] == 4 or pop_room_label[i] == 8 or pop_room_label[i] == 12 or pop_room_label[i] == 16:
            #     modular_infos2[i] = ms.Modular_Info_Initialization(type='regular', column=sections_data_c1[
            #         pop_num[z * 5 * 3 + 3 * 0 + 0]],short_member=sections_data_c1[pop_num[z * 5 * 3 + 3 * j + 1]],
            #         long_member=sections_data_c1[pop_num[z * 5 * 3 + 3 * j + 2]])
# 每层分两组
def run_column_room_story5(building_room_labels, pop_room_label, modular_all_num, sections_data_c1,
                       modular_infos2, pop_num):
    edges_labels = [['top_edge', 'bottom_edge', 'column_edge'], ['left_member', 'right_member', 'short_member'],
                    ['front_member', 'back_member', 'long_member'], ['column', 'short_member', 'long_member']]

    for z in range(6):
        for i in range(int(z * (modular_all_num / 6)), int((z + 1) * (modular_all_num / 6))):
            #房间标签
            pop_room_label = list(map(int, pop_room_label))
            j = int(building_room_labels[i])-1
            # modular_infos2[i] = ms.Modular_Info_Initialization(type='regular', top_edge=sections_data_c1[
            #     pop_num[z * 2 * 3 + 3 * j + 0]],bottom_edge=sections_data_c1[pop_num[z * 2 * 3 + 3 * j + 1]],
            #     column_edge=sections_data_c1[int(pop_num[z * 2 * 3 + 3 * j + 2]+7)])

            modular_infos2[i] = ms.Modular_Info_Initialization(type='regular', top_edge=sections_data_c1[
                pop_num[z * 2 * 3 + 3 * j + 0]],bottom_edge=sections_data_c1[pop_num[z * 2 * 3 + 3 * j + 1]],
                column_edge=sections_data_c1[pop_num[z * 2 * 3 + 3 * j + 2]])

def run_column_room_story1(building_room_labels, pop_room_label, modular_all_num, sections_data_c1,
                       modular_infos2, pop_num,story_num):
    edges_labels = [['top_edge', 'bottom_edge', 'column_edge'], ['left_member', 'right_member', 'short_member'],
                    ['front_member', 'back_member', 'long_member'], ['column', 'short_member', 'long_member']]

    for z in range(story_num):
        for i in range(int(z * (modular_all_num / story_num)), int((z + 1) * (modular_all_num / story_num))):
            #房间标签
            pop_room_label = list(map(int, pop_room_label))
            j = int(building_room_labels[i])-1
            # modular_infos2[i] = ms.Modular_Info_Initialization(type='regular', top_edge=sections_data_c1[
            #     pop_num[z * 2 * 3 + 3 * j + 0]],bottom_edge=sections_data_c1[pop_num[z * 2 * 3 + 3 * j + 1]],
            #     column_edge=sections_data_c1[int(pop_num[z * 2 * 3 + 3 * j + 2]+7)])

            modular_infos2[i] = ms.Modular_Info_Initialization(type='regular', top_edge=sections_data_c1[
                pop_num[z * 3 + 0]],bottom_edge=sections_data_c1[pop_num[z * 3 + 1]],
                column_edge=sections_data_c1[pop_num[z * 3 + 2]])

def run_column_room_modular_section(building_room_labels, pop_room_label, modular_all_num, sections_data_c1,
                       modular_infos2, pop_num,story_num,zone_num):
    edges_labels = [['top_edge', 'bottom_edge', 'column_edge'], ['left_member', 'right_member', 'short_member'],
                    ['front_member', 'back_member', 'long_member'], ['column', 'short_member', 'long_member']]
    labels = building_room_labels
    section_all_modular = []
    brace_all_modular = []

    modular_type1 = [i for i in range(3)]

    zone_type_all = []  # 所有区域的编号索引
    for i in range(zone_num):
        modular_type_temp = []
        for j in range(len(modular_type1)):
            modular_type_temp.append(modular_type1[j] + 3 * i)
        zone_type_all.append(modular_type_temp)
    #生成每个模块的三个截面
    section_all = []
    for i in range(len(labels)):
        for j in range(3):
            temp1 = zone_type_all[int(labels[i])][j]
            section_all.append(pop_num[temp1])

    #生成所有房间的索引
    modular_type1 = [i for i in range(3)]
    #生成对每个模块的截面编号索引
    modular_type_all= []
    for i in range(len(labels)):
        modular_type_temp = []
        for j in range(len(modular_type1)):
            modular_type_temp.append(modular_type1[j]+3*i)
        modular_type_all.append(modular_type_temp)

    #建立每个房间的信息
    for i in range(len(labels)):
        temp0 = int(section_all[int(modular_type_all[i][0])])
        temp1 = int(section_all[int(modular_type_all[i][1])])
        temp2 = int(section_all[int(modular_type_all[i][2])])
        modular_infos2[i] = ms.Modular_Info_Initialization(type='regular', top_edge=sections_data_c1[
            temp0], bottom_edge=sections_data_c1[temp1],column_edge=sections_data_c1[temp2])


def run_column_room_modular(building_room_labels, pop_room_label, modular_all_num, sections_data_c1,
                       modular_infos2, pop_num,story_num):
    edges_labels = [['top_edge', 'bottom_edge', 'column_edge'], ['left_member', 'right_member', 'short_member'],
                    ['front_member', 'back_member', 'long_member'], ['column', 'short_member', 'long_member']]
    modular_type1 = [i for i in range(3)]
    #生成对每个模块的截面编号索引
    modular_type_all= []
    for i in range(modular_all_num):
        modular_type_temp = []
        for j in range(len(modular_type1)):
            modular_type_temp.append(modular_type1[j]+3*i)
        modular_type_all.append(modular_type_temp)
    for i in range(modular_all_num):
        top_b = pop_num[int(modular_type_all[i][0])]
        bot_b = pop_num[int(modular_type_all[i][1])]
        col_b = pop_num[int(modular_type_all[i][0])]
        modular_infos2[i] = ms.Modular_Info_Initialization(type='regular', top_edge=sections_data_c1[int(top_b)], bottom_edge=sections_data_c1[int(bot_b)],
                                                           column_edge=sections_data_c1[int(col_b)])



#storey(2)+room
def run_story_room(building_room_labels,modular_all_num,sections_data_c1,modular_infos2,pop_num,story_num):
    for z in range(story_num):
        for i in range(int(z*6/story_num*(modular_all_num / 6)), int((z+1) * (6/story_num)*(modular_all_num / 6))):
            if building_room_labels[i] == 1:
                modular_infos2[i] = ms.Modular_Info_Initialization(type='regular', top_edge=sections_data_c1[pop_num[z*6*3+3*0+0]],
                                                                   bottom_edge=sections_data_c1[pop_num[z*6*3+3*0+1]],
                                                                   column_edge=sections_data_c1[pop_num[z*6+3*0+2]])
            elif building_room_labels[i] == 2:
                modular_infos2[i] = ms.Modular_Info_Initialization(type='regular', top_edge=sections_data_c1[pop_num[z*6*3+3*1+0]],
                                                                   bottom_edge=sections_data_c1[pop_num[z*6*3+3*1+1]],
                                                                   column_edge=sections_data_c1[pop_num[z*6*3+3*1+2]])
            elif building_room_labels[i] == 3:
                modular_infos2[i] = ms.Modular_Info_Initialization(type='regular', top_edge=sections_data_c1[pop_num[z*6*3+3*2+0]],
                                                                   bottom_edge=sections_data_c1[pop_num[z*6*3+3*2+1]],
                                                                   column_edge=sections_data_c1[pop_num[z*6*3+3*2+2]])
            elif building_room_labels[i] == 4:
                modular_infos2[i] = ms.Modular_Info_Initialization(type='regular', top_edge=sections_data_c1[pop_num[z*6*3+3*3+0]],
                                                                   bottom_edge=sections_data_c1[pop_num[z*6*3+3*3+1]],
                                                                   column_edge=sections_data_c1[pop_num[z*6*3+3*3+2]])
            elif building_room_labels[i] == 5:
                modular_infos2[i] = ms.Modular_Info_Initialization(type='regular', top_edge=sections_data_c1[pop_num[z*6*3+3*4+0]],
                                                                   bottom_edge=sections_data_c1[pop_num[z*6*3+3*4+1]],
                                                                   column_edge=sections_data_c1[pop_num[z*6*3+3*4+2]])
            elif building_room_labels[i] == 6:
                modular_infos2[i] = ms.Modular_Info_Initialization(type='regular', top_edge=sections_data_c1[pop_num[z*6*3+3*5+0]],
                                                                   bottom_edge=sections_data_c1[pop_num[z*6*3+3*5+1]],
                                                                   column_edge=sections_data_c1[pop_num[z*6*3+3*5+2]])