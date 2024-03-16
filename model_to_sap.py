import configparser
import numpy as np
import copy
import os
import sys
import comtypes.client
import win32com.client
import xlwt
import  math as m

import data_to_json as dj
import modular_utils as md


def rewrite(dataset):
    """
    提取.ini文件的内容，并放入字典中
    :param dataset:
    :return:
    """
    if isinstance(dataset, configparser.SectionProxy) == True:
        temp_dict = {}
        for i in dataset:
            temp_dict[i] = dataset[i]
        return temp_dict
    else:
        print("the data set is not from the Configuration file")

def get_label_info(cfg_file_name="School_load_info.ini"):
    cfg_section_data = configparser.ConfigParser()
    cfg_section_data.read(f"{cfg_file_name}", encoding='utf-8')
    # 创建section_datasets保存所有.ini文件的"头文件"名称
    section_datasets = []
    for k in cfg_section_data:
        section_datasets.append(k)
    # 删除DEFAULT默认项目
    del section_datasets[0]
    label_info = []
    for i in section_datasets:
        temp = rewrite(cfg_section_data[i])
        label_info.append(list(temp.values()))
        label_info = [[float(x) for x in sublist] for sublist in label_info]

    return label_info

def distance(a):
    x = a[0]
    y = a[1]
    z = a[2]
    r = m.sqrt(x**2+y**2+z**2)
    return r

def get_section_info(section_type, cfg_file_name="RHS_section_data.ini"):
    cfg_section_data = configparser.ConfigParser()
    cfg_section_data.read(f"{cfg_file_name}", encoding='utf-8')
    # 创建section_datasets保存所有.ini文件的"头文件"名称
    section_datasets = []
    for k in cfg_section_data:
        section_datasets.append(k)
    # 删除DEFAULT默认项目
    del section_datasets[0]
    # 创建字典temp，保存截面type为c1的所有构件信息
    dict_temp = {}
    set_temp = []
    section_data = []
    # print(len(section_datasets))
    for i in range(len(section_datasets)):
        # if cfg_section_data[section_datasets[i]]['type'] == f'{section_type}':
        set_temp.append(section_datasets[i])
        if i == len(section_datasets) and len(set_temp) == 0:
            print("empty")
    # print(set_temp)

    for i in range(len(set_temp)):
        dict_temp[i] = rewrite(cfg_section_data[set_temp[i]])
        for j in dict_temp[i].keys():
            # import pdb;
            # pdb.set_trace()
            if j != 'type':
                dict_temp[i][j] = float( dict_temp[i][j])

    for item in dict_temp.keys():
        temp1 = list(dict_temp[item].values())
        temp1.pop(0)
        section_data.append([float(j) for j in temp1])
    return dict_temp, set_temp, np.array(section_data)

def Wind_Load_Y(ModularBuilding,modulars, SapModel,storys, bz, us, uz1,uz2,w0,modular_length_num):
    for i in range(int((storys-1) * (len(modulars))/6)+modular_length_num, int(storys * (len(modulars))/6)):
        A = (distance(modulars[i].modular_nodes[modulars[i].modular_four_planes[2][0]] - modulars[i].modular_nodes[
            modulars[i].modular_four_planes[2][1]])) * (distance(modulars[i].modular_nodes[modulars[i].modular_four_planes[2][0]] - modulars[i].modular_nodes[
                modulars[i].modular_four_planes[2][3]]))
        P1 = bz * us * uz1 * w0 * A * 0.25
        P2 = bz * us * uz2 * w0 * A * 0.25
        modular_edges = ModularBuilding.building_room_edges[i]
        indx1, indx2 = modular_edges[6]
        indx3, indx4 = modular_edges[10]
        point1 = "nodes" + str(indx1)
        point2 = "nodes" + str(indx2)
        point3 = "nodes" + str(indx3)
        point4 = "nodes" + str(indx4)
        Value1 = [0, P1, 0, 0, 0, 0]
        ret = SapModel.PointObj.SetLoadForce(point1, "WINDY", Value1)
        ret = SapModel.PointObj.SetLoadForce(point2, "WINDY", Value1)
        Value2 = [0, P2, 0, 0, 0, 0]
        ret = SapModel.PointObj.SetLoadForce(point3, "WINDY", Value2)
        ret = SapModel.PointObj.SetLoadForce(point4, "WINDY", Value2)

def Wind_Load_X(ModularBuilding,modulars, SapModel,storys, bz, us, uz1,uz2,w0,modular_length_num):
    for i in [0 + (storys-1) * modular_length_num*2,modular_length_num-1 + (storys-1) * modular_length_num*2]:
        A = (distance(modulars[i].modular_nodes[modulars[i].modular_four_planes[1][0]] - modulars[i].modular_nodes[
            modulars[i].modular_four_planes[1][1]])) * (distance(modulars[i].modular_nodes[modulars[i].modular_four_planes[1][0]] - modulars[i].modular_nodes[
                modulars[i].modular_four_planes[1][3]]))
        P1 = bz * us * uz1 * w0 * A * 0.25
        P2 = bz * us * uz2 * w0 * A * 0.25
        modular_edges = ModularBuilding.building_room_edges[i]
        indx1, indx2 = modular_edges[7]
        indx3, indx4 = modular_edges[11]
        point1 = "nodes" + str(indx1)
        point2 = "nodes" + str(indx2)
        point3 = "nodes" + str(indx3)
        point4 = "nodes" + str(indx4)
        Value1 = [-P1, 0, 0, 0, 0, 0]
        ret = SapModel.PointObj.SetLoadForce(point1, "WINDX", Value1)
        ret = SapModel.PointObj.SetLoadForce(point2, "WINDX", Value1)
        Value2 = [-P2, 0, 0, 0, 0, 0]
        ret = SapModel.PointObj.SetLoadForce(point3, "WINDX", Value2)
        ret = SapModel.PointObj.SetLoadForce(point4, "WINDX", Value2)
    # #走廊
    # for i in [len(modulars)-6 + (storys-1)]:
    #     A = (distance(modulars[i].modular_nodes[modulars[i].modular_four_planes[1][0]] - modulars[i].modular_nodes[
    #         modulars[i].modular_four_planes[1][1]])) * (distance(modulars[i].modular_nodes[modulars[i].modular_four_planes[1][0]] - modulars[i].modular_nodes[
    #             modulars[i].modular_four_planes[1][3]]))
    #     P1 = bz * us * uz1 * w0 * A * 0.25
    #     P2 = bz * us * uz2 * w0 * A * 0.25
    #     modular_edges = ModularBuilding.building_room_edges[i]
    #     indx1, indx2 = modular_edges[7]
    #     indx3, indx4 = modular_edges[11]
    #     point1 = "nodes" + str(indx1)
    #     point2 = "nodes" + str(indx2)
    #     point3 = "nodes" + str(indx3)
    #     point4 = "nodes" + str(indx4)
    #     Value1 = [P1, 0, 0, 0, 0, 0]
    #     ret = SapModel.PointObj.SetLoadForce(point1, "WINDX", Value1)
    #     ret = SapModel.PointObj.SetLoadForce(point2, "WINDX", Value1)
    #     Value2 = [P2, 0, 0, 0, 0, 0]
    #     ret = SapModel.PointObj.SetLoadForce(point3, "WINDY", Value2)
    #     ret = SapModel.PointObj.SetLoadForce(point4, "WINDY", Value2)

def get_point_displacement(nodes,SapModel):
    displacements = []
    ObjectElm = 0
    NumberResults = 0
    m001 = []
    result = []
    Obj = []
    Elm = []
    ACase = []
    StepType = []
    StepNum = []
    U1 = []
    U2 = []
    U3 = []
    R1 = []
    R2 = []
    R3 = []
    ObjectElm = 0
    [NumberResults, Obj, Elm, ACase, StepType, StepNum, U1, U2, U3, R1, R2, R3,ret] = SapModel.Results.JointDispl(nodes, ObjectElm, NumberResults, Obj,Elm, ACase, StepType, StepNum, U1, U2, U3, R1, R2, R3)
    return Obj,U1, U2, U3, R1, R2, R3

def get_frame_reactions(frames,SapModel):
    result = []
    Object11 = 0
    Obj = []
    ObjSta = []
    Elm = []
    ElmSta = []
    LoadCase = []
    StepType = []
    StepNum = []
    NumberResults = 0
    P = []
    V2 = []
    V3 = []
    T = []
    M2 = []
    M3 = []
    [NumberResults, Obj, ObjSta, Elm, ElmSta, LoadCase, StepType, StepNum, P, V2, V3, T, M2, M3,
     ret] = SapModel.Results.FrameForce(frames, Object11, NumberResults, Obj, ObjSta, Elm, ElmSta, LoadCase, StepType, StepNum, P, V2, V3, T, M2,M3)
    return Obj, ObjSta,P, V2, V3, T, M2,M3

def get_section_area(a,b):
    height = a[0]
    width = a[1]
    t1 = a[2]
    t2 = a[3]
    if b[1] == 'c0':
        area_1 = height * width - (height-2*t1)*(width-t2)
    elif b[1] == 'I0':
        area_1 = height * width - 2*(height-2*t2)*(width-2*t1)
    elif b[1] == 'b0':
        area_1 = height * width - (height - 2 * t2) * (width - 2 * t1)

    return area_1

def output_force(Nodes,SapModel,modulars,ModularBuilding,frame_section_info,frame_section_info000,modular_length_num,story_num):
    name_re = []
    frame_reactions = []
    frame_reactions_all = []
    for modular_indx in range(len(modulars)):
        modular_edges = ModularBuilding.building_room_edges[modular_indx]
        for edge_indx in range(len(modular_edges)):
            result = []
            P_na = []
            mm1 = np.zeros((7, 3))
            mm2 = []
            Obj, ObjSta, P, V2, V3, T, M2, M3 = get_frame_reactions(
                "frame_" + str(modular_indx) + '_' + str(edge_indx), SapModel)
            if len(P) != 0:
                # result.append(Obj)
                result.append(ObjSta)
                result.append(P)
                result.append(V2)
                result.append(V3)
                result.append(T)
                result.append(M2)
                result.append(M3)
                num_fra = len(Obj)
                mid_num = int(0.5 * (num_fra))
                name_re.append(Obj[0])
                for i in range(len(result)):
                    mm1[i][0] = result[i][0]
                    mm1[i][1] = result[i][mid_num]
                    mm1[i][2] = result[i][num_fra - 1]
                frame_reactions.append(mm1)
                frame_reactions_all.append(result)
    mm = ["ObjSta", "P", "V2", "V3", "T", "M2", "M3"]



    # 计算reactions
    column_section_area = []
    column_name = []
    jaingjiaqi = frame_section_info000
    for i in range(len(modulars)):
        for j in range(4):
            section_area = get_section_area(frame_section_info[12 * i + j], frame_section_info000[12 * i + j])
            column_section_area.append(section_area)
    column_section_length = []
    for modular_indx in range(len(modulars)):
        modular_edges = ModularBuilding.building_room_edges[modular_indx]
        for edge_indx in range(4):
            indx1, indx2 = modular_edges[edge_indx]
            # Point1 = "nodes" + str(indx1)
            # Point2 = "nodes" + str(indx2)
            x1, y1, z1 = Nodes[indx1]
            x2, y2, z2 = Nodes[indx2]
            l = m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
            column_section_length.append(l)
            column_name.append(name_re[modular_indx * 12 + edge_indx])
    weight_all = 0.0
    for i in range(len(column_section_length)):
        weight_all += column_section_length[i] * column_section_area[i] * 0.00000000785

    # print(len(column_section_length))
    # print(len(column_section_area))

    # print(weight_all)
    # print(column_section_length)
    # print(column_section_area)
    # 输出总用钢量
    all_section_area = []
    for i in range(len(modulars)):
        for j in range(12):
            section_area = get_section_area(frame_section_info[12 * i + j],frame_section_info000[12 * i + j])
            all_section_area.append(section_area)
    all_section_length = []
    for modular_indx in range(len(modulars)):
        modular_edges = ModularBuilding.building_room_edges[modular_indx]
        for edge_indx in range(12):
            indx1, indx2 = modular_edges[edge_indx]
            # Point1 = "nodes" + str(indx1)
            # Point2 = "nodes" + str(indx2)
            x1, y1, z1 = Nodes[indx1]
            x2, y2, z2 = Nodes[indx2]
            l = m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
            all_section_length.append(l)
    weight_all1 = 0.0
    for i in range(len(all_section_length)):
        weight_all1 += all_section_length[i] * all_section_area[i] * 0.00000000785

    # 计算质量比
    mass_all = []
    mass_all_radio = []
    for i in range(story_num):
        each_floor_mass = 0.0
        for j in range(int(i*(len(all_section_length))/story_num),int((i+1)*(len(all_section_length))/story_num)):
            each_floor_mass += all_section_length[j] * all_section_area[j] * 0.00000000785
        mass_all.append(each_floor_mass)
    for i in range(story_num-1):
        mass_all_radio.append(mass_all[i+1]/mass_all[i])
    mass_all_radio.append(1.0)

    # 柱强度验算
    rx = 1
    ry = 1
    f = 355
    wnx = 906908.8
    wny = 101172.04
    faix = 0.8
    faiy = 0.8
    bmx = 0.9
    btx = 0.9
    bmy = 0.9
    bty = 0.9
    Nex = 5
    Ney = 5
    n_canshu = 1
    faiby = 0.8
    faibx = 0.8
    G1 = []
    G2 = []
    G3 = []
    #每个柱最大值
    G1_all = []
    G2_all = []
    G3_all = []
    # 柱的稳定验算

    for i in range(len(modulars)):
        for j in range(4):
            if i*4+j<=100000:
                G11 = (abs(frame_reactions[12 * i + j][1]) / f / column_section_area[4 * i + j]) + (
                            abs(frame_reactions[12 * i + j][5]) / f / rx / frame_section_info[12 * i + j][4]
                            ) + (abs(frame_reactions[12 * i + j][6]) / f / ry / frame_section_info[12 * i + j][5]) - 1
                G21 = (abs(frame_reactions[12 * i + j][1]) / f / column_section_area[4 * i + j] / faix) + (
                            bmx * abs(frame_reactions[12 * i + j][5]) / f / rx / frame_section_info[12 * i + j][4]
                            / (1 - 0.8 * abs(frame_reactions[12 * i + j][1]) / abs(
                        frame_section_info[12 * i + j][6]) / 1846434.18 * column_section_length[4 * i + j] *
                               column_section_length[4 * i + j])) + n_canshu * (
                                  bty * abs(frame_reactions[12 * i + j][6]) / f / frame_section_info[12 * i + j][5]
                                  / faiby) - 1
                G21001 = (abs(frame_reactions[12 * i + j][1]) / f / column_section_area[4 * i + j] / faix)
                G21002 = (bmx * abs(frame_reactions[12 * i + j][5]) / f / rx / frame_section_info[12 * i + j][4]
                          / (1 - 0.8 * abs(frame_reactions[12 * i + j][1]) / abs(
                            frame_section_info[12 * i + j][6]) / 1846434.18 * column_section_length[4 * i + j] *
                             column_section_length[4 * i + j]))
                G21003 = n_canshu * (bty * abs(frame_reactions[12 * i + j][6]) / f / frame_section_info[12 * i + j][5]
                                     / faiby) - 1
                G21008 = abs(frame_reactions[12 * i + j][1]) / abs(
                        frame_section_info[12 * i + j][6]) / 1846434.18 * column_section_length[4 * i + j] * column_section_length[4 * i + j]
                G21004 = abs(frame_reactions[12 * i + j][6])
                G21005 = frame_section_info[12 * i + j][4]
                G21006 = abs(frame_reactions[12 * i + j][1])
                G21007 = abs(frame_section_info[12 * i + j][6])
                G31 = (abs(frame_reactions[12 * i + j][1]) / f / column_section_area[4 * i + j] / faiy) + n_canshu * (
                            btx * abs(frame_reactions[12 * i + j][5]) / f / frame_section_info[12 * i + j][4]
                            / faibx) + (
                                  bmy * abs(frame_reactions[12 * i + j][6]) / f / ry / frame_section_info[12 * i + j][
                              5] / (1 - 0.8 * abs(frame_reactions[12 * i + j][1]) / frame_section_info[12 * i + j][
                              7] / 1846434.18 * column_section_length[4 * i + j] * column_section_length[
                                        4 * i + j])) - 1
                G1.append(G11)
                G2.append(G21)
                G3.append(G31)
                b001 = frame_reactions[12 * i + j]
                b002 = frame_section_info[12 * i + j]
                G31001 = G31
                G31002 = (abs(frame_reactions[12 * i + j][1]) / f / column_section_area[4 * i + j] / faiy)
                G31003 = n_canshu * (btx * abs(frame_reactions[12 * i + j][5]) / f / frame_section_info[12 * i + j][4]
                                     / faibx)
                G31004 = (bmy * abs(frame_reactions[12 * i + j][6]) / f / ry / frame_section_info[12 * i + j][5] / (
                            1 - 0.8 * (abs(frame_reactions[12 * i + j][1]) * column_section_length[4 * i + j] *
                                       column_section_length[4 * i + j] / frame_section_info[12 * i + j][
                                           7] / 1846434.18))) - 1
                G31005 = (1 - 0.8 * (abs(frame_reactions[12 * i + j][1]) * column_section_length[4 * i + j] *
                                     column_section_length[4 * i + j] / frame_section_info[12 * i + j][7] / 1846434.18))
                G31006 = abs(frame_reactions[12 * i + j][1])
                G31007 = column_section_length[4 * i + j]
                G31008 = frame_section_info[12 * i + j][7]
                G31009 = abs(frame_reactions[12 * i + j][6])
                G310010 = frame_section_info[12 * i + j][5]
            # G31005
    # 梁的稳定验算
    L11 = []
    L21 = []
    L31 = []
    beam_name = []
    for i in range(len(modulars)):
        for j in range(4, 12):
            beam_name.append(name_re[12 * i + j])
            L111 = (abs(frame_reactions[12 * i + j][5]) / f / rx / frame_section_info[12 * i + j][4]) + (
                        abs(frame_reactions[12 * i + j][6]) / f / ry / frame_section_info[12 * i + j][5]) - 1
            L211 = (abs(frame_reactions[12 * i + j][1]) / f / all_section_area[12 * i + j] / faix) + (
                        bmx * abs(frame_reactions[12 * i + j][5]) / f / rx / frame_section_info[12 * i + j][4]
                        / (1 - 0.8 * abs(frame_reactions[12 * i + j][1]) / abs(
                    frame_section_info[12 * i + j][6]) / 1846434.18 * all_section_length[12 * i + j] *
                           all_section_length[12 * i + j])) + n_canshu * (
                               bty * abs(frame_reactions[12 * i + j][6]) / f / frame_section_info[12 * i + j][5]
                               / faiby) - 1
            L311 = (abs(frame_reactions[12 * i + j][1]) / f / all_section_area[12 * i + j] / faiy) + n_canshu * (
                        btx * abs(frame_reactions[12 * i + j][5]) / f / frame_section_info[12 * i + j][4]
                        / faibx) + (bmy * abs(frame_reactions[12 * i + j][5]) / f / ry / frame_section_info[12 * i + j][
                6] / (1 - 0.8
                      * abs(frame_reactions[12 * i + j][1]) / frame_section_info[12 * i + j][7] / 1846434.18 *
                      all_section_length[12 * i + j] * all_section_length[12 * i + j])) - 1
            L11.append(L111)
            L21.append(L211)
            L31.append(L311)

    for i in range(len(G1)):
        clo1 = []
        clo2 = []
        clo3 = []
        for j in range(3):
            clo11 = abs(G1[i][j])
            clo21 = abs(G2[i][j])
            clo31 = abs(G3[i][j])
            clo1.append(clo11)
            clo2.append(clo21)
            clo3.append(clo31)
        G1_all.append(G1[i][clo1.index(max(clo1))])
        G2_all.append(G2[i][clo2.index(max(clo2))])
        G3_all.append(G3[i][clo3.index(max(clo3))])

    G_max = []
    for i in range(story_num):
        G_1_rea = []
        for j in range(modular_length_num*4*2 * i, modular_length_num*4*2 * (i + 1)):
            G_1_rea.append(G1_all[j])
        mm33 = G1_all[modular_length_num*4*2 * i + G_1_rea.index(max(G_1_rea))]
        G_max.append(mm33)
        # print(f"第{i}层柱G1最大=",mm33)
    for i in range(story_num):
        G_2_rea = []
        for j in range(modular_length_num*4*2 * i, modular_length_num*4*2 * (i + 1)):
            G_2_rea.append(G2_all[j])
        mm33 = G2_all[modular_length_num*4*2 * i + G_2_rea.index(max(G_2_rea))]
        G_max.append(mm33)
        # print(f"第{i}层柱G2最大=",mm33)
    for i in range(story_num):
        G_3_rea = []
        for j in range(modular_length_num*2*4 * i, modular_length_num*4*2 * (i + 1)):
            G_3_rea.append(G3_all[j])
        mm33 = G3_all[modular_length_num*4*2 * i + G_3_rea.index(max(G_3_rea))]
        G_max.append(mm33)
        # print(f"第{i}层柱G3最大=",mm33)
    # print(G1)
    # print(G2)
    # print(G3)
    # print(G1_all)
    # print(G2_all)
    # print(G3_all)
    G1_up = []
    G2_up = []
    G3_up = []
    G1_index = []
    G2_index = []
    G3_index = []
    G1_cloup_name = []
    G2_cloup_name = []
    G3_cloup_name = []
    for i in range(len(G1_all)):
        if G1_all[i] >= 0:
            G1_up.append(G1_all[i])
            G1_index.append(G1_all.index(G1_all[i]))
            # G1_cloup_name.append(column_name[G1_index[i]])
    for i in range(len(G1_index)):
        G1_cloup_name.append(column_name[G1_index[i]])
    for i in range(len(G2_all)):
        if G2_all[i] >= 0:
            G2_up.append(G2_all[i])
            G2_index.append(G2_all.index(G2_all[i]))
            # G2_cloup_name.append(column_name[G2_index[i]])
    for i in range(len(G2_index)):
        G2_cloup_name.append(column_name[G2_index[i]])
    for i in range(len(G3_all)):
        if G3_all[i] >= 0:
            G3_up.append(G3_all[i])
            G3_index.append(G3_all.index(G3_all[i]))
            # G3_cloup_name.append(column_name[G3_index[i]])
    for i in range(len(G3_index)):
        G3_cloup_name.append(column_name[G3_index[i]])

    Bea1_all = []
    Bea2_all = []
    Bea3_all = []
    for i in range(len(L11)):
        clo1 = []
        clo2 = []
        clo3 = []
        for j in range(3):
            clo11 = abs(L11[i][j])
            clo21 = abs(L21[i][j])
            clo31 = abs(L31[i][j])
            clo1.append(clo11)
            clo2.append(clo21)
            clo3.append(clo31)
        Bea1_all.append(L11[i][clo1.index(max(clo1))])
        Bea2_all.append(L21[i][clo2.index(max(clo2))])
        Bea3_all.append(L31[i][clo3.index(max(clo3))])

    G_max_beam = []
    for i in range(story_num):
        G_1_bea = []
        for j in range(modular_length_num*16 * i, modular_length_num*16 * (i + 1)):
            G_1_bea.append(Bea1_all[j])
        mm33 = Bea1_all[modular_length_num*16 * i + G_1_bea.index(max(G_1_bea))]
        G_max_beam.append(mm33)
        # print(f"第{i}层柱G1最大=",mm33)
    for i in range(story_num):
        G_2_bea = []
        for j in range(modular_length_num*16 * i, modular_length_num*16 * (i + 1)):
            G_2_bea.append(Bea2_all[j])
        mm33 = Bea2_all[modular_length_num*16 * i + G_2_bea.index(max(G_2_bea))]
        G_max_beam.append(mm33)
        # print(f"第{i}层柱G2最大=",mm33)
    for i in range(story_num):
        G_3_bea = []
        for j in range(modular_length_num*16 * i, modular_length_num*16 * (i + 1)):
            G_3_bea.append(Bea3_all[j])
        mm33 = Bea3_all[modular_length_num*16 * i + G_3_bea.index(max(G_3_bea))]
        G_max_beam.append(mm33)
        # print(f"第{i}层柱G3最大=",mm33)

    G1_up_bea = []
    G2_up_bea = []
    G3_up_bea = []
    G1_index_bea = []
    G2_index_bea = []
    G3_index_bea = []
    G1_beamup_name = []
    G2_beamup_name = []
    G3_beamup_name = []
    for i in range(len(Bea1_all)):
        if Bea1_all[i] >= 0:
            G1_up_bea.append(Bea1_all[i])
            G1_index_bea.append(Bea1_all.index(Bea1_all[i]))
            # G1_beamup_name.append(beam_name[G1_index_bea[i]])
    for i in range(len(G1_index_bea)):
        G1_beamup_name.append(beam_name[G1_index_bea[i]])
    for i in range(len(Bea2_all)):
        if Bea2_all[i] >= 0:
            G2_up_bea.append(Bea2_all[i])
            G2_index_bea.append(Bea2_all.index(Bea2_all[i]))
            # G2_beamup_name.append(beam_name[G2_index_bea[i]])
    for i in range(len(G2_index_bea)):
        G2_beamup_name.append(beam_name[G2_index_bea[i]])
    for i in range(len(Bea3_all)):
        if Bea3_all[i] >= 0:
            G3_up_bea.append(Bea3_all[i])
            G3_index_bea.append(Bea3_all.index(Bea3_all[i]))
            # G3_beamup_name.append(beam_name[G3_index_bea[i]])
    for i in range(len(G3_index_bea)):
        G3_beamup_name.append(beam_name[G3_index_bea[i]])
    all_up_frame_name = []
    G1_cloup_num = []
    G2_cloup_num = []
    G3_cloup_num = []
    all_up_frame_name.extend(G1_cloup_name)
    all_up_frame_name.extend(G2_cloup_name)
    all_up_frame_name.extend(G3_cloup_name)
    all_up_frame_name.extend(G1_beamup_name)
    all_up_frame_name.extend(G2_beamup_name)
    all_up_frame_name.extend(G3_beamup_name)
    all_up_frame_number = len(all_up_frame_name)
    for i in range(len(G1_cloup_name)):
        G1_cloup_num.append(Get_section_location(G1_cloup_name[i]))
    for i in range(len(G1_beamup_name)):
        G1_cloup_num.append(Get_section_location(G1_beamup_name[i]))
    for i in range(len(G2_cloup_name)):
        G2_cloup_num.append(Get_section_location(G2_cloup_name[i]))
    for i in range(len(G2_beamup_name)):
        G2_cloup_num.append(Get_section_location(G2_beamup_name[i]))
    for i in range(len(G3_cloup_name)):
        G3_cloup_num.append(Get_section_location(G3_cloup_name[i]))
    for i in range(len(G3_beamup_name)):
        G3_cloup_num.append(Get_section_location(G3_beamup_name[i]))
    all_up_num = [G1_cloup_num,G2_cloup_num,G3_cloup_num]
    all_up_frame_data = np.zeros((all_up_frame_number, 3))
    num1 = 0
    for i in range(len(G1_cloup_name)):
        all_up_frame_data[num1][0] = G1_up[i]
        all_up_frame_data[num1][1] = G1_index[i]
        all_up_frame_data[num1][2] = 1
        num1 += 1
    for i in range(len(G2_cloup_name)):
        all_up_frame_data[num1][0] = G2_up[i]
        all_up_frame_data[num1][1] = G2_index[i]
        all_up_frame_data[num1][2] = 2
        num1 += 1
    for i in range(len(G3_cloup_name)):
        all_up_frame_data[num1][0] = G3_up[i]
        all_up_frame_data[num1][1] = G3_index[i]
        all_up_frame_data[num1][2] = 3
        num1 += 1
    for i in range(len(G1_beamup_name)):
        all_up_frame_data[num1][0] = G1_up_bea[i]
        all_up_frame_data[num1][1] = G1_index_bea[i]
        all_up_frame_data[num1][2] = 1
        num1 += 1
    for i in range(len(G2_beamup_name)):
        all_up_frame_data[num1][0] = G2_up_bea[i]
        all_up_frame_data[num1][1] = G2_index_bea[i]
        all_up_frame_data[num1][2] = 2
        num1 += 1
    for i in range(len(G3_beamup_name)):
        all_up_frame_data[num1][0] = G3_up_bea[i]
        all_up_frame_data[num1][1] = G3_index_bea[i]
        all_up_frame_data[num1][2] = 3
        num1 += 1
    all_information = []
    all_information.append(frame_reactions)
    all_information.append(name_re)
    all_information.append(G_max)
    all_information.append(G_max_beam)
    all_information.append(frame_reactions_all)
    all_information.append(all_up_frame_name)
    all_information.append(all_up_frame_data)
    all_information.append(weight_all1)
    all_information.append(mm)
    all_information.append(mass_all)
    all_information.append(mass_all_radio)
    all_information.append(all_up_num)
    return all_information

def output_dis(Nodes,SapModel,modulars,ModularBuilding,modular_length_num,story_num):
    mid_displacements = []
    name_frame_mid = []
    all_mid_frame_labels = []#框架编号
    displacements = []
    frame_reactions = []
    name_re = []
    name_all_nodes = []
    all_mid_dis = []
    X_dis_bottom = []
    Y_dis_bottom = []
    Z_dis_bottom = []
    X_dis_top = []
    Y_dis_top = []
    Z_dis_top = []
    # all nodes displacements
    for i in range(len(Nodes)):
        result = []
        Obj,U1, U2, U3, R1, R2, R3 = get_point_displacement("nodes"+str(i), SapModel)
        # if len(U1) != 0:
        name_all_nodes.append(Obj[0])
        result.append(U1[0])
        result.append(U2[0])
        result.append(U3[0])
        result.append(R1[0])
        result.append(R2[0])
        result.append(R3[0])
        displacements.append(result)
    displacements = np.array(displacements)
    # mid nodes displacements
    for modular_indx in range(len(modulars)):
        modular_edges = ModularBuilding.building_room_edges[modular_indx]
        room_dis = []
        for edge_indx in range(len(modular_edges)):
            mid_frame_label = []
            result = []
            Obj,U1, U2, U3, R1, R2, R3 = get_point_displacement("nodes_mid" + str(modular_indx) + '_' + str(edge_indx), SapModel)
            if len(U1) != 0:
                result.append(U1[0])
                result.append(U2[0])
                result.append(U3[0])
                name_frame_mid.append("nodes_mid" + str(modular_indx) + '_' + str(edge_indx))
                mid_displacements.append(result)
                mid_frame_label.append(modular_indx)
                mid_frame_label.append(edge_indx)
            if len(mid_frame_label) != 0:
               all_mid_frame_labels.append(mid_frame_label)
            # if edge_indx<=3 and len(U1) != 0:
            #     all_mid_dis.append(result)
    for i in range(modular_length_num*2*story_num):
        for j in range(4):
            all_mid_dis.append(mid_displacements[j+i*12])

    mid_displacements = np.array(mid_displacements)
    all_mid_dis = np.array(all_mid_dis)
    # 输出最大位移

    for i in range(modular_length_num*2*story_num):
        for j in range(8*i,8*i+4):
            X_dis_bottom.append(displacements[j][0])
            Y_dis_bottom.append(displacements[j][1])
            Z_dis_bottom.append(displacements[j][2])
        for j in range(8*i+4,8*(i+1)):
            X_dis_top.append(displacements[j][0])
            Y_dis_top.append(displacements[j][1])
            Z_dis_top.append(displacements[j][2])
    X_dis_max = []
    X_dis_max_label = []
    Y_dis_max = []
    Y_dis_max_label = []
    Z_dis_max = []
    Z_dis_max_label = []
    X_dis_min= []
    Y_dis_min = []
    Z_dis_min = []

    for i in range(story_num):
        X_dis = []
        Y_dis = []
        Z_dis = []
        for j in range(modular_length_num*2*4*i,modular_length_num*2*4*(i+1)):
            X_dis.append(X_dis_top[j])
            Y_dis.append(Y_dis_top[j])
            Z_dis.append(Z_dis_top[j])
        X_dis_max.append(max(X_dis))
        X_dis_min.append(min(X_dis))
        X_dis_max_label.append((X_dis.index(max(X_dis)))+modular_length_num*16*i)
        Y_dis_max.append(max(Y_dis))
        Y_dis_min.append(min(Y_dis))
        Y_dis_max_label.append(Y_dis.index(max(Y_dis))+modular_length_num*16*i)
        Z_dis_max.append(max(Z_dis))
        Z_dis_min.append(min(Z_dis))
        Z_dis_max_label.append(Z_dis.index(max(Z_dis))+modular_length_num*16*i)
    X_dis_ave = []
    Y_dis_ave = []
    for i in range(len(Y_dis_max)):
        X_dis_ave.append((X_dis_max[i] + X_dis_min[i])*0.5)
        Y_dis_ave.append((Y_dis_max[i] + Y_dis_min[i])*0.5)
    X_dis_radio = []
    Y_dis_radio = []
    for i in range(len(Y_dis_max)):
        Y_dis_radio.append(Y_dis_max[i] / (3000*(i+1)))
        X_dis_radio.append(X_dis_max[i] / (3000*(i+1)))

    #输出层间位移
    X_interdis_top = []
    Y_interdis_top = []
    for i in range(modular_length_num*8):
        X_interdis_top.append(X_dis_top[i]/3000)
        Y_interdis_top.append(Y_dis_top[i]/3000)
    for i in range(modular_length_num*8,len(X_dis_top)):
        X_interdis_top.append((X_dis_top[i] - X_dis_top[i - modular_length_num*2*4])/3000)
        Y_interdis_top.append((Y_dis_top[i] - Y_dis_top[i - modular_length_num*2*4])/3000)
    X_interdis_max = []
    X_interdis_max_label = []
    Y_interdis_max = []
    Y_interdis_max_label = []
    X_interdis_min= []
    Y_interdis_min = []
    for i in range(story_num):
        X_interdis = []
        Y_interdis = []

        for j in range(modular_length_num*8*i,modular_length_num*8*(i+1)):
            X_interdis.append(X_interdis_top[j])
            Y_interdis.append(Y_interdis_top[j])

        X_interdis_max.append(max(X_interdis))
        X_interdis_min.append(min(X_interdis))
        X_interdis_max_label.append((X_interdis.index(max(X_interdis)))+modular_length_num*16*i)
        Y_interdis_max.append(max(Y_interdis))
        Y_interdis_min.append(min(Y_interdis))
        Y_interdis_max_label.append(Y_interdis.index(max(Y_interdis))+modular_length_num*16*i)

    X_interdis_ave = []
    Y_interdis_ave = []
    for i in range(len(Y_interdis_max)):
        X_interdis_ave.append((X_interdis_max[i] + X_interdis_min[i])*0.5)
        Y_interdis_ave.append((Y_interdis_max[i] + Y_interdis_min[i])*0.5)
    X_interdis_radio = []
    Y_interdis_radio = []
    for i in range(len(Y_interdis_ave)):
        if Y_interdis_ave[i]==0:
            Y_interdis_ave[i]+=0.00001
    for i in range(len(X_interdis_ave)):
        if X_interdis_ave[i]==0:
            X_interdis_ave[i]+=0.00001
    for i in range(len(Y_interdis_max)):
        if X_interdis_max[i] <= 0.001:
            X_interdis_radio.append(1)
        else:
            X_interdis_radio.append(X_interdis_max[i] / X_interdis_ave[i])
        if Y_interdis_max[i] <= 0.001:
            Y_interdis_radio.append(1)
        else:
            Y_interdis_radio.append(Y_interdis_max[i] / Y_interdis_ave[i])

    # 计算欧式距离
    ou_all_dis = []
    for i in range(len(displacements)):
        ou_all_dis.append(distance(displacements[i]))
    ou_mid_dis = []
    for i in range(len(all_mid_dis)):
        ou_mid_dis.append(distance(all_mid_dis[i]))
    Joint_dis = []
    Joint_dis.append(X_dis_max)
    Joint_dis.append(Y_dis_max)
    # Joint_dis.append(Z_dis_max)

    Joint_dis.append(X_dis_ave)
    Joint_dis.append(Y_dis_ave)
    Joint_dis.append(X_dis_radio)
    Joint_dis.append(Y_dis_radio)

    Joint_dis.append(X_interdis_max)
    Joint_dis.append(Y_interdis_max)
    Joint_dis.append(X_interdis_ave)
    Joint_dis.append(Y_interdis_ave)
    Joint_dis.append(X_interdis_radio)
    Joint_dis.append(Y_interdis_radio)

    Joint_dis.append(X_dis_max_label)
    Joint_dis.append(Y_dis_max_label)
    Joint_dis.append(Z_dis_max_label)
    return ou_all_dis, ou_mid_dis, displacements, mid_displacements, name_frame_mid, name_all_nodes,Joint_dis

def list_int(list1):
    lista = []
    for i in list1:
        # i = int(i)
        lista.append(i)
    return lista

def list_int11(list1):
    lista = []
    for i in range(len(list1)):
        # i = int(i)
        lista.append(list1[i])
    return lista

def SAPanalysis_GA_run(APIPath):


    cfg = configparser.ConfigParser()
    cfg.read("Configuration.ini", encoding='utf-8')
    ProgramPath = cfg['SAP2000PATH']['dirpath']


    AttachToInstance = False
    SpecifyPath = True
    if not os.path.exists(APIPath):
        try:
            os.makedirs(APIPath)
        except OSError:
            pass
    ModelPath = os.path.join(APIPath, 'API_1-001.sdb')
    helper = comtypes.client.CreateObject('SAP2000v1.Helper')
    helper = helper.QueryInterface(comtypes.gen.SAP2000v1.cHelper)
    if AttachToInstance:
        # attach to a running instance of SAP2000
        try:
            # get the active SapObject
            mySapObject = helper.Getobject("CSI.SAP2000.API.SapObject")
        except (OSError, comtypes.COMError):
            print("No running instance of the program found or failed to attach.")
            sys.exit(-1)
    else:
        if SpecifyPath:
            try:
                # 'create an instance of the SAPObject from the specified path
                mySapObject = helper.CreateObject(ProgramPath)
            except (OSError, comtypes.COMError):
                print("Cannot start a new instance of the program from" + ProgramPath)
                sys.exit(-1)
        else:
            try:
                # create an instance of the SapObject from the latest installed SAP2000
                mySapObject = helper.CreateObjectProgID("CSI.SAP2000.API.SapObject")
            except (OSError, comtypes.COMError):
                print("Cannot start a new instance of the program")
                sys.exit(-1)

        # start SAP2000 application
        mySapObject.ApplicationStart()

    # create SapModel object
    SapModel = mySapObject.SapModel
    # initialize model
    SapModel.InitializeNewModel()

    # create new blank model
    return mySapObject, ModelPath, SapModel

def Modular_Info_Initialization(**kwargs):
    # import pdb;
    # pdb.set_trace()
    modular_info = {}
    for key in kwargs.keys():
        modular_info[key] = kwargs[key]
    return modular_info

def Get_section_location(fr_name):
    name = fr_name
    for i in range(6,len(name)):
        if name[i] == '_':
            num = i
    front_num = []
    back_num = []
    for i in range(6,num):
        front_num.append(int(name[i]))
    for i in range(num+1,len(name)):
        back_num.append(int(name[i]))
    room_num = 0
    fr_num = 0
    for i in range(len(front_num)):
        room_num += front_num[i]*10**(len(front_num)-1-i)
    for i in range(len(back_num)):
        fr_num += back_num[i]*10**(len(back_num)-1-i)
    return [room_num,fr_num]


def Run_GA_sap(mySapObject, ModelPath, SapModel, ModularBuilding,pop_room_label,width_joint,modular_length_num,story_num):

    ret = SapModel.File.NewBlank()

    # switch units
    N_mm_C = 9
    ret = SapModel.SetPresentUnits(N_mm_C)

    """ material definition """
    # to be added
    conf = configparser.ConfigParser()
    # print(type(conf))  # conf是类
    conf.read('materials_data.ini')

    sections = conf.sections()  # 获取配置文件中所有sections，sections是列表
    # print(sections)
    option = conf.options(conf.sections()[0])

    # item = conf.items(sections[0])
    # print(item[0][1])
    material_title = []
    material_info_all = []
    pop_room_label = list(map(int, pop_room_label))
    for i in range(len(sections)):
        material_info = []
        item = conf.items(sections[i])
        ret = SapModel.PropMaterial.SetMaterial(f"{item[0][1]}", 1, -1)
        ret = SapModel.PropMaterial.SetWeightAndMass(f"{item[0][1]}", 2, float(item[1][1]))
        ret = SapModel.PropMaterial.SetMPIsotropic(f"{item[0][1]}", float(item[2][1]), float(item[3][1]), float(item[4][1]))
        ret = SapModel.PropMaterial.SetOSteel_1(f"{item[0][1]}", float(item[5][1]), float(item[6][1]), float(item[7][1]),
                                                float(item[8][1]), int(item[9][1]), int(item[10][1]), float(item[11][1]),
                                                float(item[12][1]), float(item[13][1]), float(item[14][1]))
        for i in range(14):
            material_info.append(item[i][1])
        material_info_all.append(material_info)
        material_title.append(item[0][1])

    """ cross section definition """
    '''defination from the modular perspective'''
    frame_section_name = []
    frame_section_info = []
    frame_section_info000 = []
    modular_all_section = []
    modulars = ModularBuilding.building_modulars
    # modular_nodes_indx = ModularBuilding.nodes_indx
    for modular_indx in range(len(modulars)):
        modular_i_info = modulars[modular_indx].modular_info
        modular_i_edges = ModularBuilding.building_room_edges[modular_indx]
        for edge_indx in range(len(modular_i_edges)):
            section_info = []
            frame_section_info111 = []
            section_name = "frame_section_" + str(modular_indx) + '_' + str(edge_indx)
            material_name = f"{material_title[2]}"
            section_data = modular_i_info[modulars[modular_indx].modular_edge_labels[edge_indx]]
            # if pop_room_label[modular_indx] == 1 or pop_room_label[modular_indx] == 4 or pop_room_label[modular_indx] == 7 or pop_room_label[modular_indx] == 10:
            #     section_data = modular_i_info[modulars[modular_indx].modular_edge_labels[edge_indx]]
            # elif pop_room_label[modular_indx] == 2 or pop_room_label[modular_indx] == 5 or pop_room_label[modular_indx] == 8 or pop_room_label[modular_indx] == 11:
            #     section_data = modular_i_info[modulars[modular_indx].modular_edge_labels_1[edge_indx]]
            # elif pop_room_label[modular_indx] == 3 or pop_room_label[modular_indx] == 6 or pop_room_label[modular_indx] == 9 or pop_room_label[modular_indx] == 12:
            #     section_data = modular_i_info[modulars[modular_indx].modular_edge_labels_2[edge_indx]]
            # elif pop_room_label[modular_indx] == 4 or pop_room_label[modular_indx] == 8 or pop_room_label[modular_indx] == 12 or pop_room_label[modular_indx] == 16:
            #     section_data = modular_i_info[modulars[modular_indx].modular_edge_labels_3[edge_indx]]

            if section_data['type'] == 'c0':
                ret = SapModel.PropFrame.SetChannel(section_name, material_name,
                                                    section_data['outside_depth'], section_data['outside_flange_width'],
                                                    section_data['flange_thickness'], section_data['web_thickness'], -1)
                frame_section_name.append(section_name)
                frame_section_info111.append( f"{material_title[0]}")
                frame_section_info111.append("c0")
                frame_section_info000.append(frame_section_info111)
                section_info.append(section_data['outside_depth'])
                section_info.append(section_data['outside_flange_width'])
                section_info.append(section_data['flange_thickness'])
                section_info.append(section_data['web_thickness'])
                section_info.append(section_data['s22'])
                section_info.append(section_data['s33'])
                section_info.append(section_data['i22'])
                section_info.append(section_data['i33'])
                frame_section_info.append(section_info)
            elif section_data['type'] == 'I0':
                ret = SapModel.PropFrame.SetISection(section_name, material_name,
                                                    section_data['heigth'], section_data['width'],
                                                    section_data['tf'], section_data['tw'],section_data['width'],section_data['tf'],-1)
                frame_section_name.append(section_name)
                frame_section_info111.append( f"{material_title[0]}")
                frame_section_info111.append("I0")
                frame_section_info000.append(frame_section_info111)
                section_info.append(section_data['heigth'])
                section_info.append(section_data['width'])
                section_info.append(section_data['tw'])
                section_info.append(section_data['tf'])
                section_info.append(section_data['s22'])
                section_info.append(section_data['s33'])
                section_info.append(section_data['i22'])
                section_info.append(section_data['i33'])
                frame_section_info.append(section_info)
            elif section_data['type'] == 'b0':
                ret = SapModel.PropFrame.SetTube(section_name, material_name,
                                                    section_data['outside_depth'], section_data['outside_flange_width'],
                                                    section_data['flange_thickness'], section_data['web_thickness'],-1)
                frame_section_name.append(section_name)
                frame_section_info111.append( f"{material_title[0]}")
                frame_section_info111.append("b0")
                frame_section_info000.append(frame_section_info111)
                section_info.append(section_data['outside_depth'])
                section_info.append(section_data['outside_flange_width'])
                section_info.append(section_data['flange_thickness'])
                section_info.append(section_data['web_thickness'])
                section_info.append(section_data['s22'])
                section_info.append(section_data['s33'])
                section_info.append(section_data['i22'])
                section_info.append(section_data['i33'])
                frame_section_info.append(section_info)

    """ connection definition """
    # 设置连接性质
    # 自由度
    MyDof = [True, True, True, True, True, True]
    # 固定
    MyFixed = [False, False, False, False, False, False]
    # 非线性
    MyNonLinear = [True, True, True, True, True, True]
    # 初始刚度
    MyKe = [980000, 120000, 120000, 2200000000, 3400000000, 3400000000]
    # 阻尼系数
    MyCe = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
    MyF = [-12000000, -12000000, 0, 12000000, 12000000]
    MyD = [-30, -5, 0, 5, 30]

    ret = SapModel.PropLink.SetMultiLinearElastic("HOR1", MyDof, MyFixed, MyNonLinear, MyKe, MyCe, 2, 0)
    ret = SapModel.PropLink.SetMultiLinearElastic("VER1", MyDof, MyFixed, MyNonLinear, MyKe, MyCe, 2, 0)
    # 竖直连接力位移曲线
    VERF_u1 = [-1.15e6, -1.15e6, 0, 1.07e6, 1.07e6]
    VERD_u1 = [-2.34, -1.14, 0, 1.14, 2.34]
    VERF_u23 = [-3.31e5, -3.31e5, 0, 3.31e5, 3.31e5]
    VERD_u23 = [-5.42, -2.70, 0, 2.70, 5.42]
    VERF_r1 = [-4.8e7, -4.8e7, -3.15e7, 0, 3.15e7, 4.8e7, 4.8e7]
    VERD_r1 = [-0.073, -0.038, -0.015, 0, 0.015, 0.038, 0.073]
    VERF_r23 = [-5.5e7, -5.5e7, -3.68e7, 0, 3.68e7, 5.38e7, 5.43e7]
    VERD_r23 = [-0.055, -0.027, -0.011, 0, 0.011, 0.027, 0.055]

    HORF_u1 = [-6.85e5, -6.90e5, 0, 6.90e5, 6.85e5]
    HORD_u1 = [-0.45, -0.23, 0, 0.23, 0.45]
    HORF_u23 = [-6.24e5, -6.24e5, 0, 6.30e5, 6.30e5]
    HORD_u23 = [-0.23, -0.11, 0, 0.11, 0.23]
    HORF_r1 = [-75000000, -75000000, -4.85e7, 0, 4.93e7, 75000000, 75000000]
    HORD_r1 = [-0.012, -0.0063, -0.0024, 0, 0.0024, 0.0063, 0.012]
    HORF_r2 = [-129000000, -129000000, -8.56e7, 0, 8.56e7, 1.29e8, 1.29e8]
    HORD_r2 = [-0.022, -0.011, -0.0044, 0, 0.0041, 0.011, 0.0215]
    HORF_r3 = [-3.40e7, -3.40e7, -2.31e7, 0, 2.22e7, 3.40e7, 3.40e7]
    HORD_r3 = [-0.0060, -0.0032, -0.0014, 0, 0.0012, 0.0026, 0.0053]
    # 多段连接塑性
    # ret = SapModel.PropLink.SetMultiLinearPoints("MLE1", 2, 5, MyF, MyD, 3, 9, 12, 0.75, 0.8, .1)
    # 多段连接弹性
    ret = SapModel.PropLink.SetMultiLinearPoints("HOR1", 1, 5, HORF_u1, HORD_u1)
    ret = SapModel.PropLink.SetMultiLinearPoints("HOR1", 2, 5, HORF_u23, HORD_u23)
    ret = SapModel.PropLink.SetMultiLinearPoints("HOR1", 3, 5, HORF_u23, HORD_u23)
    ret = SapModel.PropLink.SetMultiLinearPoints("HOR1", 4, 7, HORF_r1, HORD_r1)
    ret = SapModel.PropLink.SetMultiLinearPoints("HOR1", 5, 7, HORF_r2, HORD_r2)
    ret = SapModel.PropLink.SetMultiLinearPoints("HOR1", 6, 7, HORF_r3, HORD_r3)

    ret = SapModel.PropLink.SetMultiLinearPoints("VER1", 1, 5, VERF_u1, VERD_u1)
    ret = SapModel.PropLink.SetMultiLinearPoints("VER1", 2, 5, VERF_u23, VERD_u23)
    ret = SapModel.PropLink.SetMultiLinearPoints("VER1", 3, 5, VERF_u23, VERD_u23)
    ret = SapModel.PropLink.SetMultiLinearPoints("VER1", 4, 7, VERF_r1, VERD_r1)
    ret = SapModel.PropLink.SetMultiLinearPoints("VER1", 5, 7, VERF_r23, VERD_r23)
    ret = SapModel.PropLink.SetMultiLinearPoints("VER1", 6, 7, VERF_r23, VERD_r23)




    # if modular_i_info['type']

    """ define the structure """
    Nodes = ModularBuilding.building_nodes
    Joints_hor = ModularBuilding.building_room_joints_hor
    Joints_ver = ModularBuilding.building_room_joints_ver
    Corr_beams = ModularBuilding.corr_beams
    Room_indx = ModularBuilding.building_nodes_indx

    ''' 1st adding points '''
    for node_indx in range(len(Nodes)):
        x, y, z = Nodes[node_indx]
        ret = SapModel.PointObj.AddCartesian(x, y, z, None, "nodes"+str(node_indx), "Global")
    """ define frames mid points """
    for modular_indx in range(len(modulars)):
        modular_edges = ModularBuilding.building_room_edges[modular_indx]
        for edge_indx in range(len(modular_edges)):
            indx1, indx2 = modular_edges[edge_indx]
            # Point1 = "nodes" + str(indx1)
            # Point2 = "nodes" + str(indx2)
            x1, y1, z1 = Nodes[indx1]
            x2, y2, z2 = Nodes[indx2]
            x3 = 0.5 * (x1 + x2)
            y3 = 0.5 * (y1 + y2)
            z3 = 0.5 * (z1 + z2)
            ret = SapModel.PointObj.AddCartesian(x3, y3, z3, None, "nodes_mid" + str(modular_indx) + '_' + str(edge_indx), "Global")
    """define frame 3 point"""
    for modular_indx in range(len(modulars)):
        modular_edges = ModularBuilding.building_room_edges[modular_indx]
        for edge_indx in [5, 7, 9, 11]:
            indx1, indx2 = modular_edges[edge_indx]
            # Point1 = "nodes" + str(indx1)
            # Point2 = "nodes" + str(indx2)
            x1, y1, z1 = Nodes[indx1]
            x2, y2, z2 = Nodes[indx2]
            x3 = 0.5 * (x1 + x2)
            y3 = 0.5 * (y1 + y2) -750
            z3 = 0.5 * (z1 + z2)
            y4 = 0.5 * (y1 + y2) +750
            ret = SapModel.PointObj.AddCartesian(x3, y3, z3, None, "nodes_brace" + str(modular_indx) + '_' + str(edge_indx) + '_0', "Global")
            ret = SapModel.PointObj.AddCartesian(x3, y4, z3, None, "nodes_brace" + str(modular_indx) + '_' + str(edge_indx) + '_1', "Global")
    ''' 2nd adding frames '''
    for modular_indx in range(len(modulars)):
        modular_edges = ModularBuilding.building_room_edges[modular_indx]
        for edge_indx in range(len(modular_edges)):
            indx1, indx2 = modular_edges[edge_indx]
            Point1 = "nodes" + str(indx1)
            Point2 = "nodes" + str(indx2)
            name = "frame_" + str(modular_indx) + '_' + str(edge_indx)
            section_name = "frame_section_" + str(modular_indx) + '_' + str(edge_indx)
            ret = SapModel.FrameObj.AddByPoint(Point1, Point2, " ", section_name, name)
    ''' 3rd adding joints '''
    joint_sec, joint_mater = "Rectang", f"{material_title[2]}"
    ret = SapModel.PropFrame.SetRectangle(joint_sec, joint_mater,width_joint,width_joint,-1)
    brace_sec, brace_mater = "Rectang", f"{material_title[2]}"
    ret = SapModel.PropFrame.SetTube(brace_sec, brace_mater, 100, 100,10,10, -1)
    ''' 4rd adding braces '''
    weight_brace = 0.0
    for i in range(len(modulars)):
        #人字支撑
        if pop_room_label[i] == 1:
            nodes_indx = ModularBuilding.building_nodes_indx[i]
            ret = SapModel.FrameObj.AddByPoint("nodes"+str(nodes_indx[0]), "nodes_mid" + str(i) + '_' + str(11), " ", brace_sec, "brace_" + str(i) + '_' + str(0))
            ret = SapModel.FrameObj.AddByPoint("nodes" + str(nodes_indx[6]), "nodes_mid" + str(i) + '_' + str(11), " ",
                                               brace_sec, "brace_" + str(i) + '_' + str(1))
            ret = SapModel.FrameObj.AddByPoint("nodes" + str(nodes_indx[2]), "nodes_mid" + str(i) + '_' + str(9), " ",
                                               brace_sec, "brace_" + str(i) + '_' + str(2))
            ret = SapModel.FrameObj.AddByPoint("nodes" + str(nodes_indx[4]), "nodes_mid" + str(i) + '_' + str(9), " ",
                                               brace_sec, "brace_" + str(i) + '_' + str(3))
            ret = SapModel.FrameObj.AddByPoint("nodes" + str(nodes_indx[4]), "nodes" + str(nodes_indx[7]), " ",
                                               brace_sec, "brace_" + str(i) + '_' + str(4))
            ret = SapModel.FrameObj.AddByPoint("nodes" + str(nodes_indx[5]), "nodes" + str(nodes_indx[6]), " ",
                                               brace_sec, "brace_" + str(i) + '_' + str(5))
            br_back = [Nodes[nodes_indx[5]][0]-Nodes[nodes_indx[6]][0],Nodes[nodes_indx[5]][1]-Nodes[nodes_indx[6]][1],Nodes[nodes_indx[5]][2]-Nodes[nodes_indx[6]][2]]
            leng = distance(br_back)
            wb = (5000*4+leng*2)*(100*100-90*90)*0.00000000785
            weight_brace +=wb

        # 交叉支撑
        elif pop_room_label[i] == 2 :
            nodes_indx = ModularBuilding.building_nodes_indx[i]
            ret = SapModel.FrameObj.AddByPoint("nodes" + str(nodes_indx[0]), "nodes" + str(nodes_indx[7]), " ",
                                               brace_sec, "brace_" + str(i) + '_' + str(0))
            ret = SapModel.FrameObj.AddByPoint("nodes" + str(nodes_indx[6]), "nodes" + str(nodes_indx[1]), " ",
                                               brace_sec, "brace_" + str(i) + '_' + str(1))
            ret = SapModel.FrameObj.AddByPoint("nodes" + str(nodes_indx[2]), "nodes" + str(nodes_indx[5]), " ",
                                               brace_sec, "brace_" + str(i) + '_' + str(2))
            ret = SapModel.FrameObj.AddByPoint("nodes" + str(nodes_indx[3]), "nodes" + str(nodes_indx[4]), " ",
                                               brace_sec, "brace_" + str(i) + '_' + str(3))
            ret = SapModel.FrameObj.AddByPoint("nodes" + str(nodes_indx[4]), "nodes" + str(nodes_indx[7]), " ",
                                               brace_sec, "brace_" + str(i) + '_' + str(4))
            ret = SapModel.FrameObj.AddByPoint("nodes" + str(nodes_indx[5]), "nodes" + str(nodes_indx[6]), " ",
                                               brace_sec, "brace_" + str(i) + '_' + str(5))

            br_back = [Nodes[nodes_indx[5]][0] - Nodes[nodes_indx[6]][0], Nodes[nodes_indx[5]][1] - Nodes[nodes_indx[6]][1],
                       Nodes[nodes_indx[5]][2] - Nodes[nodes_indx[6]][2]]
            leng = distance(br_back)
            wb = (8544 * 4 + leng * 2) *(100*100-90*90)* 0.00000000785
            weight_brace += wb
        #双交叉支撑
        elif pop_room_label[i] == 3:
            nodes_indx = ModularBuilding.building_nodes_indx[i]
            ret = SapModel.FrameObj.AddByPoint("nodes" + str(nodes_indx[0]), "nodes_brace" + str(i) + '_' + str(11) + '_0', " ",
                                               brace_sec, "brace_" + str(i) + '_' + str(0))
            ret = SapModel.FrameObj.AddByPoint("nodes" + str(nodes_indx[1]), "nodes_brace" + str(i) + '_' + str(7) + '_0', " ",
                                               brace_sec, "brace_" + str(i) + '_' + str(1))
            ret = SapModel.FrameObj.AddByPoint("nodes" + str(nodes_indx[6]), "nodes_brace" + str(i) + '_' + str(11) + '_1', " ",
                                               brace_sec, "brace_" + str(i) + '_' + str(2))
            ret = SapModel.FrameObj.AddByPoint("nodes" + str(nodes_indx[7]), "nodes_brace" + str(i) + '_' + str(7) + '_1', " ",
                                               brace_sec, "brace_" + str(i) + '_' + str(3))
            ret = SapModel.FrameObj.AddByPoint("nodes" + str(nodes_indx[2]), "nodes_brace" + str(i) + '_' + str(9) + '_0', " ",
                                               brace_sec, "brace_" + str(i) + '_' + str(4))
            ret = SapModel.FrameObj.AddByPoint("nodes" + str(nodes_indx[3]), "nodes_brace" + str(i) + '_' + str(5) + '_0', " ",
                                               brace_sec, "brace_" + str(i) + '_' + str(5))
            ret = SapModel.FrameObj.AddByPoint("nodes" + str(nodes_indx[4]), "nodes_brace" + str(i) + '_' + str(9) + '_1', " ",
                                               brace_sec, "brace_" + str(i) + '_' + str(6))
            ret = SapModel.FrameObj.AddByPoint("nodes" + str(nodes_indx[5]), "nodes_brace" + str(i) + '_' + str(5) + '_1', " ",
                                               brace_sec, "brace_" + str(i) + '_' + str(7))
            ret = SapModel.FrameObj.AddByPoint("nodes" + str(nodes_indx[4]), "nodes" + str(nodes_indx[7]), " ",
                                               brace_sec, "brace_" + str(i) + '_' + str(8))
            ret = SapModel.FrameObj.AddByPoint("nodes" + str(nodes_indx[5]), "nodes" + str(nodes_indx[6]), " ",
                                               brace_sec, "brace_" + str(i) + '_' + str(9))

            br_back = [Nodes[nodes_indx[5]][0]-Nodes[nodes_indx[6]][0],Nodes[nodes_indx[5]][1]-Nodes[nodes_indx[6]][1],Nodes[nodes_indx[5]][2]-Nodes[nodes_indx[6]][2]]
            leng = distance(br_back)
            wb = (4423*8+leng*2)*(100*100-90*90)*0.00000000785
            weight_brace +=wb
    zjq = weight_brace
    ''' 刚接连接 '''
    # for joint_indx in range(len(Joints_hor)):
    #     indx1, indx2 = Joints_hor[joint_indx]
    #     Point1 = "nodes" + str(indx1)
    #     Point2 = "nodes" + str(indx2)
    #     name = "frame_" + str(joint_indx) + '_' + str(edge_indx)
    #     ret = SapModel.FrameObj.AddByPoint(Point1, Point2, " ", joint_sec, name)
    # for i in range(len(Joints_ver)):
    #     indx1, indx2 = Joints_ver[i]
    #     Point1 = "nodes" + str(indx1)
    #     Point2 = "nodes" + str(indx2)
    #     name = "verlink_" + str(i)
    #     ret = SapModel.LinkObj.AddByPoint(Point1, Point2, "  ", joint_sec, name)
    ''' 连接单元 '''
    for j_indx in range(len(Joints_hor)):
        indx1, indx2 = Joints_hor[j_indx]
        Point1 = "nodes" + str(indx1)
        Point2 = "nodes" + str(indx2)
        name = "horlink_" + str(j_indx)
        ret = SapModel.LinkObj.AddByPoint(Point1, Point2, name, False, "HOR1")
    for j_indx in range(len(Joints_ver)):
        indx1, indx2 = Joints_ver[j_indx]
        Point1 = "nodes" + str(indx1)
        Point2 = "nodes" + str(indx2)
        name = "verlink_" + str(j_indx)
        ret = SapModel.LinkObj.AddByPoint(Point1, Point2, name, False, "VER1")
    ''' 4th adding corridor_beams '''
    for co_beam_indx in range(len(Corr_beams)):
        indx1, indx2 = Corr_beams[co_beam_indx]
        Point1 = "nodes" + str(indx1)
        Point2 = "nodes" + str(indx2)
        name = "frame_" + str(co_beam_indx) + '_' + str(edge_indx)
        section_name = "frame_section_" + str(modular_indx) + '_' + str(edge_indx)
        ret = SapModel.FrameObj.AddByPoint(Point1, Point2, " ", joint_sec, name)


    num_points = int(len(Corr_beams)/(story_num*2))
    cor_joint_data = []
    for i in range(story_num*2):
        cor_joint_floor = []
        for j in range(i*num_points,(i+1)*num_points):
            indx1=Corr_beams[j][0]
            cor_joint_floor.append(ModularBuilding.building_nodes[indx1].tolist())

        for j in range((i + 1) * num_points-1,i * num_points-1, -1):
            indx2 = Corr_beams[j][1]
            cor_joint_floor.append(ModularBuilding.building_nodes[indx2].tolist())

        cor_joint_data.append(cor_joint_floor)
    cor_joint_data=np.array(cor_joint_data)



    ret = SapModel.File.Save(ModelPath)
    # import pdb;
    # pdb.set_trace()
    # import pdb;
    # pdb.set_trace()

    """ define load """
    force_info = get_label_info()
    line_load_pattern_name = "LineLoad"
    # ret = SapModel.LoadPatterns.Add(line_load_pattern_name, 1, 0.001, True)
    area_load_pattern_name = "AreaLoad"
    # ret = SapModel.LoadPatterns.Add(area_load_pattern_name, 1, 0.001, True)
    live_load_pattern_name = "LIVE"
    ret = SapModel.LoadPatterns.Add("WINDX", 8, 0, True)
    ret = SapModel.LoadPatterns.Add("WINDY", 8, 0, True)
    ret = SapModel.LoadPatterns.Add(live_load_pattern_name, 3, 0, True)
    ret = SapModel.PropArea.SetShell_1("Plane0", 1, True, "4000Psi", 0, 0, 0)
    # ret = SapModel.PropArea.SetShellDesign("Plane0", "4000Psi", 2, 2, 3, 2.5, 3.5)
    # 添加面（顶面+底面）
    # 添加地面荷载
    for i in range(len(modulars)):
        for j in range(len(modulars[i].modular_bottom_edges)):
            node_bottom = modulars[i].modular_nodes[modulars[i].modular_bottom_edges[j]]
            node_bottem_x = np.array(node_bottom)[:, 0]
            node_bottem_y = np.array(node_bottom)[:, 1]
            node_bottem_z = np.array(node_bottom)[:, 2]
            ret = SapModel.AreaObj.AddByCoord(len(node_bottom), node_bottem_x, node_bottem_y,
                                              node_bottem_z, 'Plane0', "Default", f"plane_{modulars[i].modular_label}_{i}_{j}bottom", "Global")

            ret = SapModel.AreaObj.SetLoadUniformToFrame(f"plane_{modulars[i].modular_label}_{i}_{j}bottom", "DEAD", -0.0012, 9, 2, True, "Global")
            ret = SapModel.AreaObj.SetLoadUniformToFrame(f"plane_{modulars[i].modular_label}_{i}_{j}bottom", "LIVE", -0.0015, 9, 2, True, "Global")
        for j in range(4,8):
            ret = SapModel.FrameObj.SetLoadDistributed(f"frame_{i}_{j}", "DEAD", 1, 10, 0, 1, 4.2, 4.2)
    # 添加屋顶荷载
    # for i in range(int(5*(len(modulars))/6),len(modulars)):
    for i in range(len(modulars)):
        for j in range(len(modulars[i].modular_top_edges)):
            node_top = modulars[i].modular_nodes[modulars[i].modular_top_edges[j]]
            node_top_x = np.array(node_top)[:, 0]
            node_top_y = np.array(node_top)[:, 1]
            node_top_z = np.array(node_top)[:, 2]
            # if (i) % 26 >= 5:
            ret = SapModel.AreaObj.AddByCoord(len(modulars[i].modular_top_edges[j]), node_top_x, node_top_y, node_top_z,
                                                  'Plane0', "Default", f"plane_{modulars[i].modular_label}_{i}_{j}top", "Global")
            ret = SapModel.AreaObj.SetLoadUniformToFrame(f"plane_{modulars[i].modular_label}_{i}_{j}top", "DEAD", -0.001, 9, 2, True, "Global")
            # ret = SapModel.AreaObj.SetLoadUniformToFrame(f"plane_{modulars[i].modular_label}_{i}_{j}top", "LIVE", -force_info[1][4], 9, 2, True, "Global")

    for i in range(int(len(cor_joint_data)/2)):
        nodes_floor_cor = cor_joint_data[i]
        node_top_x = np.array(nodes_floor_cor)[:, 0]
        node_top_y = np.array(nodes_floor_cor)[:, 1]
        node_top_z = np.array(nodes_floor_cor)[:, 2]
        ret = SapModel.AreaObj.AddByCoord(num_points*2, node_top_x, node_top_y, node_top_z,
                                          'Plane0', "Default", f"plane_cor{i}top",
                                          "Global")
        ret = SapModel.AreaObj.SetLoadUniformToFrame(f"plane_cor{i}top", "DEAD", -0.0012, 9,
                                                     2, True, "Global")
        ret = SapModel.AreaObj.SetLoadUniformToFrame(f"plane_cor{i}top", "LIVE", -0.0015, 9,
                                                     2, True, "Global")
        for i in range(int(len(cor_joint_data) / 2),len(cor_joint_data)):
            nodes_floor_cor = cor_joint_data[i]
            node_top_x = np.array(nodes_floor_cor)[:, 0]
            node_top_y = np.array(nodes_floor_cor)[:, 1]
            node_top_z = np.array(nodes_floor_cor)[:, 2]
            ret = SapModel.AreaObj.AddByCoord(num_points * 2, node_top_x, node_top_y, node_top_z,
                                              'Plane0', "Default", f"plane_cor{i}bottom",
                                              "Global")
            ret = SapModel.AreaObj.SetLoadUniformToFrame(f"plane_cor{i}bottom", "DEAD", -0.001, 9,
                                                         2, True, "Global")

        ret = SapModel.AreaObj.SetLoadUniformToFrame(f"plane_{modulars[i].modular_label}_{i}_{j}top", "LIVE", -force_info[1][4], 9, 2, True, "Global")

    #添加风荷载下的围覆面
    a = modulars[0].modular_planes
    ret = SapModel.PropArea.SetShell_1("Cladding1", 1, True, "4000Psi", 0, 0, 0)

    #添加Y向风荷载
    # 一层
    Wind_Load_Y(ModularBuilding, modulars, SapModel, 1, 1, 1.3, 0, 1, 0.00055, modular_length_num)
    # 二层
    Wind_Load_Y(ModularBuilding, modulars, SapModel, 2, 1, 1.3, 1, 1.09, 0.00055, modular_length_num)
    # 三层
    Wind_Load_Y(ModularBuilding, modulars, SapModel, 3, 1, 1.3, 1.09, 1.09, 0.00055, modular_length_num)
    # 四层
    Wind_Load_Y(ModularBuilding, modulars, SapModel, 4, 1, 1.3, 1.09, 1.28, 0.00055, modular_length_num)
    # 五层
    Wind_Load_Y(ModularBuilding, modulars, SapModel, 5, 1, 1.3, 1.28, 1.42, 0.00055, modular_length_num)
    # 六层
    Wind_Load_Y(ModularBuilding, modulars, SapModel, 6, 1, 1.3, 1.42, 1.42, 0.00055, modular_length_num)

    # 添加X向风荷载
    # 一层
    Wind_Load_X(ModularBuilding, modulars, SapModel, 1, 1, 1.3, 0, 1, 0.00055, modular_length_num)
    # 二层
    Wind_Load_X(ModularBuilding, modulars, SapModel, 2, 1, 1.3, 1, 1.09, 0.00055, modular_length_num)
    # 三层
    Wind_Load_X(ModularBuilding, modulars, SapModel, 3, 1, 1.3, 1.09, 1.09, 0.00055, modular_length_num)
    # 四层
    Wind_Load_X(ModularBuilding, modulars, SapModel, 4, 1, 1.3, 1.09, 1.28, 0.00055, modular_length_num)
    # 五层
    Wind_Load_X(ModularBuilding, modulars, SapModel, 5, 1, 1.3, 1.28, 1.42, 0.00055, modular_length_num)
    # 六层
    Wind_Load_X(ModularBuilding, modulars, SapModel, 6, 1, 1.3, 1.42, 1.42, 0.00055, modular_length_num)

    ret = SapModel.LoadPatterns.Add("EX", 5)
    ret = SapModel.LoadPatterns.Add("EY", 5)
    ret = SapModel.LoadPatterns.AutoSeismic.SetChinese2002("EX", 1, 0.05, 2, 0, False, 0, 0, 0.04, 1, 0.05, 0.35, 1, 1)
    ret = SapModel.LoadPatterns.AutoSeismic.SetChinese2002("EY", 2, 0.05, 2, 0, False, 0, 0, 0.04, 1, 0.05, 0.35, 1, 1)
    EX_case_SF = [0.01]
    EY_case_SF = [0.01]
    ret = SapModel.LoadCases.StaticLinear.SetLoads("EX", 1, "Load", "EX", EX_case_SF)
    ret = SapModel.LoadCases.StaticLinear.SetLoads("EY", 1, "Load", "EY", EY_case_SF)

    """ set mass source """
    LoadPat_mass = ["DEAD", "LIVE"]
    MySF_mass = [1, 0.5]
    ret = SapModel.PropMaterial.SetMassSource(2, 2, LoadPat_mass, MySF_mass)

    #定义隔膜约束

    # 添加风荷载

    """ set load case """
    ret = SapModel.Func.FuncRS.SetChinese2010("RS-1", 0.16, 4, 0.36, 1, 0.04)
    ret = SapModel.LoadCases.ResponseSpectrum.SetCase("Quake")
    ret = SapModel.LoadCases.ResponseSpectrum.SetModalCase("Quake", "MODAL")
    ret = SapModel.LoadCases.ResponseSpectrum.SetModalComb("Quake", 2)
    MyLoadName_quake = ["U1", "U2"]
    MySF_quake = [9800, 9800]
    MyCSys_quake = ["Global","Global"]
    MyAng_quake = [10, 10]
    MyFunc_quake = ["RS-1", "RS-1"]

    ret = SapModel.LoadCases.ResponseSpectrum.SetLoads("Quake", 2, MyLoadName_quake, MyFunc_quake, MySF_quake, MyCSys_quake, MyAng_quake)
    """ define load combination """

    ret = SapModel.RespCombo.Add("COMB1", 0)
    ret = SapModel.RespCombo.SetCaseList("COMB1", 0, "DEAD", 1.3)
    ret = SapModel.RespCombo.SetCaseList("COMB1", 0, "LIVE", 1.5)
    ret = SapModel.RespCombo.SetCaseList("COMB1", 0, "WINDX", 1.0)
    ret = SapModel.RespCombo.SetCaseList("COMB1", 0, "WINDY", 1.0)
    ret = SapModel.RespCombo.SetCaseList("COMB1", 0, "EX", 1.0)
    ret = SapModel.RespCombo.SetCaseList("COMB1", 0, "EY", 1.0)
    """ define True """
    # to be added
    res1 = [True, True, True, True, True, True]
    res2 = [True, True, True, False, False, False]

    for modular_indx in range(modular_length_num*2):
        modular_edges = ModularBuilding.building_room_edges[modular_indx]
        for edge_indx in range(4,8):
            indx1, indx2 = modular_edges[edge_indx]
            Point1 = "nodes" + str(indx1)
            Point2 = "nodes" + str(indx2)
            ret = SapModel.PointObj.setRestraint(Point1, res1)
            ret = SapModel.PointObj.setRestraint(Point2, res1)

    """ run the analysis """
    # # save model
    ret = SapModel.File.Save(ModelPath)
    ret = SapModel.Analyze.RunAnalysis()

    """ results output """
    ret = SapModel.Results.Setup.DeselectAllCasesAndCombosForOutput()
    # ret = SapModel.Results.Setup.SetCaseSelectedForOutput("DEAD")

    ret = SapModel.Results.Setup.SetComboSelectedForOutput("COMB1")
    # output displacement
    weight_sap = 0.00
    mass_sap = 0.00
    # [weight_sap,mass_sap,ret] = SapModel.PropMaterial.GetWeightAndMass("Q355", weight_sap, mass_sap)

    ou_all_dis, ou_mid_dis, displacements, mid_displacements, name_frame_mid, name_all_nodes, Joint_dis = output_dis(Nodes, SapModel, modulars, ModularBuilding,modular_length_num,story_num)

    # 输出每层位移最大值
    max_dis_story = []
    for i in range(6):
        all_1_dis = []
        for j in range(modular_length_num*16*i,modular_length_num*16*(i+1)):
            all_1_dis.append(ou_all_dis[j])
        mm = max(all_1_dis)
        max_dis_story.append(mm)
        # print(f"第{i}层节点最大位移=",mm)
    max_dis_mid = []
    for i in range(6):
        mid_1_dis = []
        for j in range(modular_length_num*8*i,modular_length_num*8*(i+1)):
            mid_1_dis.append(ou_mid_dis[j])
        mm = max(mid_1_dis)
        max_dis_mid.append(mm)
        # print(f"第{i}层柱挠度最大位移=",mm)


    # import pdb;
    # pdb.set_trace()

    # frame reactions
    all_force_information = output_force(Nodes, SapModel, modulars, ModularBuilding, frame_section_info,frame_section_info000,modular_length_num,story_num)
    frame_reactions = all_force_information[0]
    name_re = all_force_information[1]
    G_max = all_force_information[2]
    G_max_beam = all_force_information[3]
    frame_reactions_all = all_force_information[4]
    all_up_fream_name = all_force_information[5]
    all_up_fream_data = all_force_information[6]
    weight_all1 = all_force_information[7] + weight_brace
    all_up_num = all_force_information[11]
    mmm = all_force_information[8]

    # output reactions infor.files
    conf1 = configparser.ConfigParser()
    for i in range(len(frame_reactions)):
        conf1.add_section(f'{name_re[i]}')
        for j in range(len(frame_reactions[i])):
            conf1.set(f'{name_re[i]}', f'{mmm[j]}', f'{list_int11(frame_reactions[i][j])}')

    # output all nodes displacements infor.files
    conf1.add_section('joints_displacements')
    for i in range(len(Nodes)):
        conf1.set('joints_displacements', f'{name_all_nodes[i]}', f'{list_int(displacements[i])}')
    conf1.add_section('mid_joints_dis')
    for i in range(len(mid_displacements)):
        conf1.set('mid_joints_dis', f'{name_frame_mid[i]}', f'{list_int(mid_displacements[i])}')

    with open('output_info.ini', 'w') as fw:
        conf1.write(fw)

    # output section infor.files
    section_info_name1 = ['material', 'type']
    section_info_name2 = ['outside_depth', 'outside_flange_width', 'flange_thickness', 'web_thickness', 's22', 's33', 'i22', 'i33']
    conf2 = configparser.ConfigParser()
    for i in range(len(name_re)):
        conf2.add_section(f'{name_re[i]}')
        for j in range(len(frame_section_info000[i])):
            conf2.set(f'{name_re[i]}', f'{section_info_name1[j]}', f'{frame_section_info000[i][j]}')
        for j in range(len(frame_section_info[i])):
            conf2.set(f'{name_re[i]}', f'{section_info_name2[j]}', f'{frame_section_info[i][j]}')

    with open('output_section_info.ini', 'w') as fw:
        conf2.write(fw)

    # output excel
    # output sections_info
    # style0 = xlwt.easyxf('font: name Times New Roman, color-index red, bold on', num_format_str='#,##0.00')
    # style1 = xlwt.easyxf(num_format_str='D-MMM-YY')

    wb = xlwt.Workbook()
    outsec = wb.add_sheet('section_info')
    loc = 0
    for i in range(len(name_re)):
        outsec.write(loc, 0, f'{name_re[i]}')
        for j in range(len(frame_section_info000[i])):
            loc += 1
            outsec.write(loc, 0, f'{section_info_name1[j]}')
            outsec.write(loc, 1, f'{frame_section_info000[i][j]}')
        for j in range(len(frame_section_info[i])):
            loc += 1
            outsec.write(loc, 0, f'{section_info_name2[j]}')
            outsec.write(loc, 1, f'{frame_section_info[i][j]}')
        loc += 1
    # out frameinfo
    outframe = wb.add_sheet('frame_reaction_info')
    loc_frame = 0
    for i in range(len(frame_reactions)):
        outframe.write(loc_frame, 0, f'{name_re[i]}')
        for j in range(len(frame_reactions[i])):
            loc_frame += 1
            outframe.write(loc_frame, 0, f'{mmm[j]}')
            outframe.write(loc_frame, 1, f'{frame_reactions[i][j][0]}')
            outframe.write(loc_frame, 2, f'{frame_reactions[i][j][1]}')
            outframe.write(loc_frame, 3, f'{frame_reactions[i][j][2]}')
        loc_frame += 1
    # out jointinfo
    outjoint = wb.add_sheet('joint_displacement_info')
    loc_joint = 0
    for i in range(len(Nodes)):
        outjoint.write(loc_joint, 0, "nodes" + str(i))
        for j in range(1, 7):
            outjoint.write(loc_joint, j, f'{displacements[i][j-1]}')
        loc_joint += 1

    #out mid_nodes_dis
    outmidnodes = wb.add_sheet('mid_nodes_displacement_info')
    loc_midnodes = 0
    for i in range(len(mid_displacements)):
        outmidnodes.write(loc_midnodes, 0, f'{name_frame_mid[i]}')
        for j in range(1, 4):
            outmidnodes.write(loc_midnodes, j, f'{mid_displacements[i][j-1]}')
        loc_midnodes += 1

    wb.save('out_info_all.xls')

    all_data = [weight_all1, G_max, G_max_beam,frame_reactions_all,frame_section_info,all_up_fream_name,all_up_fream_data,Joint_dis,all_force_information]
    return all_data

def Run_GA_sap_2(mySapObject, ModelPath, SapModel, ModularBuilding,pop_room_label,width_joint,modular_length_num,story_num):

    ret = SapModel.File.NewBlank()

    # switch units
    N_mm_C = 9
    ret = SapModel.SetPresentUnits(N_mm_C)

    """ material definition """
    # to be added
    conf = configparser.ConfigParser()
    # print(type(conf))  # conf是类
    conf.read('materials_data.ini')

    sections = conf.sections()  # 获取配置文件中所有sections，sections是列表
    # print(sections)
    option = conf.options(conf.sections()[0])

    # item = conf.items(sections[0])
    # print(item[0][1])
    material_title = []
    material_info_all = []
    pop_room_label = list(map(int, pop_room_label))
    for i in range(len(sections)):
        material_info = []
        item = conf.items(sections[i])
        ret = SapModel.PropMaterial.SetMaterial(f"{item[0][1]}", 1, -1)
        ret = SapModel.PropMaterial.SetWeightAndMass(f"{item[0][1]}", 2, float(item[1][1]))
        ret = SapModel.PropMaterial.SetMPIsotropic(f"{item[0][1]}", float(item[2][1]), float(item[3][1]), float(item[4][1]))
        ret = SapModel.PropMaterial.SetOSteel_1(f"{item[0][1]}", float(item[5][1]), float(item[6][1]), float(item[7][1]),
                                                float(item[8][1]), int(item[9][1]), int(item[10][1]), float(item[11][1]),
                                                float(item[12][1]), float(item[13][1]), float(item[14][1]))
        for i in range(14):
            material_info.append(item[i][1])
        material_info_all.append(material_info)
        material_title.append(item[0][1])

    """ cross section definition """
    '''defination from the modular perspective'''
    frame_section_name = []
    frame_section_info = []
    frame_section_info000 = []
    modular_all_section = []
    modulars = ModularBuilding.building_modulars
    # modular_nodes_indx = ModularBuilding.nodes_indx
    for modular_indx in range(len(modulars)):
        modular_i_info = modulars[modular_indx].modular_info
        modular_i_edges = ModularBuilding.building_room_edges[modular_indx]
        for edge_indx in range(len(modular_i_edges)):
            section_info = []
            frame_section_info111 = []
            section_name = "frame_section_" + str(modular_indx) + '_' + str(edge_indx)
            material_name = f"{material_title[2]}"
            section_data = modular_i_info[modulars[modular_indx].modular_edge_labels[edge_indx]]
            # if pop_room_label[modular_indx] == 1 or pop_room_label[modular_indx] == 4 or pop_room_label[modular_indx] == 7 or pop_room_label[modular_indx] == 10:
            #     section_data = modular_i_info[modulars[modular_indx].modular_edge_labels[edge_indx]]
            # elif pop_room_label[modular_indx] == 2 or pop_room_label[modular_indx] == 5 or pop_room_label[modular_indx] == 8 or pop_room_label[modular_indx] == 11:
            #     section_data = modular_i_info[modulars[modular_indx].modular_edge_labels_1[edge_indx]]
            # elif pop_room_label[modular_indx] == 3 or pop_room_label[modular_indx] == 6 or pop_room_label[modular_indx] == 9 or pop_room_label[modular_indx] == 12:
            #     section_data = modular_i_info[modulars[modular_indx].modular_edge_labels_2[edge_indx]]
            # elif pop_room_label[modular_indx] == 4 or pop_room_label[modular_indx] == 8 or pop_room_label[modular_indx] == 12 or pop_room_label[modular_indx] == 16:
            #     section_data = modular_i_info[modulars[modular_indx].modular_edge_labels_3[edge_indx]]

            if section_data['type'] == 'c0':
                ret = SapModel.PropFrame.SetChannel(section_name, material_name,
                                                    section_data['outside_depth'], section_data['outside_flange_width'],
                                                    section_data['flange_thickness'], section_data['web_thickness'], -1)
                frame_section_name.append(section_name)
                frame_section_info111.append( f"{material_title[0]}")
                frame_section_info111.append("c0")
                frame_section_info000.append(frame_section_info111)
                section_info.append(section_data['outside_depth'])
                section_info.append(section_data['outside_flange_width'])
                section_info.append(section_data['flange_thickness'])
                section_info.append(section_data['web_thickness'])
                section_info.append(section_data['s22'])
                section_info.append(section_data['s33'])
                section_info.append(section_data['i22'])
                section_info.append(section_data['i33'])
                frame_section_info.append(section_info)
            elif section_data['type'] == 'I0':
                ret = SapModel.PropFrame.SetISection(section_name, material_name,
                                                    section_data['heigth'], section_data['width'],
                                                    section_data['tf'], section_data['tw'],section_data['width'],section_data['tf'],-1)
                frame_section_name.append(section_name)
                frame_section_info111.append( f"{material_title[0]}")
                frame_section_info111.append("I0")
                frame_section_info000.append(frame_section_info111)
                section_info.append(section_data['heigth'])
                section_info.append(section_data['width'])
                section_info.append(section_data['tw'])
                section_info.append(section_data['tf'])
                section_info.append(section_data['s22'])
                section_info.append(section_data['s33'])
                section_info.append(section_data['i22'])
                section_info.append(section_data['i33'])
                frame_section_info.append(section_info)
            elif section_data['type'] == 'b0':
                ret = SapModel.PropFrame.SetTube(section_name, material_name,
                                                    section_data['outside_depth'], section_data['outside_flange_width'],
                                                    section_data['flange_thickness'], section_data['web_thickness'],-1)
                frame_section_name.append(section_name)
                frame_section_info111.append( f"{material_title[0]}")
                frame_section_info111.append("b0")
                frame_section_info000.append(frame_section_info111)
                section_info.append(section_data['outside_depth'])
                section_info.append(section_data['outside_flange_width'])
                section_info.append(section_data['flange_thickness'])
                section_info.append(section_data['web_thickness'])
                section_info.append(section_data['s22'])
                section_info.append(section_data['s33'])
                section_info.append(section_data['i22'])
                section_info.append(section_data['i33'])
                frame_section_info.append(section_info)

    """ connection definition """
    # 设置连接性质
    # 自由度
    MyDof = [True, True, True, True, True, True]
    # 固定
    MyFixed = [False, False, False, False, False, False]
    # 非线性
    MyNonLinear = [True, True, True, True, True, True]
    # 初始刚度
    MyKe = [980000, 120000, 120000, 2200000000, 3400000000, 3400000000]
    # 阻尼系数
    MyCe = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
    MyF = [-12000000, -12000000, 0, 12000000, 12000000]
    MyD = [-30, -5, 0, 5, 30]

    ret = SapModel.PropLink.SetMultiLinearElastic("HOR1", MyDof, MyFixed, MyNonLinear, MyKe, MyCe, 2, 0)
    ret = SapModel.PropLink.SetMultiLinearElastic("VER1", MyDof, MyFixed, MyNonLinear, MyKe, MyCe, 2, 0)
    # 竖直连接力位移曲线
    VERF_u1 = [-1.15e6, -1.15e6, 0, 1.07e6, 1.07e6]
    VERD_u1 = [-2.34, -1.14, 0, 1.14, 2.34]
    VERF_u23 = [-3.31e5, -3.31e5, 0, 3.31e5, 3.31e5]
    VERD_u23 = [-5.42, -2.70, 0, 2.70, 5.42]
    VERF_r1 = [-4.8e7, -4.8e7, -3.15e7, 0, 3.15e7, 4.8e7, 4.8e7]
    VERD_r1 = [-0.073, -0.038, -0.015, 0, 0.015, 0.038, 0.073]
    VERF_r23 = [-5.5e7, -5.5e7, -3.68e7, 0, 3.68e7, 5.38e7, 5.43e7]
    VERD_r23 = [-0.055, -0.027, -0.011, 0, 0.011, 0.027, 0.055]

    HORF_u1 = [-6.85e5, -6.90e5, 0, 6.90e5, 6.85e5]
    HORD_u1 = [-0.45, -0.23, 0, 0.23, 0.45]
    HORF_u23 = [-6.24e5, -6.24e5, 0, 6.30e5, 6.30e5]
    HORD_u23 = [-0.23, -0.11, 0, 0.11, 0.23]
    HORF_r1 = [-75000000, -75000000, -4.85e7, 0, 4.93e7, 75000000, 75000000]
    HORD_r1 = [-0.012, -0.0063, -0.0024, 0, 0.0024, 0.0063, 0.012]
    HORF_r2 = [-129000000, -129000000, -8.56e7, 0, 8.56e7, 1.29e8, 1.29e8]
    HORD_r2 = [-0.022, -0.011, -0.0044, 0, 0.0041, 0.011, 0.0215]
    HORF_r3 = [-3.40e7, -3.40e7, -2.31e7, 0, 2.22e7, 3.40e7, 3.40e7]
    HORD_r3 = [-0.0060, -0.0032, -0.0014, 0, 0.0012, 0.0026, 0.0053]
    # 多段连接塑性
    # ret = SapModel.PropLink.SetMultiLinearPoints("MLE1", 2, 5, MyF, MyD, 3, 9, 12, 0.75, 0.8, .1)
    # 多段连接弹性
    ret = SapModel.PropLink.SetMultiLinearPoints("HOR1", 1, 5, HORF_u1, HORD_u1)
    ret = SapModel.PropLink.SetMultiLinearPoints("HOR1", 2, 5, HORF_u23, HORD_u23)
    ret = SapModel.PropLink.SetMultiLinearPoints("HOR1", 3, 5, HORF_u23, HORD_u23)
    ret = SapModel.PropLink.SetMultiLinearPoints("HOR1", 4, 7, HORF_r1, HORD_r1)
    ret = SapModel.PropLink.SetMultiLinearPoints("HOR1", 5, 7, HORF_r2, HORD_r2)
    ret = SapModel.PropLink.SetMultiLinearPoints("HOR1", 6, 7, HORF_r3, HORD_r3)

    ret = SapModel.PropLink.SetMultiLinearPoints("VER1", 1, 5, VERF_u1, VERD_u1)
    ret = SapModel.PropLink.SetMultiLinearPoints("VER1", 2, 5, VERF_u23, VERD_u23)
    ret = SapModel.PropLink.SetMultiLinearPoints("VER1", 3, 5, VERF_u23, VERD_u23)
    ret = SapModel.PropLink.SetMultiLinearPoints("VER1", 4, 7, VERF_r1, VERD_r1)
    ret = SapModel.PropLink.SetMultiLinearPoints("VER1", 5, 7, VERF_r23, VERD_r23)
    ret = SapModel.PropLink.SetMultiLinearPoints("VER1", 6, 7, VERF_r23, VERD_r23)




    # if modular_i_info['type']

    """ define the structure """
    Nodes = ModularBuilding.building_nodes
    Joints_hor = ModularBuilding.building_room_joints_hor
    Joints_ver = ModularBuilding.building_room_joints_ver
    Corr_beams = ModularBuilding.corr_beams
    Room_indx = ModularBuilding.building_nodes_indx

    ''' 1st adding points '''
    for node_indx in range(len(Nodes)):
        x, y, z = Nodes[node_indx]
        ret = SapModel.PointObj.AddCartesian(x, y, z, None, "nodes"+str(node_indx), "Global")
    """ define frames mid points """
    for modular_indx in range(len(modulars)):
        modular_edges = ModularBuilding.building_room_edges[modular_indx]
        for edge_indx in range(len(modular_edges)):
            indx1, indx2 = modular_edges[edge_indx]
            # Point1 = "nodes" + str(indx1)
            # Point2 = "nodes" + str(indx2)
            x1, y1, z1 = Nodes[indx1]
            x2, y2, z2 = Nodes[indx2]
            x3 = 0.5 * (x1 + x2)
            y3 = 0.5 * (y1 + y2)
            z3 = 0.5 * (z1 + z2)
            ret = SapModel.PointObj.AddCartesian(x3, y3, z3, None, "nodes_mid" + str(modular_indx) + '_' + str(edge_indx), "Global")
    """define frame 3 point"""
    for modular_indx in range(len(modulars)):
        modular_edges = ModularBuilding.building_room_edges[modular_indx]
        for edge_indx in [5, 7, 9, 11]:
            indx1, indx2 = modular_edges[edge_indx]
            # Point1 = "nodes" + str(indx1)
            # Point2 = "nodes" + str(indx2)
            x1, y1, z1 = Nodes[indx1]
            x2, y2, z2 = Nodes[indx2]
            x3 = 0.5 * (x1 + x2)
            y3 = 0.5 * (y1 + y2) -750
            z3 = 0.5 * (z1 + z2)
            y4 = 0.5 * (y1 + y2) +750
            ret = SapModel.PointObj.AddCartesian(x3, y3, z3, None, "nodes_brace" + str(modular_indx) + '_' + str(edge_indx) + '_0', "Global")
            ret = SapModel.PointObj.AddCartesian(x3, y4, z3, None, "nodes_brace" + str(modular_indx) + '_' + str(edge_indx) + '_1', "Global")
    ''' 2nd adding frames '''
    for modular_indx in range(len(modulars)):
        modular_edges = ModularBuilding.building_room_edges[modular_indx]
        for edge_indx in range(len(modular_edges)):
            indx1, indx2 = modular_edges[edge_indx]
            Point1 = "nodes" + str(indx1)
            Point2 = "nodes" + str(indx2)
            name = "frame_" + str(modular_indx) + '_' + str(edge_indx)
            section_name = "frame_section_" + str(modular_indx) + '_' + str(edge_indx)
            ret = SapModel.FrameObj.AddByPoint(Point1, Point2, " ", section_name, name)
    ''' 3rd adding joints '''
    joint_sec, joint_mater = "Rectang", f"{material_title[2]}"
    ret = SapModel.PropFrame.SetRectangle(joint_sec, joint_mater,width_joint,width_joint,-1)
    brace_sec, brace_mater = "Rectang", f"{material_title[2]}"
    ret = SapModel.PropFrame.SetTube(brace_sec, brace_mater, 100, 100,10,10, -1)
    ''' 4rd adding braces '''
    weight_brace = 0.0
    for i in range(len(modulars)):
        #人字支撑
        if pop_room_label[i] == 1:
            nodes_indx = ModularBuilding.building_nodes_indx[i]
            ret = SapModel.FrameObj.AddByPoint("nodes"+str(nodes_indx[0]), "nodes_mid" + str(i) + '_' + str(11), " ", brace_sec, "brace_" + str(i) + '_' + str(0))
            ret = SapModel.FrameObj.AddByPoint("nodes" + str(nodes_indx[6]), "nodes_mid" + str(i) + '_' + str(11), " ",
                                               brace_sec, "brace_" + str(i) + '_' + str(1))
            ret = SapModel.FrameObj.AddByPoint("nodes" + str(nodes_indx[2]), "nodes_mid" + str(i) + '_' + str(9), " ",
                                               brace_sec, "brace_" + str(i) + '_' + str(2))
            ret = SapModel.FrameObj.AddByPoint("nodes" + str(nodes_indx[4]), "nodes_mid" + str(i) + '_' + str(9), " ",
                                               brace_sec, "brace_" + str(i) + '_' + str(3))
            ret = SapModel.FrameObj.AddByPoint("nodes" + str(nodes_indx[4]), "nodes" + str(nodes_indx[7]), " ",
                                               brace_sec, "brace_" + str(i) + '_' + str(4))
            ret = SapModel.FrameObj.AddByPoint("nodes" + str(nodes_indx[5]), "nodes" + str(nodes_indx[6]), " ",
                                               brace_sec, "brace_" + str(i) + '_' + str(5))
            br_back = [Nodes[nodes_indx[5]][0]-Nodes[nodes_indx[6]][0],Nodes[nodes_indx[5]][1]-Nodes[nodes_indx[6]][1],Nodes[nodes_indx[5]][2]-Nodes[nodes_indx[6]][2]]
            leng = distance(br_back)
            wb = (5000*4+leng*2)*(100*100-90*90)*0.00000000785
            weight_brace +=wb

        # 交叉支撑
        elif pop_room_label[i] == 2 :
            nodes_indx = ModularBuilding.building_nodes_indx[i]
            ret = SapModel.FrameObj.AddByPoint("nodes" + str(nodes_indx[0]), "nodes" + str(nodes_indx[7]), " ",
                                               brace_sec, "brace_" + str(i) + '_' + str(0))
            ret = SapModel.FrameObj.AddByPoint("nodes" + str(nodes_indx[6]), "nodes" + str(nodes_indx[1]), " ",
                                               brace_sec, "brace_" + str(i) + '_' + str(1))
            ret = SapModel.FrameObj.AddByPoint("nodes" + str(nodes_indx[2]), "nodes" + str(nodes_indx[5]), " ",
                                               brace_sec, "brace_" + str(i) + '_' + str(2))
            ret = SapModel.FrameObj.AddByPoint("nodes" + str(nodes_indx[3]), "nodes" + str(nodes_indx[4]), " ",
                                               brace_sec, "brace_" + str(i) + '_' + str(3))
            ret = SapModel.FrameObj.AddByPoint("nodes" + str(nodes_indx[4]), "nodes" + str(nodes_indx[7]), " ",
                                               brace_sec, "brace_" + str(i) + '_' + str(4))
            ret = SapModel.FrameObj.AddByPoint("nodes" + str(nodes_indx[5]), "nodes" + str(nodes_indx[6]), " ",
                                               brace_sec, "brace_" + str(i) + '_' + str(5))

            br_back = [Nodes[nodes_indx[5]][0] - Nodes[nodes_indx[6]][0], Nodes[nodes_indx[5]][1] - Nodes[nodes_indx[6]][1],
                       Nodes[nodes_indx[5]][2] - Nodes[nodes_indx[6]][2]]
            leng = distance(br_back)
            wb = (8544 * 4 + leng * 2) *(100*100-90*90)* 0.00000000785
            weight_brace += wb
        #双交叉支撑
        elif pop_room_label[i] == 3:
            nodes_indx = ModularBuilding.building_nodes_indx[i]
            ret = SapModel.FrameObj.AddByPoint("nodes" + str(nodes_indx[0]), "nodes_brace" + str(i) + '_' + str(11) + '_0', " ",
                                               brace_sec, "brace_" + str(i) + '_' + str(0))
            ret = SapModel.FrameObj.AddByPoint("nodes" + str(nodes_indx[1]), "nodes_brace" + str(i) + '_' + str(7) + '_0', " ",
                                               brace_sec, "brace_" + str(i) + '_' + str(1))
            ret = SapModel.FrameObj.AddByPoint("nodes" + str(nodes_indx[6]), "nodes_brace" + str(i) + '_' + str(11) + '_1', " ",
                                               brace_sec, "brace_" + str(i) + '_' + str(2))
            ret = SapModel.FrameObj.AddByPoint("nodes" + str(nodes_indx[7]), "nodes_brace" + str(i) + '_' + str(7) + '_1', " ",
                                               brace_sec, "brace_" + str(i) + '_' + str(3))
            ret = SapModel.FrameObj.AddByPoint("nodes" + str(nodes_indx[2]), "nodes_brace" + str(i) + '_' + str(9) + '_0', " ",
                                               brace_sec, "brace_" + str(i) + '_' + str(4))
            ret = SapModel.FrameObj.AddByPoint("nodes" + str(nodes_indx[3]), "nodes_brace" + str(i) + '_' + str(5) + '_0', " ",
                                               brace_sec, "brace_" + str(i) + '_' + str(5))
            ret = SapModel.FrameObj.AddByPoint("nodes" + str(nodes_indx[4]), "nodes_brace" + str(i) + '_' + str(9) + '_1', " ",
                                               brace_sec, "brace_" + str(i) + '_' + str(6))
            ret = SapModel.FrameObj.AddByPoint("nodes" + str(nodes_indx[5]), "nodes_brace" + str(i) + '_' + str(5) + '_1', " ",
                                               brace_sec, "brace_" + str(i) + '_' + str(7))
            ret = SapModel.FrameObj.AddByPoint("nodes" + str(nodes_indx[4]), "nodes" + str(nodes_indx[7]), " ",
                                               brace_sec, "brace_" + str(i) + '_' + str(8))
            ret = SapModel.FrameObj.AddByPoint("nodes" + str(nodes_indx[5]), "nodes" + str(nodes_indx[6]), " ",
                                               brace_sec, "brace_" + str(i) + '_' + str(9))

            br_back = [Nodes[nodes_indx[5]][0]-Nodes[nodes_indx[6]][0],Nodes[nodes_indx[5]][1]-Nodes[nodes_indx[6]][1],Nodes[nodes_indx[5]][2]-Nodes[nodes_indx[6]][2]]
            leng = distance(br_back)
            wb = (4423*8+leng*2)*(100*100-90*90)*0.00000000785
            weight_brace +=wb
    zjq = weight_brace
    ''' 刚接连接 '''
    # for joint_indx in range(len(Joints_hor)):
    #     indx1, indx2 = Joints_hor[joint_indx]
    #     Point1 = "nodes" + str(indx1)
    #     Point2 = "nodes" + str(indx2)
    #     name = "frame_" + str(joint_indx) + '_' + str(edge_indx)
    #     ret = SapModel.FrameObj.AddByPoint(Point1, Point2, " ", joint_sec, name)
    # for i in range(len(Joints_ver)):
    #     indx1, indx2 = Joints_ver[i]
    #     Point1 = "nodes" + str(indx1)
    #     Point2 = "nodes" + str(indx2)
    #     name = "verlink_" + str(i)
    #     ret = SapModel.LinkObj.AddByPoint(Point1, Point2, "  ", joint_sec, name)
    ''' 连接单元 '''
    for j_indx in range(len(Joints_hor)):
        indx1, indx2 = Joints_hor[j_indx]
        Point1 = "nodes" + str(indx1)
        Point2 = "nodes" + str(indx2)
        name = "horlink_" + str(j_indx)
        ret = SapModel.LinkObj.AddByPoint(Point1, Point2, name, False, "HOR1")
    for j_indx in range(len(Joints_ver)):
        indx1, indx2 = Joints_ver[j_indx]
        Point1 = "nodes" + str(indx1)
        Point2 = "nodes" + str(indx2)
        name = "verlink_" + str(j_indx)
        ret = SapModel.LinkObj.AddByPoint(Point1, Point2, name, False, "VER1")
    ''' 4th adding corridor_beams '''
    for co_beam_indx in range(len(Corr_beams)):
        indx1, indx2 = Corr_beams[co_beam_indx]
        Point1 = "nodes" + str(indx1)
        Point2 = "nodes" + str(indx2)
        name = "frame_" + str(co_beam_indx) + '_' + str(edge_indx)
        section_name = "frame_section_" + str(modular_indx) + '_' + str(edge_indx)
        ret = SapModel.FrameObj.AddByPoint(Point1, Point2, " ", joint_sec, name)

    # node_change = copy.deepcopy(Nodes)
    # num_points = int(len(Corr_beams)/(story_num*2))
    # cor_joint_data = []
    # for i in range(story_num*2):
    #     cor_joint_floor = []
    #     for j in range(i*num_points,(i+1)*num_points):
    #         indx1=Corr_beams[j][0]
    #         cor_joint_floor.append(node_change[indx1].tolist())
    #
    #     for j in range((i + 1) * num_points-1,i * num_points-1, -1):
    #         indx2 = Corr_beams[j][1]
    #         cor_joint_floor.append(node_change[indx2].tolist())
    #
    #     cor_joint_data.append(cor_joint_floor)
    # cor_joint_data=np.array(cor_joint_data)

    node_change = copy.deepcopy(Nodes)
    num_points = int(len(Corr_beams)/(story_num*2))
    cor_joint_data = []
    for i in range(story_num*2):
        cor_joint_floor = []
        for j in range(i*num_points,(i+1)*num_points):
            indx1=Corr_beams[j][0]
            cor_joint_floor.append('nodes' + str(indx1))

        for j in range((i + 1) * num_points-1,i * num_points-1, -1):
            indx2 = Corr_beams[j][1]
            cor_joint_floor.append('nodes' + str(indx2))

        cor_joint_data.append(cor_joint_floor)
    # cor_joint_data=np.array(cor_joint_data)




    ret = SapModel.File.Save(ModelPath)


    # import pdb;
    # pdb.set_trace()
    # import pdb;
    # pdb.set_trace()

    """ define load """
    force_info = get_label_info()
    line_load_pattern_name = "LineLoad"
    # ret = SapModel.LoadPatterns.Add(line_load_pattern_name, 1, 0.001, True)
    area_load_pattern_name = "AreaLoad"
    # ret = SapModel.LoadPatterns.Add(area_load_pattern_name, 1, 0.001, True)
    live_load_pattern_name = "LIVE"
    ret = SapModel.LoadPatterns.Add("WINDX", 8, 0, True)
    ret = SapModel.LoadPatterns.Add("WINDY", 8, 0, True)
    ret = SapModel.LoadPatterns.Add(live_load_pattern_name, 3, 0, True)
    ret = SapModel.PropArea.SetShell_1("Plane0", 1, True, "4000Psi", 0, 0, 0)
    # ret = SapModel.PropArea.SetShellDesign("Plane0", "4000Psi", 2, 2, 3, 2.5, 3.5)
    # 添加面（顶面+底面）
    # 添加地面荷载
    for i in range(len(modulars)):
        for j in range(len(modulars[i].modular_bottom_edges)):
            node_bottom = modulars[i].modular_nodes[modulars[i].modular_bottom_edges[j]]
            node_bottem_x = np.array(node_bottom)[:, 0]
            node_bottem_y = np.array(node_bottom)[:, 1]
            node_bottem_z = np.array(node_bottom)[:, 2]
            ret = SapModel.AreaObj.AddByCoord(len(node_bottom), node_bottem_x, node_bottem_y,
                                              node_bottem_z, 'Plane0', "Default", f"plane_{modulars[i].modular_label}_{i}_{j}bottom", "Global")

            ret = SapModel.AreaObj.SetLoadUniformToFrame(f"plane_{modulars[i].modular_label}_{i}_{j}bottom", "DEAD", -0.0012, 9, 2, True, "Global")
            ret = SapModel.AreaObj.SetLoadUniformToFrame(f"plane_{modulars[i].modular_label}_{i}_{j}bottom", "LIVE", -0.0015, 9, 2, True, "Global")
        for j in range(4,8):
            ret = SapModel.FrameObj.SetLoadDistributed(f"frame_{i}_{j}", "DEAD", 1, 10, 0, 1, 4.2, 4.2)
    # 添加屋顶荷载
    # for i in range(int(5*(len(modulars))/6),len(modulars)):
    for i in range(len(modulars)):
        for j in range(len(modulars[i].modular_top_edges)):
            node_top = modulars[i].modular_nodes[modulars[i].modular_top_edges[j]]
            node_top_x = np.array(node_top)[:, 0]
            node_top_y = np.array(node_top)[:, 1]
            node_top_z = np.array(node_top)[:, 2]
            # if (i) % 26 >= 5:
            ret = SapModel.AreaObj.AddByCoord(len(modulars[i].modular_top_edges[j]), node_top_x, node_top_y, node_top_z,
                                                  'Plane0', "Default", f"plane_{modulars[i].modular_label}_{i}_{j}top", "Global")
            ret = SapModel.AreaObj.SetLoadUniformToFrame(f"plane_{modulars[i].modular_label}_{i}_{j}top", "DEAD", -0.001, 9, 2, True, "Global")
            # ret = SapModel.AreaObj.SetLoadUniformToFrame(f"plane_{modulars[i].modular_label}_{i}_{j}top", "LIVE", -force_info[1][4], 9, 2, True, "Global")

    # for i in range(int(len(cor_joint_data)/2)):
    #     nodes_floor_cor = cor_joint_data[i]
    #     node_top_x = np.array(nodes_floor_cor)[:, 0]
    #     node_top_y = np.array(nodes_floor_cor)[:, 1]
    #     node_top_z = np.array(nodes_floor_cor)[:, 2]
    #     ret = SapModel.AreaObj.AddByCoord(num_points*2, node_top_x, node_top_y, node_top_z,
    #                                       'Plane0', "Default", f"plane_corrid{i}top",
    #                                       "Global")
    #     ret = SapModel.AreaObj.SetLoadUniformToFrame(f"plane_corrid{i}top", "DEAD", -0.0012, 9,
    #                                                  2, True, "Global")
    #     ret = SapModel.AreaObj.SetLoadUniformToFrame(f"plane_corrid{i}top", "LIVE", -0.0015, 9,
    #                                                  2, True, "Global")

    # Point = ['nodes0','nodes3','nodes7','nodes4']
    for i in range(int(len(cor_joint_data)/2)):
        ret = SapModel.AreaObj.AddByPoint(len(cor_joint_data[0]), cor_joint_data[0], 'Plane0', "Default",  f"plane_corrid{0}bottom")
        ret = SapModel.AreaObj.SetLoadUniformToFrame(f"plane_corrid{i}bottom", "DEAD", -0.0012, 9,2, True, "Global")
        ret = SapModel.AreaObj.SetLoadUniformToFrame(f"plane_corrid{i}bottom", "LIVE", -0.0015, 9,2, True, "Global")

    for i in range(int(len(cor_joint_data) / 2), len(cor_joint_data)):
        ret = SapModel.AreaObj.AddByPoint(len(cor_joint_data[i]), cor_joint_data[i], 'Plane0', "Default",  f"plane_corrid{i}top")
        ret = SapModel.AreaObj.SetLoadUniformToFrame(f"plane_corrid{i}top", "DEAD", -0.001, 9,2, True, "Global")

    # for i in range(int(len(cor_joint_data) / 2),len(cor_joint_data)):
    #     nodes_floor_cor = cor_joint_data[i]
    #     node_top_x = np.array(nodes_floor_cor)[:, 0]
    #     node_top_y = np.array(nodes_floor_cor)[:, 1]
    #     node_top_z = np.array(nodes_floor_cor)[:, 2]
    #     ret = SapModel.AreaObj.AddByCoord(num_points * 2, node_top_x, node_top_y, node_top_z,
    #                                       'Plane0', "Default", f"plane_corrid{i}bottom",
    #                                       "Global")
    #     ret = SapModel.AreaObj.SetLoadUniformToFrame(f"plane_corrid{i}bottom", "DEAD", -0.001, 9,
    #                                                  2, True, "Global")



    #添加风荷载下的围覆面
    a = modulars[0].modular_planes
    ret = SapModel.PropArea.SetShell_1("Cladding1", 1, True, "4000Psi", 0, 0, 0)

    #添加Y向风荷载
    # 一层
    Wind_Load_Y(ModularBuilding, modulars, SapModel, 1, 1, 1.3, 0, 1, 0.00055, modular_length_num)
    # 二层
    Wind_Load_Y(ModularBuilding, modulars, SapModel, 2, 1, 1.3, 1, 1.09, 0.00055, modular_length_num)
    # 三层
    Wind_Load_Y(ModularBuilding, modulars, SapModel, 3, 1, 1.3, 1.09, 1.09, 0.00055, modular_length_num)
    # 四层
    Wind_Load_Y(ModularBuilding, modulars, SapModel, 4, 1, 1.3, 1.09, 1.28, 0.00055, modular_length_num)
    # 五层
    Wind_Load_Y(ModularBuilding, modulars, SapModel, 5, 1, 1.3, 1.28, 1.42, 0.00055, modular_length_num)
    # 六层
    Wind_Load_Y(ModularBuilding, modulars, SapModel, 6, 1, 1.3, 1.42, 1.42, 0.00055, modular_length_num)

    # 添加X向风荷载
    # 一层
    Wind_Load_X(ModularBuilding, modulars, SapModel, 1, 1, 1.3, 0, 1, 0.00055, modular_length_num)
    # 二层
    Wind_Load_X(ModularBuilding, modulars, SapModel, 2, 1, 1.3, 1, 1.09, 0.00055, modular_length_num)
    # 三层
    Wind_Load_X(ModularBuilding, modulars, SapModel, 3, 1, 1.3, 1.09, 1.09, 0.00055, modular_length_num)
    # 四层
    Wind_Load_X(ModularBuilding, modulars, SapModel, 4, 1, 1.3, 1.09, 1.28, 0.00055, modular_length_num)
    # 五层
    Wind_Load_X(ModularBuilding, modulars, SapModel, 5, 1, 1.3, 1.28, 1.42, 0.00055, modular_length_num)
    # 六层
    Wind_Load_X(ModularBuilding, modulars, SapModel, 6, 1, 1.3, 1.42, 1.42, 0.00055, modular_length_num)

    ret = SapModel.LoadPatterns.Add("EX", 5)
    ret = SapModel.LoadPatterns.Add("EY", 5)
    ret = SapModel.LoadPatterns.AutoSeismic.SetChinese2002("EX", 1, 0.05, 2, 0, False, 0, 0, 0.04, 1, 0.05, 0.35, 1, 1)
    ret = SapModel.LoadPatterns.AutoSeismic.SetChinese2002("EY", 2, 0.05, 2, 0, False, 0, 0, 0.04, 1, 0.05, 0.35, 1, 1)
    EX_case_SF = [0.01]
    EY_case_SF = [0.01]
    ret = SapModel.LoadCases.StaticLinear.SetLoads("EX", 1, "Load", "EX", EX_case_SF)
    ret = SapModel.LoadCases.StaticLinear.SetLoads("EY", 1, "Load", "EY", EY_case_SF)

    """ set mass source """
    LoadPat_mass = ["DEAD", "LIVE"]
    MySF_mass = [1, 0.5]
    ret = SapModel.PropMaterial.SetMassSource(2, 2, LoadPat_mass, MySF_mass)

    #定义隔膜约束
    ret = SapModel.ConstraintDef.SetDiaphragm("Diaph1",3)

    # 添加风荷载

    """ set load case """
    # wind
    # ret = SapModel.LoadCases.StaticLinear.SetCase("Wind")
    # MyLoadType = ["Load", "Load"]
    # MyLoadName = ["WX", "WY"]
    # MySF = [1,1]
    # ret = SapModel.LoadCases.StaticLinear.SetLoads("Wind", 2, MyLoadType, MyLoadName, MySF)
    # Quake
    # ResponseSpectrum
    ret = SapModel.Func.FuncRS.SetChinese2010("RS-1", 0.16, 4, 0.36, 1, 0.04)

    ret = SapModel.LoadCases.ResponseSpectrum.SetCase("Quake")
    ret = SapModel.LoadCases.ResponseSpectrum.SetModalCase("Quake", "MODAL")
    ret = SapModel.LoadCases.ResponseSpectrum.SetModalComb("Quake", 2)
    MyLoadName_quake = ["U1", "U2"]
    MySF_quake = [9800, 9800]
    MyCSys_quake = ["Global","Global"]
    MyAng_quake = [10, 10]
    MyFunc_quake = ["RS-1", "RS-1"]

    ret = SapModel.LoadCases.ResponseSpectrum.SetLoads("Quake", 2, MyLoadName_quake, MyFunc_quake, MySF_quake, MyCSys_quake, MyAng_quake)
    """ define load combination """

    ret = SapModel.RespCombo.Add("COMB1", 0)
    ret = SapModel.RespCombo.SetCaseList("COMB1", 0, "DEAD", 1.3)
    ret = SapModel.RespCombo.SetCaseList("COMB1", 0, "LIVE", 1.5)
    ret = SapModel.RespCombo.SetCaseList("COMB1", 0, "WINDX", 1.0)
    ret = SapModel.RespCombo.SetCaseList("COMB1", 0, "WINDY", 1.0)
    ret = SapModel.RespCombo.SetCaseList("COMB1", 0, "EX", 1.0)
    ret = SapModel.RespCombo.SetCaseList("COMB1", 0, "EY", 1.0)
    """ define True """
    # to be added
    res1 = [True, True, True, True, True, True]
    res2 = [True, True, True, False, False, False]

    for modular_indx in range(modular_length_num*2):
        modular_edges = ModularBuilding.building_room_edges[modular_indx]
        for edge_indx in range(4,8):
            indx1, indx2 = modular_edges[edge_indx]
            Point1 = "nodes" + str(indx1)
            Point2 = "nodes" + str(indx2)
            ret = SapModel.PointObj.setRestraint(Point1, res1)
            ret = SapModel.PointObj.setRestraint(Point2, res1)

    """ run the analysis """
    # # save model
    ret = SapModel.File.Save(ModelPath)
    ret = SapModel.Analyze.RunAnalysis()

    """ results output """
    ret = SapModel.Results.Setup.DeselectAllCasesAndCombosForOutput()
    # ret = SapModel.Results.Setup.SetCaseSelectedForOutput("DEAD")

    ret = SapModel.Results.Setup.SetComboSelectedForOutput("COMB1")
    # output displacement
    weight_sap = 0.00
    mass_sap = 0.00
    # [weight_sap,mass_sap,ret] = SapModel.PropMaterial.GetWeightAndMass("Q355", weight_sap, mass_sap)

    ou_all_dis, ou_mid_dis, displacements, mid_displacements, name_frame_mid, name_all_nodes, Joint_dis = output_dis(Nodes, SapModel, modulars, ModularBuilding,modular_length_num,story_num)

    # 输出每层位移最大值
    max_dis_story = []
    for i in range(6):
        all_1_dis = []
        for j in range(modular_length_num*16*i,modular_length_num*16*(i+1)):
            all_1_dis.append(ou_all_dis[j])
        mm = max(all_1_dis)
        max_dis_story.append(mm)
        # print(f"第{i}层节点最大位移=",mm)
    max_dis_mid = []
    for i in range(6):
        mid_1_dis = []
        for j in range(modular_length_num*8*i,modular_length_num*8*(i+1)):
            mid_1_dis.append(ou_mid_dis[j])
        mm = max(mid_1_dis)
        max_dis_mid.append(mm)
        # print(f"第{i}层柱挠度最大位移=",mm)


    # import pdb;
    # pdb.set_trace()

    # frame reactions
    all_force_information = output_force(Nodes, SapModel, modulars, ModularBuilding, frame_section_info,frame_section_info000,modular_length_num,story_num)
    frame_reactions = all_force_information[0]
    name_re = all_force_information[1]
    G_max = all_force_information[2]
    G_max_beam = all_force_information[3]
    frame_reactions_all = all_force_information[4]
    all_up_fream_name = all_force_information[5]
    all_up_fream_data = all_force_information[6]
    weight_all1 = all_force_information[7] + weight_brace
    all_up_num = all_force_information[11]
    mmm = all_force_information[8]

    all_data = [weight_all1, G_max, G_max_beam,frame_reactions_all,frame_section_info,all_up_fream_name,all_up_fream_data,Joint_dis,all_force_information]
    return all_data

# modular_length = 8000
# modular_width = [4000,4000,5400,3600,3600,4400,4400,4000]
# modular_heigth = 4000
# modular_length_num = 8
# modular_dis = 400
# story_num = 6
# corridor_width = 4000
# model_data = dj.generate_model_data(modular_length,modular_width,modular_heigth,modular_length_num,modular_dis,story_num,corridor_width)
# nodes = model_data[0]
# edges_all = model_data[1]
# labels = model_data[2]
# cor_edges = model_data[3]
# joint_hor = model_data[4]
# joint_ver = model_data[5]
# room_indx = model_data[6]
#
# sections_data_c1, type_keys_c1, sections_c1 = get_section_info(section_type='c0',
#                                                                   cfg_file_name="Steel_section_data33.ini")
#
# APIPath = os.path.join(os.getcwd(), 'cases')
# mySapObject, ModelPath, SapModel = SAPanalysis_GA_run(APIPath)
# modular_building = md.ModularBuilding(nodes, room_indx, edges_all, labels, joint_hor, joint_ver, cor_edges)
# modulars_of_building = modular_building.building_modulars
# modular_nums = len(labels)
# modular_infos = {}
# for i in range(modular_nums):
#     modular_infos[i] = Modular_Info_Initialization(type='regular', top_edge=sections_data_c1[0],
#                                                       bottom_edge=sections_data_c1[8],
#                                                       column_edge=sections_data_c1[17])
# for i in range(len(modulars_of_building)):
#     modulars_of_building[i].Add_Info_And_Update_Modular(
#         modular_infos[modulars_of_building[i].modular_label - 1])
# modular_building.Building_Assembling(modulars_of_building)
#
#
# all_data = Run_GA_sap(mySapObject, ModelPath, SapModel, modular_building,200,modular_length_num,story_num)



