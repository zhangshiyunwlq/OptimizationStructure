import copy
import numpy as np
import pyvista as pv
import data_to_json as dj
import modular_utils as md
import model_to_sap as ms
import os
import sap_run as sr
import sys
import comtypes.client
import win32com.client
import xlwt
import math as m
import configparser
import time
import random
from random import randint
import xlwt
import xlrd
import gc
import openpyxl
import threading
import queue
import threading
import queue


def mulit_Sap_analy_allroom(ModelPath,mySapObject, SapModel,pop_room,pop_room_label):

    sections_data_c1, type_keys_c1, sections_c1 = ms.get_section_info(section_type='c0',
                                                                      cfg_file_name="Steel_section_data.ini")
    modular_building = md.ModularBuilding(nodes, room_indx, edges_all, labels, joint_hor, joint_ver, cor_edges)
    # 按房间分好节点
    modulars_of_building = modular_building.building_modulars
    modular_nums = len(labels)
    modular_infos = {}
    # 每个房间定义梁柱截面信息
    sr.run_column_room_story5(labels,pop_room_label, modular_length_num * 2 * story_num, sections_data_c1, modular_infos, pop_room)
    #
    for i in range(len(modulars_of_building)):
        modulars_of_building[i].Add_Info_And_Update_Modular(
            # modular_infos[modulars_of_building[i].modular_label - 1])
            modular_infos[i])
    modular_building.Building_Assembling(modulars_of_building)

    all_data = ms.Run_GA_sap_2(mySapObject, ModelPath, SapModel, modular_building,pop_room_label,200,modular_length_num,story_num)
    aa, bb, cc, dd, ee, ff, gg, hh, ii = all_data

    ret = SapModel.SetModelIsLocked(False)
    return aa,bb,cc,dd,ee,ff,gg,hh,ii

def SAPanalysis_GA_run2(APIPath):


    cfg = configparser.ConfigParser()
    cfg.read("Configuration.ini", encoding='utf-8')
    ProgramPath = cfg['SAP2000PATH']['dirpath']
    if not os.path.exists(APIPath):
        try:
            os.makedirs(APIPath)
        except OSError:
            pass

    AttachToInstance = False
    SpecifyPath = True

    # ModelPath = os.path.join(APIPath, 'API_1-001.sdb')
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
    ModelPath = os.path.join(APIPath, 'API_1-001.sdb')
    # create new blank model
    return mySapObject,ModelPath, SapModel

def mulit_GA(ModelPath,mySapObject, SapModel,pop1,pop3):
    result = []
    weight_1 = []
    col_up = []
    beam_up = []
    for time in range(len(pop1)):
        pop = pop1[time]
        pop_room_label = pop3[time]
        # pop_all.append(pop)
        we,co,be,r1,r2,r3,r4,dis_all,force_all =mulit_Sap_analy_allroom(ModelPath,mySapObject, SapModel,pop,pop_room_label)
        res1,res2 = Fun_1(we, co, be,dis_all,force_all, 10000)
        # num3 += 1
        weight_1.append(res2)
        col_up.append(co)
        beam_up.append(be)
        result.append(res1)
    return result,weight_1,col_up,beam_up

def Fun_1(weight,g_col,g_beam,dis_all,all_force,u):
    g_col_all = 0
    g_beam_all = 0
    Y_dis_radio_all = 0
    Y_interdis_all = 0
    Y_interdis_radio_all = 0
    floor_radio = 0
    for i in range(len(g_col)):
        if g_col[i]<= 0:
            g_col[i] = 0
        else:
            g_col[i] = g_col[i]
        g_col_all += g_col[i]
    for i in range(len(g_beam)):
        if g_beam[i]<= 0:
            g_beam[i] = 0
        else:
            g_beam[i] = g_beam[i]
        g_beam_all += g_beam[i]
    #y dis ratio
    for i in range(len(dis_all[5])):
        if dis_all[5][i] <= 1.5 and dis_all[5][i] >= -1.5:
            dis_all[5][i] = 0
        else:
            dis_all[5][i] = dis_all[5][i]
        Y_dis_radio_all += dis_all[5][i]
    # y interdis max
    for i in range(len(dis_all[7])):
        if dis_all[7][i] <= 0.004 and dis_all[7][i] >= -0.004:
            dis_all[7][i] = 0
        else:
            dis_all[7][i] = dis_all[7][i]
        Y_interdis_all += dis_all[7][i]
    # y interdis radio
    for i in range(len(dis_all[11])):
        if dis_all[11][i] <= 1.5 and dis_all[11][i] >= -1.5:
            dis_all[11][i] = 0
        else:
            dis_all[11][i] = dis_all[11][i]
        Y_interdis_radio_all += dis_all[11][i]
    # x interdis ratio
    for i in range(len(all_force[10])):
        if all_force[10][i] <= 1.5 and all_force[10][i] >= -1.5:
            all_force[10][i] = 0
        else:
            all_force[10][i] = all_force[10][i]
        floor_radio += all_force[10][i]

    G_value=u * (abs(g_col_all) + abs(g_beam_all) + abs(Y_dis_radio_all) + abs(Y_interdis_all) + abs(Y_interdis_radio_all))
    result = weight + G_value

    return result,weight

def mulitrun_GA(ModelPath,mySapObject, SapModel,pop1,pop_all,pop3,q,result,weight_1,col_up,beam_up,memory_pools_all,memory_pools_fit,memory_pools_weight,memory_pools_col,memory_pools_beam):
    while True:
        if q.empty():
            break

        time = q.get()
        pop = pop1[time]
        pop_room_label = pop3[time]
        pop2= pop_all[time]
        # 判断记忆池
        for j in range(len(memory_pools_all)):
            tf = []
            for z in range(len(pop2)):
                if pop2[z] != memory_pools_all[j][z]:
                    tf.append(z)
            if len(tf) == 0:
                break
        # 记忆池中存在满足条件的个体
        if len(tf) == 0:
            result[time] = memory_pools_fit[j]
            weight_1[time] = memory_pools_weight[j]
            col_up[time] = memory_pools_col[j]
            beam_up[time] = memory_pools_beam[j]
        # 记忆池中不存在满足条件的个体
        else:
            # pop_all.append(pop)
            we, co, be, r1, r2, r3, r4, dis_all, force_all = mulit_Sap_analy_allroom(ModelPath, mySapObject, SapModel, pop,
                                                                                     pop_room_label)
            res1, res2 = Fun_1(we, co, be, dis_all, force_all, 10000)
            # num3 += 1
            weight_1[time] = res2
            col_up[time] = co
            beam_up[time] = be
            result[time] = res1
        # 记忆池更新
            memory_pools_all.extend(pop2)
            memory_pools_fit.extend(res1)
            memory_pools_weight.extend(res2)
            memory_pools_col.extend(co)
            memory_pools_beam.extend(be)

def thread_sap(num,pop1,pop2,pop3,result,weight_1,col_up,beam_up,memory_pools_all,memory_pools_fit,memory_pools_weight,memory_pools_col,memory_pools_beam):
    case_name = []
    APIPath_name = []
    mySapObject_name = []
    SapModel_name = []
    ModelPath_name = []
    num_thread = num
    for i in range(num_thread):
        case_name.append(f"cases{i}")
        APIPath_name.append(os.path.join(os.getcwd(), f"cases{i}"))
        mySapObject_name.append(f"mySapObject{i}")
        SapModel_name.append(f"SapModel{i}")
        ModelPath_name.append(f"ModelPath{i}")
        mySapObject_name[i],ModelPath_name[i],SapModel_name[i] = SAPanalysis_GA_run2(APIPath_name[i])
    # for i in range(2):
    #     fit1, weight1,clo_up_val, beam_up_val = mulit_GA(ModelPath_name[i],mySapObject_name[i],SapModel_name[i],[pop_room],[pop_room_label])

    q = queue.Queue()

    threads = []
    for i in range(len(pop1)):
        q.put(i)

    for i in range(num_thread):
        t = threading.Thread(target=mulitrun_GA, args=(ModelPath_name[i],mySapObject_name[i], SapModel_name[i],pop1,pop2,pop3,q,memory_pools_all
                                                       ,memory_pools_fit,memory_pools_weight,memory_pools_col,memory_pools_beam))
        t.start()
        threads.append(t)
    for i in threads:
        i.join()
    return result,weight_1,col_up,beam_up

'''model data'''
# modular size
modular_length = 8000
modular_width = [4000,4000,5400,3600,3600,4400,4400,4000]
modular_heigth = 3000
modular_length_num = 8
modular_dis = 400
story_num = 6
corridor_width = 4000

# steel section information
sections_data_c1, type_keys_c1, sections_c1 = ms.get_section_info(section_type='c0',
                                                                  cfg_file_name="Steel_section_data.ini")

# generate model
model_data = dj.generate_model_data(modular_length,modular_width,modular_heigth,modular_length_num,modular_dis,story_num,corridor_width)
nodes = model_data[0]
edges_all = model_data[1]
labels = model_data[2]
cor_edges = model_data[3]
joint_hor = model_data[4]
joint_ver = model_data[5]
room_indx = model_data[6]

pop_room = []
pop_room_label = []
for i in range(2*story_num*3):
    pop_room.append(12)
for i in range(96):
    pop_room_label.append(0)
pop1 = []
pop3 = []
for i in range(4):
    pop1.append(pop_room)
    pop3.append(pop_room_label)


# APIPath = os.path.join(os.getcwd(), 'cases')
# mySapObject, ModelPath, SapModel = SAPanalysis_GA_run(APIPath)
# weight1,g_col,g_beam,reaction_all,section_all,all_up_name,all_up_data,Joint_dis,all_force = Sap_analy_allroom(pop_room,pop_room_label)



result = [0 for _ in range(len(pop1))]
weight_1 = [0 for _ in range(len(pop1))]
col_up = [0 for _ in range(len(pop1))]
beam_up = [0 for _ in range(len(pop1))]

zhanjiaqi,yangtingting,gongmengna,jiangjiaqi = thread_sap(2,pop1,pop3)



