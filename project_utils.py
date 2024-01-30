import configparser
import numpy as np
import copy
import os
import sys
import comtypes.client
import win32com.client
import xlwt
import  math as m

def get_section_info(section_type, cfg_file_name="Steel_section_data33.ini"):
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
        if cfg_section_data[section_datasets[i]]['type'] == f'{section_type}':
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
