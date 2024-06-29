import pandas as pd
import copy
import numpy as np
from CNN import create_model
import data_to_json as dj
from random import randint
import model_to_sap as ms
import os
import modular_utils as md
import sap_run as sr
import threading
import queue
import configparser
import comtypes.client
import random

from keras.callbacks import EarlyStopping,LearningRateScheduler
import sys
import xlsxwriter
import matplotlib.pyplot as plt
#修改区间

# 定义余弦退火学习率调度函数
import pickle
import os


num_var = 10
modular_num=10
time = 10

# # 假设我们有一个要保存的对象，例如一个字典
data = {'name': 'Alice', 'age': 25, 'city': 'Wonderland'}
#
# 指定保存路径

APIPath = os.path.join(os.getcwd(), 'out_DNN_model')
SpecifyPath = True
if not os.path.exists(APIPath):
    try:
        os.makedirs(APIPath)
    except OSError:
        pass

path1 = os.path.join(APIPath, f'DNN_model_{num_var}_{modular_num}_{time}.pkl')
file_name = 'DNN_model_10_10_10.pkl'

# 保存对象为PKL文件
with open(file_name, 'wb') as file:
    pickle.dump(data, file)

print(f"Data saved to {path1}")

# # 指定保存路径
# save_path = r'D:\desktop\os\optimization of structure\out_DNN_model'
# file_name = 'DNN_model_10_10_10.pkl'
# full_path = os.path.join(save_path, file_name)
#
# file_path = full_path
# os.chmod(file_path, 0o755)
#
# # 确保目录存在
# try:
#     os.makedirs(save_path, exist_ok=True)
# except PermissionError:
#     print(f"Permission denied: Cannot create directory {save_path}")
#     raise
#
# # 保存对象为PKL文件
# try:
#     with open(full_path, 'wb') as file:
#         pickle.dump(data, file)
#     print(f"Data saved to {full_path}")
# except PermissionError:
#     print(f"Permission denied: Cannot write to file {full_path}")
#     raise
# except FileNotFoundError:
#     print(f"File not found: The system cannot find the path specified {full_path}")
#     raise
# except Exception as e:
#     print(f"An error occurred: {e}")
#     raise
