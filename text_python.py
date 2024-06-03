import numpy as np
import xlsxwriter
import os
import pandas as pd
num_var = 5
modular_num = 3
time =7
N_GENERATIONS = 140
POP_SIZE = 30

memorize_pool = []
memorize_fit = []
memorize_weight = []
memorize_col = []
memorize_beam = []
memorize_sum = []
memorize_gx = []
memorize_gx_nor = []
memorize_num = []

a = [[1,2,3],[2,5,8],[3,5,7]]
a = np.array(a)
b = a.tolist()
