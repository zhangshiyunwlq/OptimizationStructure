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

def get_continue_data():
    path_memo = f"D:\desktop\os\optimization of structure\optimization of structure\optimization of structure\out_all_memorize_case4\memorize_infor_{num_var}_{modular_num}_{time}.xlsx"
    path_infor = f"D:\desktop\os\optimization of structure\optimization of structure\optimization of structure\out_all_infor_case4\\run_infor_{num_var}_{modular_num}_{time}.xlsx"
    gx_nor = pd.read_excel(io=path_memo, sheet_name="memorize_gx_nor")
    gx_nor_data = gx_nor.values.tolist()

    memorize_pool_pop1 = pd.read_excel(io=path_memo, sheet_name="memorize_pool")
    memorize_pool = memorize_pool_pop1.values.tolist()

    memorize_fit1 = pd.read_excel(io=path_memo, sheet_name="memorize_fit")
    memorize_fit = memorize_fit1.values.tolist()

    memorize_weight1 = pd.read_excel(io=path_memo, sheet_name="memorize_weight")
    memorize_weight = memorize_weight1.values.tolist()


    memorize_gx1 = pd.read_excel(io=path_memo, sheet_name="memorize_gx")
    memorize_gx = memorize_gx1.values.tolist()

    gx_prediction1 = pd.read_excel(io=path_memo, sheet_name="gx_prediction")
    gx_prediction = gx_prediction1.values.tolist()

    memorize_loss1 = pd.read_excel(io=path_memo, sheet_name="memorize_loss")
    memorize_loss = memorize_loss1.values.tolist()

    memorize_mae1 = pd.read_excel(io=path_memo, sheet_name="memorize_mae")
    memorize_mae = memorize_mae1.values.tolist()

    memorize_gx_nor1 = pd.read_excel(io=path_memo, sheet_name="memorize_gx_nor")
    memorize_gx_nor = memorize_gx_nor1.values.tolist()

    memorize_num1 = pd.read_excel(io=path_memo, sheet_name="memorize_num")
    memorize_num = memorize_num1.values.tolist()

    pop2_best1 = pd.read_excel(io=path_infor, sheet_name="pop2_all")
    pop2_fitness1 = pd.read_excel(io=path_infor, sheet_name="pop_all_fitness")
    pop2_pool_all = pop2_best1.values.tolist()
    fitness_pool_all = pop2_fitness1.values.tolist()
    pop2_remove= []
    fitness_remove = []
    for i in range(len(pop2_pool_all)):
        if i <= len(pop2_pool_all):
            if type(pop2_pool_all[i][0]) == str:
                pop2_remove.append(i)

    for i in range(len(fitness_pool_all)):
        if i <= len(pop2_pool_all):
            if type(fitness_pool_all[i][0]) == str:
                fitness_remove.append(i)

    for i in range(len(pop2_remove)):
        pop2_pool_all.remove(pop2_pool_all[int(pop2_remove[len(pop2_remove)-1-i])])


    for i in range(len(fitness_remove)):
        fitness_pool_all.remove(fitness_pool_all[int(fitness_remove[len(fitness_remove)-1-i])])

    pop2_best = pop2_pool_all[(N_GENERATIONS-1)*POP_SIZE]
    fitness_best = fitness_pool_all[N_GENERATIONS-1][0]
    return fitness_best