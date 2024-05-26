import xlsxwriter
import os
num_var = 8
modular_num = 55
time =4

DNN_fit_pred = [[1,2,3,4],[5,6,7,9],[0.5,0.8]]
APIPath = os.path.join(os.getcwd(), 'out_all_DNN_fitness_case4')
SpecifyPath = True
if not os.path.exists(APIPath):
    try:
        os.makedirs(APIPath)
    except OSError:
        pass

path1 = os.path.join(APIPath, f'prediction_fitness_{num_var}_{modular_num}_{time}')
wb1 = xlsxwriter.Workbook(f'{path1}.xlsx')
fit_pred = wb1.add_worksheet(f'DNN_fitness')
loc = 0
for ii in range(len(DNN_fit_pred)):

    for i in range(len(DNN_fit_pred[ii])):
        fit_pred.write(loc, i, DNN_fit_pred[ii][i])
    loc += 1

wb1.close()
