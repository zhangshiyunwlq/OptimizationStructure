import xlsxwriter
import os
num_var = 8
modular_num = 55
time =4

gx_all = [[[2,5],[5,6]],[[3,3,5],[4,4,3]]]
APIPath = os.path.join(os.getcwd(), 'out_all_prediction_case4')
SpecifyPath = True
if not os.path.exists(APIPath):
    try:
        os.makedirs(APIPath)
    except OSError:
        pass

path1 = os.path.join(APIPath, f'prediction_infor_{num_var}_{modular_num}_{time}')

wb1 = xlsxwriter.Workbook(f'{path1}.xlsx')

for ii in range(len(gx_all)):
    gx_pred = wb1.add_worksheet(f'gx_prediction_{ii}')
    loc = 0
    for i in range(len(gx_all[ii])):
        for j in range(len(gx_all[ii][i])):
            gx_pred.write(loc, j, gx_all[ii][i][j])
        loc += 1

wb1.close()
