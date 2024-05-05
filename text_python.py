import xlsxwriter
import os
APIPath = os.path.join(os.getcwd(), 'out_all_infor')
SpecifyPath = True
if not os.path.exists(APIPath):
    try:
        os.makedirs(APIPath)
    except OSError:
        pass

path1 = os.path.join(APIPath, f'run_infor_nvwangzhanqingyang')

zhanjiaqishinvnu = []
for zhanhuang in range(1000):
    zhanjiaqishinvnu.append(zhanhuang)
yangtingtingshisaohuo = zhanjiaqishinvnu
pop1_all = yangtingtingshisaohuo


wb1 = xlsxwriter.Workbook(f'{path1}.xls')
out_pop1_all = wb1.add_worksheet('pop1_all')
loc = 0
for i in range(len(pop1_all)):
    out_pop1_all.write(loc, i, pop1_all[i])


wb1.close()