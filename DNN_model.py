
from random import random
import subprocess
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from numpy.lib.function_base import disp
from torch._C import dtype
import GA as GApart
import pickle
import pandas as pd
import imp
import copy

import TrussElement3D as Truss

# %matplotlib qt

# ------------------- 1. initialization ------------------
# region[rgba(255,255,255,0.1)]   
# xyTruss = Truss.Truss3D()
imp.reload(Truss)
xyTruss_plt = Truss.Truss3D_plot()

tp = 75.0/2.0
Nodes = np.array([[-tp,0.0,200.0],[tp,0.0,200.0],[-tp,tp,100.0],[tp,tp,100.0],
                  [tp,-tp,100.0],[-tp,-tp,100.0],[-100.0,100.0,0.0],[100.0,100.0,0.0],
                  [100.0,-100.0,0.0],[-100.0,-100.0,0.0]], dtype=float)
Elements = np.array([[0,1],[0,3],[1,2],[0,4],[1,5],[1,3],[1,4],[0,2],[0,5],
                     [2,5],[3,4],[2,3],[4,5], 
                     [2,9],[5,6],[3,8],[4,7],[3,6],[2,7],[4,9],[5,8],
                     [5,9],[2,6],[3,7],[4,8]], dtype=int)
Area = np.zeros((Elements.shape[0],)) + 2
Ey = np.zeros((Elements.shape[0],)) + 10e6
BCsDof = [[6*3+0,6*3+1,6*3+2, 7*3+0,7*3+1,7*3+2, 8*3+0,8*3+1,8*3+2, 9*3+0,9*3+1,9*3+2]]
FF = np.zeros((Nodes.shape[0]*3,))
FF[0] = 1000
FF[6] = 500
FF[15] = 600
FF[[1,4]] = -10000
FF[[2,5]] = -10000


TrussOri = {}
TrussOri['Nodes'] = Nodes
TrussOri['Elements']  = Elements

# Area = [22.9,26.5,13.5,1.62,1.8,7.97,22.0,1.62,30,1.62] 
TrussOri['Area'] = Area
TrussOri['Ey'] = Ey
TrussOri['BCsDof'] = BCsDof
TrussOri['FF'] = FF
TrussOri['density'] = 0.1

Truss1 = Truss.Truss3D(TrussOri)
U, SE, Stress, Weight = Truss1.Solve()


xyTruss_plt.plot_member(Nodes, Elements)
xyTruss_plt.plot_disp(Nodes,Elements,U,1,100)

imp.reload(Truss)
xyTruss_plt = Truss.Truss3D_plot()
fig2 = xyTruss_plt.plot_3D_member(Nodes, Elements, N=50, R1=10, R2=10)

fig2.savefig('1.jpg',dpi=400)
# endregion
    
# --------------- 2. GA -------------
# region[rgba(255,255,255,0.1)]   

r_cross=0.5
r_mut = 0.2
pop_num = 30
var_num = Elements.shape[0]

var_low = np.zeros((var_num,))+0
var_upp =  np.zeros((var_num,))+41

penalties = [1e6, 1e7]
req = [25e3, 2.0011] # 2m

imp.reload(GApart)
xyGA = GApart.GeneticAlgorithm(pop_num,var_num, var_low, var_upp)
CurrentPop = xyGA.PopGeneration()
Best_ind_History = []
Pop_History = [] 


for i in range(300):

    CurrentPop, CurBestIndividual = xyGA.runGA(Truss.Truss3D, TrussOri, CurrentPop, r_cross, r_mut, penalties, req, epoch=1)
    
    fitavg = [fit['fitOri'] for fit in CurrentPop]
    fitavg = np.sum(fitavg) / len(fitavg)
    Best_ind_History.append(CurBestIndividual)
    Pop_History.append(CurrentPop)
    
    print()
    print("iters:{:3d} | c1:{:6.2f} | g1:{:6.2} | g2:{:8.2f}".format(len(Pop_History), CurBestIndividual['c1'], CurBestIndividual['g1'], CurBestIndividual['g2']))
    # print("gen:{}, {}, {}, {}".format(CurBestIndividual['gen'][0], CurBestIndividual['gen'][1],CurBestIndividual['gen'][2],CurBestIndividual['gen'][3]))
    # print()
    # print(CurBestIndividual)

# name = 'Case1-1_'+str(len(Pop_History))+'.pkl' 
# with open(name,'wb') as f:
#     pickle.dump([Pop_History,Best_ind_History],f)


# endregion

# ------------------- 2.1 data plot -------------------
# region[rgba(255,255,255,0.1)]  


imp.reload(Truss)
xyTruss_plt = Truss.Truss3D_plot()
Truss1 = copy.deepcopy(TrussOri)

with open('Case1-'+str(8)+'_300.pkl','rb') as f:
    [tp1, tp2] = pickle.load(f)    

xyTruss_plt.plot_member_size(Nodes, Elements, xyGA.encoding(tp2[-1]['gen']), 20)
fig2 = xyTruss_plt.plot_3D_member_size(Nodes, Elements, xyGA.encoding(tp2[-1]['gen']), N=50, R1=100*0.3, R2=110*0.3)




Truss1_ana = Truss.Truss3D(Truss1)
U, SE, Stress, Weight = Truss1_ana.Solve()


import DataPlot as DataPlt
imp.reload(DataPlt)
x = np.arange(len(Best_ind_History))
y = np.array([i['c1'] for i in Best_ind_History])
DataPlt.PlotCurve(x,y)    
    
    
tpp = []   
for i in range(10):
    with open('Case1-'+str(i+1)+'_300.pkl','rb') as f:
        [tp1, tp2] = pickle.load(f)    
    tpp.append(tp2)

xx = []
yy = []
for i in range(len(tpp)):
    x = np.arange(len(tpp[i]))
    y = np.array([i['c1'] for i in tpp[i]])
    xx.append(x)
    yy.append(y)    


imp.reload(DataPlt)
# DataPlt.PlotMultiCurves(xx,yy)   
fig2 = DataPlt.PlotMultiCurves_fill(xx,yy,[7,1])

fig2.savefig('1.jpg',dpi=400)
#endregion


# -------------------- 3. ANN + GA ---------------
# --------------------- 3.1 GA -----------------------------
# region[rgba(255,255,150,0.1)]
r_cross=0.5
r_mut = 0.2
pop_num = 30
var_num = Elements.shape[0]

var_low = np.zeros((var_num,))+0
var_upp =  np.zeros((var_num,))+41

penalties = [1e7, 1e7]
req = [25e3, 2.0011] # 2m

imp.reload(GApart)
xyGA = GApart.GeneticAlgorithm(pop_num,var_num, var_low, var_upp)
CurrentPop = xyGA.PopGeneration()
Best_ind_History = []
Pop_History = [] 


for i in range(20):

    CurrentPop, CurBestIndividual = xyGA.runGA(Truss.Truss3D, TrussOri, CurrentPop, r_cross, r_mut, penalties, req, epoch=1)
    
    fitavg = [fit['fitOri'] for fit in CurrentPop]
    fitavg = np.sum(fitavg) / len(fitavg)
    Best_ind_History.append(CurBestIndividual)
    Pop_History.append(CurrentPop)
    
    print()
    print("iters:{:3d} | c1:{:6.2f} | g1:{:6.2} | g2:{:8.2f}".format(len(Pop_History), CurBestIndividual['c1'], CurBestIndividual['g1'], CurBestIndividual['g2']))
    # print("gen:{}, {}, {}, {}".format(CurBestIndividual['gen'][0], CurBestIndividual['gen'][1],CurBestIndividual['gen'][2],CurBestIndividual['gen'][3]))
    # print()
    # print(CurBestIndividual)


with open('P1-20.pkl','wb') as f:
    pickle.dump([Pop_History,Best_ind_History],f)


# with open('Case1-5_300.pkl','rb') as f:
#     [tp1, tp2] = pickle.load(f)    
# tpp.append(tp2)

# endregion


# ------------------- 3.2 Data saving ------------------
# region[rgba(255,255,150,0.1)]

import DataProcess as DProcess
imp.reload(DProcess)

dir1 = os.path.join(os.getcwd(),'data\\DataX1.csv')
dir2 = os.path.join(os.getcwd(),'data\\DataY1.csv')
for xyi in range(len(Pop_History)):
    for xyj in range(len(Pop_History[0])):
        
        xyPop = Pop_History[xyi][xyj]
        tpX = xyGA.encoding(xyPop['gen'])

        tpY = [xyPop['c1'], xyPop['g1'], xyPop['g2']]
        tpXX = tpX
        tpXX = np.array(tpXX).reshape((1,-1))
        tpY = np.array(tpY).reshape((1,-1)) 
        # DataX,DataY = DProcess.DataCheckandStore(DataX,DataY,tpXX,tpY)
        if DProcess.DataCheck(dir1, tpXX):
            DProcess.StoreData(dir1, tpXX)
            DProcess.StoreData(dir2, tpY)        
            
# endregion

# ---------------------- 3.3 ANN training --------------
# region[rgba(255,255,150,0.1)]
import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import ANN_model as xyNN
import DataProcess as xyDP

dir1 = os.path.join(os.getcwd(),'data\\DataX1.csv')
dir2 = os.path.join(os.getcwd(),'data\\DataY1.csv')
DataX_ori = np.array(xyDP.ReadData(dir1), dtype=float)
DataY_ori = xyDP.ReadData(dir2)

imp.reload(xyDP)
x_max = np.array([33.50]*10, dtype=float)
x_min = np.array([1.62]*10, dtype=float)
DataX = xyDP.norm_data_withMaxMin(DataX_ori, x_max, x_min)
DataY,mean_y,std_y = xyDP.norm_data2(DataY_ori)


x_train = DataX[:DataX.shape[0]*4//5,:]
x_test = DataX[DataX.shape[0]*4//5:,:]
y_train = DataY[:DataY.shape[0]*4//5,:]
y_test = DataY[DataY.shape[0]*4//5:,:]
    

imp.reload(xyNN)
y_model = xyNN.xyModel1(10, 3, 200, 200)
# y_model = xyNN.xyModel2(10, 3, 100)

# y_model = xyNN.xyModel3(10, 3, 200, 200,200)
loss_fun1 = xyNN.LossFun1()

# train_loss1 = xyNN.training_model(x_train, y1_train, epochs=2000, batch_size=20, model=y1_model, opt=opt1, loss_func=loss_fun1)

# train_loss1, valid_loss1, test_loss1 = xyNN.training_model_kthFolded(5, x_train, y_train, x_test, y_test, epochs=100, batch_size=20, model=y_model) 
train_loss1, test_loss1 = xyNN.training_model(x_train, y_train, x_test, y_test, epochs=200, batch_size=20, model = y_model)



y_model.eval()
indx = 1
xy_model = y_model
xy_x1 = x_train 
xy_x2 = x_test
xy_y1 = y_train
xy_y2 =y_test
with torch.no_grad():
    xypred = xy_model(torch.tensor(xy_x1,dtype=torch.float))
    xypred2 = xy_model(torch.tensor(xy_x2,dtype=torch.float))
    print(loss_fun1(xypred, torch.tensor(xy_y1,dtype=torch.float)).item()) 
    print(loss_fun1(xypred2, torch.tensor(xy_y2,dtype=torch.float)).item())
         
    # print(loss_fun1(xypred2[:,indx], torch.tensor(xy_y2[:,indx],dtype=torch.float)).item())
    # print(loss_fun1(xypred[:,indx], torch.tensor(xy_y1[:,indx],dtype=torch.float)).item())   
# endregion

# ----------- 3.4 ANN plot -------------------
# region[rgba(255,255,150,0.1)]
indx = 2
x1 = np.arange(-2,2,0.01)
y1 = np.arange(-2,2,0.01)
fig, ax = plt.subplots(1,1)
ax.scatter(y_train[:,indx], xypred.cpu()[:,indx], c='red')
ax.scatter(y_test[:,indx], xypred2.cpu()[:,indx], c='black')
ax.plot(x1, y1, color='blue', linewidth=2)
ax.set(xlim=[0,1], ylim=[0, 1]) 
plt.show()



x = np.arange(0,100,1)
y1 = train_loss1
y3 = test_loss1
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x,y1, color = 'red', linewidth = 3, label = 'train_loss')
ax.plot(x,y3, color = 'black', linewidth = 3, label = 'test_loss')
# ax.axis('equal')
ax.set(xlim=[0,220], ylim=[0, 1])
# ax.set(ylabel='Load factor', xlabel='iterations')
ax.legend(loc=1, frameon=False, bbox_to_anchor=(1, 0.5))
plt.show()

# endregion


# -------------------- 3.5. ANN + GA ---------------
# region[rgba(255,255,150,0.1)]

import GA as xyGA_ori

r_cross=0.5
r_mut = 0.2
pop_num = 30
var_num = Elements.shape[0]

var_low = np.zeros((var_num,))+0
var_upp =  np.zeros((var_num,))+41

penalties = [1e5, 1e5]
req = [25e3, 2.0011] # 2m
Pop = []


flag = True

while flag:
    imp.reload(xyGA_ori)

    xyGA = xyGA_ori.GeneticAlgorithm(pop_num, var_num, var_low, var_upp)
    CurrentPop = xyGA.PopGeneration()
    ANNmodel = y_model

    tp_best_his = []
    tp_Pop_his = []

    for i in range(100):        

        CurrentPop, CurBestIndividual = xyGA.runGA_ANN(ANNmodel, x_max, x_min, mean_y, std_y, CurrentPop, r_cross, r_mut, penalties, req, epoch=1)
        
        fitavg = [fit['fitOri'] for fit in CurrentPop]
        fitavg = np.sum(fitavg) / len(fitavg)
        tp_best_his.append(CurBestIndividual)
        tp_Pop_his.append(CurrentPop)
        
        print()
        print("iters:{:3d} | c1:{:6.2f} | g1:{:6.2} | g2:{:8.2f}".format(len(tp_best_his), CurBestIndividual['c1'], CurBestIndividual['g1'], CurBestIndividual['g2']))


    if len(Pop) == pop_num-1:
        print('OK')
        flag = False
    else:
        Pop.append(CurBestIndividual)

tp = np.argmin([i['fitOri'] for i in Pop])
Pop.append(Best_ind_History[-1])


with open('P1-160-Pop.pkl','wb') as f:
    pickle.dump([Pop],f)
    
Pop[tp]
# endregion


# ------------------ 3.6 Next sampling + GA ------
# region[rgba(255,255,150,0.1)]
r_cross=0.5
r_mut = 0.2
pop_num = 30
var_num = Elements.shape[0]

var_low = np.zeros((var_num,))+0
var_upp =  np.zeros((var_num,))+41

penalties = [1e5, 1e5]
req = [25e3, 2.0011] # 2m

imp.reload(GApart)
xyGA = GApart.GeneticAlgorithm(pop_num,var_num, var_low, var_upp)

for i in Pop:
    i['fitness'] = 0
CurrentPop = copy.deepcopy(Pop)

Best_ind_History = []
Pop_History = [] 

for i in range(20):

    CurrentPop, CurBestIndividual = xyGA.runGA(Truss.Truss3D, TrussOri, CurrentPop, r_cross, r_mut, penalties, req, epoch=1)
    
    fitavg = [fit['fitOri'] for fit in CurrentPop]
    fitavg = np.sum(fitavg) / len(fitavg)
    Best_ind_History.append(CurBestIndividual)
    Pop_History.append(CurrentPop)
    
    print()
    print("iters:{:3d} | c1:{:6.2f}".format(len(Pop_History), CurBestIndividual['c1']))
    # print("gen:{}, {}, {}, {}".format(CurBestIndividual['gen'][0], CurBestIndividual['gen'][1],CurBestIndividual['gen'][2],CurBestIndividual['gen'][3]))
    # print()
    # print(CurBestIndividual)

with open('P1-160.pkl','wb') as f:
    pickle.dump([Pop_History, Best_ind_History],f)

Best_ind_History[-1]
# endregion

# ------------------ 3.7 Next, save data  and ANN training--------------
# region[rgba(255,255,150,0.1)]

import DataProcess as xyDP
import shutil
indx1 = 6 ####################
indx2 = indx1+1
dir1 = os.path.join(os.getcwd(),'data\\DataX'+str(indx2)+'.csv')
dir2 = os.path.join(os.getcwd(),'data\\DataY'+str(indx2)+'.csv')
dirtp1 = os.path.join(os.getcwd(),'data\\DataX'+str(indx1)+'.csv')
dirtp2 = os.path.join(os.getcwd(),'data\\DataY'+str(indx1)+'.csv')
shutil.copyfile(dirtp1, dir1)
shutil.copyfile(dirtp2, dir2)

add_x = []
add_y = []
for xyi in range(len(Pop_History)):
    for xyj in range(len(Pop_History[0])):
        
        xyPop = Pop_History[xyi][xyj]
        tpX = xyGA.encoding(xyPop['gen'])

        tpY = [xyPop['c1'], xyPop['g1'], xyPop['g2']]
        tpXX = tpX
        tpXX = np.array(tpXX).reshape((1,-1))
        tpY = np.array(tpY).reshape((1,-1)) 
        # DataX,DataY = DProcess.DataCheckandStore(DataX,DataY,tpXX,tpY)

        if DProcess.DataCheck(dir1, tpXX):
            add_x.append(tpXX)
            add_y.append(tpY)
     
if len(add_x)>0:
    add_x = np.array(add_x).reshape((-1,10))
    add_y = np.array(add_y).reshape((-1,3))
    DProcess.StoreData(dir1, add_x)
    DProcess.StoreData(dir2, add_y)        


DataX_ori = np.array(xyDP.ReadData(dir1), dtype=float)
DataY_ori = xyDP.ReadData(dir2)

imp.reload(xyDP)
x_max = np.array([33.50]*10, dtype=float)
x_min = np.array([1.62]*10, dtype=float)
DataX = xyDP.norm_data_withMaxMin(DataX_ori, x_max, x_min)
DataY = xyDP.norm_data2_withMeanStd(DataY_ori, np.array(mean_y), np.array(std_y))
# DataY,mean_y,std_y = xyDP.norm_data2(DataY_ori)

DataX_tp, DataY_tp = xyDP.divideData(DataX, DataY, indx2)


x_train_tp, x_test_tp, y_train_tp, y_test_tp, x_train, x_test, y_train, y_test = xyDP.prepare_trainingData(DataX_tp, DataY_tp)

# y_model_ori = copy.deepcopy(y_model) 

indx = -1
train_loss1, test_loss1 = xyNN.training_model(x_train_tp[indx], y_train_tp[indx], x_test_tp[indx], y_test_tp[indx], epochs=150, batch_size=20, model = y_model)

train_loss1, test_loss1 = xyNN.training_model(x_train, y_train, x_test, y_test, epochs=50, batch_size=20, model = y_model)



y_model.eval()
indx = 3
xy_model = y_model

# xy_x1 = x_train_tp[indx]
# xy_x2 = x_test_tp[indx]
# xy_y1 = y_train_tp[indx]
# xy_y2 =y_test_tp[indx]

xy_x1 = x_train
xy_x2 = x_test
xy_y1 = y_train
xy_y2 =y_test

with torch.no_grad():
    xypred = xy_model(torch.tensor(xy_x1,dtype=torch.float))
    xypred2 = xy_model(torch.tensor(xy_x2,dtype=torch.float))
    print(loss_fun1(xypred, torch.tensor(xy_y1,dtype=torch.float)).item())       
    print(loss_fun1(xypred2, torch.tensor(xy_y2,dtype=torch.float)).item())
  


xytp = 2
x1 = np.arange(-2,2.5,0.01)
y1 = np.arange(-2,2.5,0.01)
fig, ax = plt.subplots(1,1)
ax.scatter(xy_y1[:,xytp], xypred[:,xytp], c='red')
ax.scatter(xy_y2[:,xytp], xypred2[:,xytp], c='black')
ax.plot(x1, y1, color='blue', linewidth=2)
ax.set(xlim=[-2,2.5], ylim=[-2, 2.5]) 
plt.show()    



xyk = 0
x = np.arange(0,100,1)
y1 = train_loss1[xyk,:].T
y2 = valid_loss1[xyk,:].T
y3 = test_loss1[xyk,:].T
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x,y1, color = 'red', linewidth = 3, label = 'train_loss')
ax.plot(x,y2, color = 'blue', linewidth = 3, label = 'valid_loss')
ax.plot(x,y3, color = 'black', linewidth = 3, label = 'valid_loss')
# ax.axis('equal')
ax.set(xlim=[0,550], ylim=[0, 2])
# ax.set(ylabel='Load factor', xlabel='iterations')
# ax.legend(loc=1, frameon=False, bbox_to_anchor=(1, 0.5))
plt.show()
# endregion


# ------------------- 3.8 Plotting truss ----------------
imp.reload(Truss)
xyTruss_plt = Truss.Truss3D_plot()
size = xyGA.encoding([39, 37, 32, 0, 0, 28, 36, 2, 40, 3])
fig2 = xyTruss_plt.plot_3D_member_size(Nodes, Elements, size, N=50, R1=100*3, R2=110*3)

fig2.savefig('1.jpeg',dpi=400)

# ----------- test xy ----------------
# region[rgba(255,255,255,0.1)]
# x = np.random.rand(1000)*(10)-5
# y = np.random.rand(1000)*(10)-5
# z = Fy(x,y)


# x_train = np.array([x[:x.shape[0]*4//5],y[:y.shape[0]*4//5]]).T
# z_train = np.array([z[:z.shape[0]*4//5]]).T
# x_test  = np.array([x[x.shape[0]*4//5:],y[y.shape[0]*4//5:]]).T
# z_test  = np.array([z[z.shape[0]*4//5:]]).T

# imp.reload(xyNN)
# y1_model, opt1, loss_fun1 = xyNN.get_model1(0.01, 2, 1, 64, 64, 64)

# # train_loss1 = xyNN.training_model(x_train, y1_train, epochs=200, batch_size=100, model=y1_model, opt=opt1, loss_func=loss_fun1)

# train_loss1 = xyNN.training_model(x_train, z_train, epochs=500, batch_size=100, model=y1_model, opt=opt1, loss_func=loss_fun1)   

# y1_model.eval()
# with torch.no_grad():
#     xypred = y1_model(torch.tensor(x_train,dtype=torch.float))
# endregion

    # save data
    # outputModel(dir, mat, CurBestIndividual, num_h=32, bt_num=0)

    # for i in range(len(Best_ind_History)):
    #     c1 = Best_ind_History[i]
    #     print()
    #     print("iters:{:3d} | r1:{:8.2f} | st1:{:8.2f}".format(i+1, c1['r1'], c1['st1']))
    #     print("gen:{}, {}, {}, {}".format(c1['gen'][0], c1['gen'][1],c1['gen'][2],c1['gen'][3]))        




    # fitness, r1, st1, weightReq1 = xyGA.evaluate(AnsysModel, ReadAnalysisData, dir, mat, c1, penalties, req)

    # testIndi = CurBestIndividual.copy()
    # testIndi['gen'][0] = 20
    # print(xyGA.evaluate(AnsysModel, ReadAnalysisData, dir, mat, testIndi,penalties, req))


    # # StoreData(dir,nodes,sec,disp,stress)

    # # var = ReadData(dir,nodes.shape[0], elements.shape[0])



