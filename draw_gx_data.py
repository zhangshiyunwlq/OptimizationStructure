import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def draw_gx_chayi(gx_truth,gx_pred):
    gx_truth_div = []
    gx_pred_div = []
    for i in range(len(gx_truth[0])):
        temp1=[]
        temp2=[]
        for j in range(len(gx_truth)):
            temp1.append(gx_truth[j][i])
            temp2.append(gx_pred[j][i])
        gx_truth_div.append(temp1)
        gx_pred_div.append(temp2)
    return gx_truth_div,gx_pred_div
def draw_gx_chayi2(gx_truth_div,gx_pred_div,time):
    fig2 = plt.figure(num=1, figsize=(23, 30))
    ax2 = fig2.add_subplot(111)
    ax2.tick_params(labelsize=40)
    ax2.set_xlabel("Iteration", fontsize=50)  # 添加x轴坐标标签，后面看来没必要会删除它，这里只是为了演示一下。
    ax2.set_ylabel('fitness', fontsize=50)  # 添加y轴标签，设置字体大小为16，这里也可以设字体样式与颜色
    ax2.spines['bottom'].set_linewidth(4);  ###设置底部坐标轴的粗细
    ax2.spines['left'].set_linewidth(4)
    ax2.spines['right'].set_color('none')
    ax2.spines['top'].set_color('none')
    # plt.ylim((150, 400))

    color_data = ['b', 'g', 'r', 'c', 'k', 'm']


    fig2 = plt.figure(num=1, figsize=(23, 30))
    ax2 = fig2.add_subplot(111)
    ax2.tick_params(labelsize=40)
    ax2.set_xlabel("Iteration", fontsize=50)  # 添加x轴坐标标签，后面看来没必要会删除它，这里只是为了演示一下。
    ax2.set_ylabel('fitness', fontsize=50)  # 添加y轴标签，设置字体大小为16，这里也可以设字体样式与颜色
    ax2.spines['bottom'].set_linewidth(4);  ###设置底部坐标轴的粗细
    ax2.spines['left'].set_linewidth(4)
    ax2.spines['right'].set_color('none')
    ax2.spines['top'].set_color('none')
    # plt.ylim((150, 400))


    bbb = np.arange(0, len(gx_pred_div[0]))
    ax2.plot(bbb, gx_pred_div[time], linewidth=3, color=color_data[time],linestyle="--",marker='x', markersize=5)
    ax2.plot(bbb, gx_truth_div[time], linewidth=3, color=color_data[time],linestyle="-",marker='o', markersize=5)
    ax2.set(xlim=(0, len(gx_pred_div[0])),ylim=(0, 1),
            xticks=np.arange(0, len(gx_pred_div[0]), 30),yticks=np.arange(0, 1, 0.1),
                )
        # for i in range(7):
        #     x_te = []
        #     for j in range(10):
        #         x_te.append(20 * i - 1)
        #     x_te = np.array(x_te)
        #     y_te = np.linspace(0, 1.5, 10)
        #     ax2.plot(x_te, y_te, linewidth=1, color='black')
    plt.show()
    # plt.clf()


path_memo = f"D:\desktop\os\optimization of structure\out_all_truth_pred_data4\\prediction_truth_5_5_0.xlsx"
gx_pred_all = pd.read_excel(io=path_memo, sheet_name="gx_pred", header=None)
gx_pred_all = gx_pred_all.values.tolist()

gx_truth_all = pd.read_excel(io=path_memo, sheet_name="gx_truth", header=None)
gx_truth_all = gx_truth_all.values.tolist()

gx_truth_div,gx_pred_div=draw_gx_chayi(gx_truth_all,gx_pred_all)
gx_truth_div = [a[:180] for a in gx_truth_div]
gx_pred_div = [a[:180] for a in gx_pred_div]

draw_gx_chayi2(gx_truth_div,gx_pred_div,0)
