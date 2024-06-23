import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, regularizers
import pandas as pd
import numpy as np

def leaky_relu(x, alpha=0.1):
    return tf.nn.leaky_relu(x, alpha=alpha)
# 创建一个简单的神经网络模型
def create_model(num_joint,num_out):
    model = models.Sequential([
        layers.Dense(100, activation=leaky_relu, input_shape=(num_joint,),kernel_regularizer=regularizers.l2(0.0005)),
        layers.Dense(100, activation=leaky_relu,kernel_regularizer=regularizers.l2(0.0005)),
        # layers.Dense(100, activation=leaky_relu),
        # layers.Dense(100, activation=leaky_relu),
        layers.Dense(num_out,activation='sigmoid',kernel_regularizer=regularizers.l2(0.0005))
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer,
                  loss='mse',
                  metrics=['mae'])
    # model.summary()
    return model

def get_data(num,path):
    # 加载MNIST手写数字数据集
    # 从Excel文件读取数据
    x_train_df = pd.read_excel(io=path,sheet_name="pop2_all")  # 假设Excel文件中没有表头
    #删除表头
    jiangjiaqi= x_train_df.values.tolist()
    head = []
    for i in range(len(jiangjiaqi)):
        if type(jiangjiaqi[i][0])== str :
            head.append(i)
    for i in range(len(head)):
            jiangjiaqi.pop(int(head[i])-i)

    #
    y_train_df = pd.read_excel(io=path,sheet_name="pop_all_fitness")
    row_data = y_train_df.values.tolist()
    head = []
    for i in range(len(row_data)):
        if type(row_data[i][0])== str :
            head.append(i)
    for i in range(len(head)):
            row_data.pop(int(head[i])-i)
    y_train = []
    for i in range(len(row_data)):
        y_train.extend(row_data[i])
    # 将数据转换为NumPy数组
    x_train = np.array(jiangjiaqi)
    y_train = np.array(y_train)

    x_train1 = x_train[0:num]
    y_train1 = y_train[0:num]
    return x_train1,y_train1

'''
#生成训练数据
num = 5000
path = "D:\desktop\os\optimization of structure\optimization of structure\optimization of structure\out_all_infor\\run_infor_14_1.xls"

x_train,y_train = get_data(num,path)

num_joint = len(x_train[0])
# 创建神经网络模型
model = create_model(num_joint)
#训练
model.fit(x_train, y_train, epochs=100, batch_size=512)
#预测生成y_test
# y_test_predicted = model.predict(x_test)
'''