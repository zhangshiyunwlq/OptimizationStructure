import tensorflow as tf
from tensorflow import keras
from keras import layers, models
import pandas as pd
import numpy as np

# 创建一个简单的神经网络模型
def create_model():
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(40,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(39)
    ])
    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['mae'])
    return model

# 加载MNIST手写数字数据集
# 从Excel文件读取数据
x_train_df = pd.read_excel(io="D:\desktop\os\optimization of structure\optimization of structure\optimization of structure\out_all_infor\\run_infor_14_1.xls",sheet_name="pop2_all")  # 假设Excel文件中没有表头
#删除表头
jiangjiaqi= x_train_df.values.tolist()
head = []
for i in range(len(jiangjiaqi)):
    if type(jiangjiaqi[i][0])== str :
        head.append(i)
for i in range(len(head)):
        jiangjiaqi.pop(int(head[i])-i)

#
y_train_df = pd.read_excel(io="D:\desktop\os\optimization of structure\optimization of structure\optimization of structure\out_all_infor\\run_infor_14_1.xls",sheet_name="pop_all_fitness")
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

x_train1 = x_train[0:500]
y_train1 = y_train[0:500]

# 数据预处理
# x_train = x_train / 255.0

# 创建神经网络模型
model = create_model()
#训练
model.fit(x=np.concatenate((x_train1, y_train1), axis=1),  # 将 x_train 和 y_train 合并作为输入
               y=x_train,  # 生成器网络的目标是生成与 x_train 相似的数据
               epochs=10,  # 进行 10 轮训练
               batch_size=32)  # 每个批次的样本数为 32

# 生成一个随机的 y_train 数据作为输入
random_y_train = np.random.rand(1, 1)

# 使用生成器网络生成新的数据
generated_data = model.predict(np.concatenate((x_train[0:1], random_y_train), axis=1))
print(generated_data)