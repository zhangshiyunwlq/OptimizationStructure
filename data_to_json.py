import numpy as np
import configparser
import matplotlib.pyplot as plt
import pyvista as pv

def generate_model_data(modular_length,modular_width,modular_heigth,modular_length_num,modular_dis,story_num,corridor_width):
    '''room labels'''
    labels = []
    labels1 = [1, 1, 1, 1, 2, 2, 2, 2]
    for i in range(story_num*2):
        labels.extend(labels1)

    '''room nodes'''
    # room nodes
    # 第一个房间
    nodes = np.zeros((8 * modular_length_num * 2 * story_num, 3))
    # nodes = np.zeros((1000,3))
    nodes[0] = [0, 0, 0]
    nodes[1] = [modular_width[0], 0, 0]
    nodes[2] = [modular_width[0], modular_length, 0]
    nodes[3] = [0, modular_length, 0]
    for i in range(4, 8):
        nodes[i] = nodes[i - 4]
        nodes[i, 2] = nodes[i, 2] + modular_heigth
    # 第一排房间
    for i in range(1, modular_length_num):
        for j in [0, 3, 4, 7]:
            nodes[i * 8 + j] = nodes[i * 8 + j - 8]
            nodes[i * 8 + j, 0] = nodes[i * 8 + j - 8, 0] + modular_width[i - 1] + modular_dis
        nodes[i * 8 + 1] = nodes[i * 8]
        nodes[i * 8 + 1, 0] = nodes[i * 8 + 0, 0] + modular_width[i]
        nodes[i * 8 + 2] = nodes[i * 8 + 3]
        nodes[i * 8 + 2, 0] = nodes[i * 8 + 3, 0] + modular_width[i]
        nodes[i * 8 + 5] = nodes[i * 8 + 4]
        nodes[i * 8 + 5, 0] = nodes[i * 8 + 4, 0] + modular_width[i]
        nodes[i * 8 + 6] = nodes[i * 8 + 7]
        nodes[i * 8 + 6, 0] = nodes[i * 8 + 7, 0] + modular_width[i]
        # for j in range(8):
        #     nodes[i*8+j]=nodes[i*8+j-8]
        #     nodes[i * 8 + j, 0] = nodes[i * 8 + j - 8, 0]+modular_width[i-1]+modular_dis
    # 第一层房间
    for i in range(modular_length_num, modular_length_num * 2):
        for j in range(8):
            nodes[i * 8 + j] = nodes[i * 8 + j - modular_length_num * 8]
            nodes[i * 8 + j, 1] = nodes[i * 8 + j - modular_length_num * 8, 1] + modular_length + corridor_width
    # 所有房间
    for i in range(1, story_num):
        for j in range(i * modular_length_num * 2, (i + 1) * modular_length_num * 2):
            for z in range(8):
                nodes[j * 8 + z] = nodes[j * 8 - modular_length_num * 2 * 8 + z]
                nodes[j * 8 + z, 2] = nodes[j * 8 - modular_length_num * 2 * 8 + z, 2] + modular_heigth + modular_dis

    '''room edges'''
    edges = []
    room1_edges = np.zeros((12, 2))
    room1_edges[0] = [0, 4]
    room1_edges[1] = [1, 5]
    room1_edges[2] = [2, 6]
    room1_edges[3] = [3, 7]
    room1_edges[4] = [0, 1]
    room1_edges[5] = [1, 2]
    room1_edges[6] = [2, 3]
    room1_edges[7] = [3, 0]
    room1_edges[8] = [4, 5]
    room1_edges[9] = [5, 6]
    room1_edges[10] = [6, 7]
    room1_edges[11] = [7, 4]
    edges.append(room1_edges)

    for i in range(1, modular_length_num * 2 * story_num):
        room2_edges = np.zeros((12, 2))
        for j in range(12):
            room2_edges[j, 0] = room1_edges[j, 0] + i * 8
            room2_edges[j, 1] = room1_edges[j, 1] + i * 8
        edges.append(room2_edges)
    edges_all = edges[0]
    for i in range(1, len(edges)):
        edges_all = np.append(edges_all, edges[i], axis=0)
    # 走廊边
    edges_corrider = []
    room3_edges = np.zeros((modular_length_num * (2) * (story_num *2), 2))
    room3_edges[0] = [3, 3 + 5 + (modular_length_num - 1) * 8]
    room3_edges[1] = [2, 2 + 7 + (modular_length_num - 1) * 8]
    for i in range(1, modular_length_num):
        room3_edges[i * 2, 0] = room3_edges[0, 0] + i * 8
        room3_edges[i * 2, 1] = room3_edges[0, 1] + i * 8
        room3_edges[i * 2 + 1, 0] = room3_edges[1, 0] + i * 8
        room3_edges[i * 2 + 1, 1] = room3_edges[1, 1] + i * 8
    for i in range(1, story_num):
        for j in range(i * modular_length_num * 2, (i + 1) * modular_length_num * 2):
            for z in range(2):
                room3_edges[j, z] = room3_edges[j - i * modular_length_num * 2, z] + modular_length_num * 8 * 2 * i
    # 添加顶层
    # for i in range(modular_length_num * story_num * 2, modular_length_num * (story_num + 1) * 2):
    #     for j in range(2):
    #         room3_edges[i, j] = room3_edges[i - (modular_length_num * 2), j] + 4

    #添加顶层
    for i in range(modular_length_num * story_num * 2, len(room3_edges)):
        for j in range(2):
            room3_edges[i, j] = room3_edges[i - (modular_length_num * story_num * 2), j] + 4
    '''room joint'''
    # 水平节点
    joint_hor = np.zeros(((modular_length_num - 1) * 4 * 2 * (story_num), 2))
    joint_hor[0] = [1, 8]
    joint_hor[1] = [2, 11]
    joint_hor[2] = [5, 12]
    joint_hor[3] = [6, 15]
    # 第一排水平节点
    for i in range(1, modular_length_num - 1):
        for j in range(4):
            joint_hor[i * 4 + j, 0] = joint_hor[j, 0] + 8 * i
            joint_hor[i * 4 + j, 1] = joint_hor[j, 1] + 8 * i
    # 第一层水平节点
    for i in range((modular_length_num - 1) * 4, 4 * 2 * (modular_length_num - 1)):
        for j in range(2):
            joint_hor[i, j] = joint_hor[i - (modular_length_num - 1) * 4, j] + modular_length_num * 8
    # 所有水平节点
    for i in range(1, story_num):
        for j in range(i * 4 * 2 * (modular_length_num - 1), (i + 1) * 4 * 2 * (modular_length_num - 1)):
            for z in range(2):
                joint_hor[j, z] = joint_hor[
                                      j - i * (modular_length_num - 1) * 2 * 4, z] + modular_length_num * 8 * 2 * i

    # 竖直节点
    joint_ver = np.zeros((modular_length_num * 2 * 2 * (story_num - 1) * 2, 2))
    joint_ver[0] = [4, modular_length_num * 8 * 2]
    # 第一个房间
    for i in range(1, 4):
        joint_ver[i, 0] = joint_ver[0, 0] + i
        joint_ver[i, 1] = joint_ver[0, 1] + i
    for i in range(1 * 4, modular_length_num * 2 * (story_num - 1) * 4):
        for j in range(2):
            joint_ver[i, j] = joint_ver[i - 4, j] + 8
    indxs = []
    for i in range(modular_length_num*2*story_num):
        indxs0 = [0, 1, 2, 3, 4, 5, 6, 7]
        for j in range(8):
            indxs0[j] =indxs0[j] + 8 * i
        indxs.append(indxs0)

    model_data=[nodes,edges_all,labels,room3_edges,joint_hor,joint_ver,indxs]
    for i in [1,3,4,5]:
        model_data[i] = model_data[i].astype(int)
    return model_data

def model_vis(model_data):
    p = pv.Plotter(shape=(1, 1))
    x = model_data[0][:, 0]
    y = model_data[0][:, 1]
    z = model_data[0][:, 2]
    for j in [1,3,4,5]:
        model_data[j] = model_data[j].astype(int)
        for i in range(len(model_data[j])):
            tube2 = pv.Tube((x[model_data[j][i, 0]], y[model_data[j][i, 0]], z[model_data[j][i, 0]]),
                            (x[model_data[j][i, 1]], y[model_data[j][i, 1]], z[model_data[j][i, 1]]), radius=100)
            p.add_mesh(tube2, color=[0.5, 0.5, 0.5], show_edges=False)
    p.set_background('white')
    p.show()


'''basic information'''
#modular size
modular_length = 8000
modular_width = [4000,4000,5400,3600,3600,4400,4400,4000]
modular_heigth = 4000
modular_length_num = 8
modular_dis = 400
story_num = 6
#corridor size
corridor_width = 4000

model_data = generate_model_data(modular_length,modular_width,modular_heigth,modular_length_num,modular_dis,story_num,corridor_width)
nodes = model_data[0]
edges_all = model_data[1]
room3_edges = model_data[3]
joint_hor = model_data[4]
joint_ver = model_data[5]
room_indx = model_data[6]

# p = pv.Plotter(shape=(1, 1))
# x = nodes[:, 0]
# y = nodes[:, 1]
# z = nodes[:, 2]
# edges_all = edges_all.astype(int)
# for i in range(len(edges_all)):
#     tube2 = pv.Tube((x[edges_all[i, 0]], y[edges_all[i, 0]], z[edges_all[i, 0]]),
#                     (x[edges_all[i, 1]], y[edges_all[i, 1]], z[edges_all[i, 1]]), radius=100)
#     p.add_mesh(tube2, color=[0.5, 0.5, 0.5], show_edges=False)
#
# room3_edges=room3_edges.astype(int)
# for i in range(len(room3_edges)):
#     tube2 = pv.Tube((x[room3_edges[i, 0]], y[room3_edges[i, 0]], z[room3_edges[i, 0]]),
#                     (x[room3_edges[i, 1]], y[room3_edges[i, 1]], z[room3_edges[i, 1]]), radius=100)
#     p.add_mesh(tube2, color=[0.5, 0.5, 0.5], show_edges=False)
#
# joint_hor=joint_hor.astype(int)
# for i in range(len(joint_hor)):
#     tube2 = pv.Tube((x[joint_hor[i, 0]], y[joint_hor[i, 0]], z[joint_hor[i, 0]]),
#                     (x[joint_hor[i, 1]], y[joint_hor[i, 1]], z[joint_hor[i, 1]]), radius=100)
#     p.add_mesh(tube2, color=[0.5, 0.5, 0.5], show_edges=False)
#
# joint_ver=joint_ver.astype(int)
# for i in range(len(joint_ver)):
#     tube2 = pv.Tube((x[joint_ver[i, 0]], y[joint_ver[i, 0]], z[joint_ver[i, 0]]),
#                     (x[joint_ver[i, 1]], y[joint_ver[i, 1]], z[joint_ver[i, 1]]), radius=100)
#     p.add_mesh(tube2, color=[0.5, 0.5, 0.5], show_edges=False)
# p.set_background('white')
# p.show()
# edges[1][0]