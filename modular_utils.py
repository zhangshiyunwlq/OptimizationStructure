import numpy as np
import copy


class ModularType2:
    def __init__(self, modular_nodes, modular_label):
        self.modular_nodes = modular_nodes
        self.modular_label = modular_label
        self.modular_edges, self.modular_edge_labels,self.modular_edge_labels_1,self.modular_edge_labels_2,self.modular_edge_labels_3 = self.Lines_Generation()
        self.modular_planes, self.modular_plane_labels = self.Planes_Generation()
        self.modular_four_planes = self.Planes_Generation_Four()[0]
        self.modular_four_planes_labels = self.Planes_Generation_Four()[1]
        # self.modular_top_edges = [self.modular_planes[i] for i in range(len(self.modular_planes)) if self.modular_plane_labels[i] == 'top_plane']
        # self.modular_bottom_edges = [self.modular_planes[i] for i in range(len(self.modular_planes)) if self.modular_plane_labels[i] == 'bottom_plane']
        self.modular_column_edges = [self.modular_edges[i] for i in range(len(self.modular_edges)) if self.modular_edge_labels[i] == 'column_edge']
        self.modular_top_edges = [np.array([4,5,6,7])]
        self.modular_bottom_edges = [np.array([0,1,2,3])]
        self.modular_column_edges = [self.modular_edges[i] for i in range(len(self.modular_edges)) if self.modular_edge_labels[i] == 'column_edge']
        '''self.modular_info'''

        return

    def Lines_Generation(self):
        """
        :return beam_edge: eight sorted beam edges of the modular; a numpy array [8,2].
        :return column_edge: four sorted column edges of the modular; a numpy array [4,2]
        """
        beam_edge = np.zeros((8, 2), dtype=int)
        for i in range(3):
            beam_edge[i, 0] = i + 1
            beam_edge[i, 1] = beam_edge[i, 0] + 1
        beam_edge[3, 0] = 4
        beam_edge[3, 1] = 1

        for i in range(4, 8):
            beam_edge[i, 0] = beam_edge[i - 4, 0] + 4
            beam_edge[i, 1] = beam_edge[i - 4, 1] + 4

        column_edge = np.zeros((4, 2), dtype=int)
        for i in range(4):
            column_edge[i, 0] = i + 1
            column_edge[i, 1] = column_edge[i, 0] + 4

        edge_labels = []
        for i in range(4):
            edge_labels.append('column_edge')
        for i in range(4):
            edge_labels.append('bottom_edge')
        for i in range(4):
            edge_labels.append('top_edge')

        edges_labels_1 = ['all_member', 'all_member', 'all_member', 'all_member', 'all_member', 'all_member',
                          'all_member', 'all_member', 'all_member', 'all_member', 'all_member', 'all_member']
        edges_labels_2 = ['short_member', 'short_member', 'short_member', 'short_member', 'short_member', 'long_member',
                          'short_member', 'long_member', 'short_member', 'long_member', 'short_member', 'long_member']

        edges_labels_3 = ['left_member', 'right_member','right_member','left_member','short_member','right_member',
                          'short_member','left_member','short_member','right_member','short_member','left_member']
        edges_labels_4 = ['front_member', 'front_member','back_member','back_member','front_member','long_member',
                          'back_member','long_member','front_member','long_member','back_member','long_member']
        edges_labels_5 = ['column', 'column','column','column','short_member','long_member',
                          'short_member','long_member','short_member','long_member','short_member','long_member']

        modular_edges = np.concatenate((column_edge-1, beam_edge-1), axis=0)



        return modular_edges, edge_labels,edges_labels_1,edges_labels_2,edges_labels_3

    def Planes_Generation(self):
        side_planes = np.array([[0, 1, 5],
                                [0, 5, 4],
                                [1, 2, 6],
                                [1, 6, 5],
                                [2, 3, 7],
                                [2, 7, 6],
                                [0, 4, 7],
                                [0, 7, 3]], dtype=int)
        top_bottom_planes = np.array([[4, 5, 6],
                                      [4, 6, 7],
                                      [0, 3, 2],
                                      [0, 2, 1]], dtype=int)

        modular_planes = np.concatenate((side_planes, top_bottom_planes), axis=0)
        plane_labels = ['front_plane', 'front_plane', 'side_plane1', 'side_plane1', 'back_plane', 'back_plane',
                        'side_plane2', 'side_plane2', 'top_plane', 'top_plane', 'bottom_plane', 'bottom_plane']
        return modular_planes, plane_labels

    def Planes_Generation_Four(self):
        modular_four_planes = np.array([[0, 1, 5, 4],
                           [1, 2, 6, 5],
                           [2, 3, 7, 6],
                           [3, 0, 4, 7],
                           [0, 1, 2, 3],
                           [4, 5, 6, 7]], dtype=int)
        modular_four_planes_labels = ['front_plane','right_plane', 'back_plane', 'left_plane', 'bottom_plane', 'top_plane']
        return modular_four_planes, modular_four_planes_labels

    """
    LoD2_1 add component data to modular
    :return: self.modular_info; modular_edge_labels;
    """
    def Add_Info_And_Update_Modular(self, modular_info):
        self.modular_info = modular_info
        current_nodes = self.modular_nodes
        current_edges = self.modular_edges
        current_edge_labels = self.modular_edge_labels
        current_planes = self.modular_planes
        current_plane_labels = self.modular_plane_labels

        if modular_info['type'] == 'support_1':
            x1,y1,z1 = current_nodes[5]
            x2,y2,z2 = current_nodes[6]
            x3,y3,z3 = current_nodes[4]
            x4,y4,z4 = current_nodes[7]
            new_nodes1 = 0.5*np.array([x1+x2, y1+y2, z1+z2])
            new_nodes2 = 0.5*np.array([x3+x4, y3+y4, z3+z4])
            current_nodes = np.vstack((current_nodes, new_nodes1))
            current_nodes = np.vstack((current_nodes, new_nodes2))

            edges = np.array([[0,4],[1,5], [2,6], [3,7],
                              [0,1], [1,2], [2,3], [0,3],
                              [4,5], [5,8], [6,8], [6,7], [7,9], [4,9],
                              [1,8], [2,8],[0,9],[3,9]], dtype=int)

            planes =  np.array([[0, 1, 5],
                                [0, 5, 4],
                                [1, 8, 5],
                                [1, 2, 8],
                                [2, 6, 8],
                                [2, 3, 6],
                                [3, 7, 6],
                                [0, 4, 9],
                                [0, 9, 3],
                                [3, 9, 7],
                                [4, 5, 6],
                                [4, 6, 7],
                                [0, 3, 2],
                                [0, 2, 1]
                                ], dtype=int)

            self.modular_nodes = current_nodes
            self.modular_edges = edges
            self.modular_planes = planes

            edge_labels = []
            for i in range(4):
                edge_labels.append('column_edge')
            for i in range(4):
                edge_labels.append('bottom_edge')
            for i in range(6):
                edge_labels.append('top_edge')
            for i in range(4):
                edge_labels.append('support_edge')

            plane_labels = []
            for i in range(2):
                plane_labels.append('front_plane')
            for i in range(3):
                plane_labels.append('side_plane1')
            for i in range(2):
                plane_labels.append('back_plane')
            for i in range(3):
                plane_labels.append('side_plane2')
            for i in range(2):
                plane_labels.append('top_plane')
            for i in range(2):
                plane_labels.append('bottom_plane')

            self.modular_edge_labels = edge_labels
            self.plane_labels = plane_labels

        return

    """LoD2_2 add info to the components"""
    def add_info_to_components(self, modular_info):
        return


class ModularBuilding:
    def __init__(self, nodes, nodes_indx, modular_edges, label,joints_hor,joints_ver, co_beam):
        self.building_nodes = nodes
        self.building_nodes_indx = nodes_indx
        self.building_room_edges = modular_edges
        self.building_room_labels = label
        self.building_room_joints_hor = joints_hor
        self.building_room_joints_ver = joints_ver
        self.corr_beams = co_beam

        tp = np.array(self.building_room_edges)
        tp2 = np.reshape(tp, (-1, 2))
        tp3 = np.sort(tp2, axis=1)

        # for i in range(2, len(self.building_room_joints)):
        #     if self.building_room_joints[i][0, 0] != self.building_room_joints[i][0, 1]:
        #         joints_all = np.append(joints_all, self.building_room_joints[i], axis=0)

        co_beam_all = self.corr_beams[0]
        for i in range(1, len(self.corr_beams)):
            co_beam_all = np.append(co_beam_all, self.corr_beams[i], axis=0)


        self.building_edges = np.unique(tp3, axis=0)
        self.building_modulars = self.Modulars_Initialization()
        tp_planes = copy.deepcopy(self.building_modulars[0].modular_planes)
        for i in range(tp_planes.shape[0]):
            for j in range(tp_planes.shape[1]):
                tp_planes[i,j] = self.building_nodes_indx[0][tp_planes[i,j]]

        for i in range(len(self.building_modulars)-1):
            tp_planes_1 = copy.deepcopy(self.building_modulars[i+1].modular_planes)
            for ii in range(tp_planes_1.shape[0]):
                for jj in range(tp_planes_1.shape[1]):
                    tp_planes_1[ii, jj] = self.building_nodes_indx[i+1][tp_planes_1[ii, jj]]
            tp_planes = np.vstack((tp_planes,tp_planes_1))

        tp_planes3 = np.sort(tp_planes, axis=1)
        tp_planes4, unique_indice = np.unique(tp_planes3,return_index=True, axis=0)
        tp_planes2 = tp_planes[unique_indice]
        self.building_planes = tp_planes2
        self.corridor_all_beams = co_beam_all

        return

    def Modulars_Initialization(self):
        modulars = []
        temp_node_indx = self.building_nodes_indx
        temp_nodes = self.building_nodes
        temp_labels = self.building_room_labels
        for i in range(len(temp_node_indx)):
            # import pdb;
            # pdb.set_trace()
            modular_nodes = temp_nodes[temp_node_indx[i]]
            modular_i = ModularType2(modular_nodes, temp_labels[i])
            modulars.append(modular_i)

        return modulars


    ''' modular assembling '''
    def Del_The_Same(self, modular_edges):
        # 将三维数组转换为二维数组
        temp = []
        for i in range(len(modular_edges)):
            for j in modular_edges[i]:
                temp.append(j)
        temp = np.array(temp).tolist()
        new_list = []
        for sublist in temp:
            if sublist not in new_list:
                new_list.append(sublist)
        return new_list

    def Building_Assembling(self, modulars):
        nodes = []
        edges = []
        label = []
        for i in range(len(modulars)):
            label.append(modulars[i].modular_label)
            nodes.append(modulars[i].modular_nodes.tolist())
            edges.append([[modulars[i].modular_nodes[k].tolist()] + [modulars[i].modular_nodes[v].tolist()] for k, v in modulars[i].modular_edges])# 把nodes代入edges

        # 删除重复nodes
        node_set = self.Del_The_Same(nodes)
        # Node_indx
        Node_set = []
        Nodes_indx = []
        for i in range(len(edges)):
            Node_set.append([[node_set.index(x) for x in inner] for inner in edges[i]])
            Nodes_indx.append(self.Del_The_Same(Node_set[i]))

        modular_edges = []
        for i in range(len(edges)):
            modular_edges.append([[node_set.index(x) for x in inner] for inner in edges[i]])


        self.building_nodes = np.array(node_set)
        self.building_nodes_indx = [np.array(tp_i, dtype=int) for tp_i in Nodes_indx]
        self.building_room_edges = [np.array(tp_i, dtype=int) for tp_i in modular_edges]
        self.building_room_labels = np.array(label)

        temp = []
        for i in range(len(self.building_room_edges)):
            for j in self.building_room_edges[i]:
                temp.append(j)
        tp = np.array(temp)
        tp2 = np.reshape(tp, (-1, 2))
        tp3 = np.sort(tp2, axis=1)
        self.building_edges = np.unique(tp3, axis=0)

        return