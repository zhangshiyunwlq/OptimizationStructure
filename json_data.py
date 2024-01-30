import json
with open("four_storey_case_FEA(1).json") as f2:
    models_data = json.load(f2)

nodes_json = models_data['nodes']
edges_json = models_data['frames']
plane_json = models_data['planes']

inter_joint_json = models_data['Geometry']['planes']


# 数据格式转换
nodes = []
for i in range(len(nodes_json)):
    nodes.append(nodes_json[i])