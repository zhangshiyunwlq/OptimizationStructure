import numpy as np

# 示例桁架结构的杆件数量
num_members = 10

# 示例截面尺寸的选择
section_sizes = [1, 2, 3, 4, 5]

# 计算总重量的函数
def calculate_weight(sections, lengths, density):
    weight = 0
    for i in range(len(sections)):
        weight += sections[i] * lengths[i] * density
    return weight

# 计算应力的函数（假设一个简单的线性应力模型）
def calculate_stress(sections, forces):
    stress = []
    for i in range(len(sections)):
        stress.append(forces[i] / sections[i])
    return stress

# 生成初始解
def generate_initial_solution(num_members, section_sizes):
    return [np.random.choice(section_sizes) for _ in range(num_members)]

# 生成邻居解
def get_neighbors(solution, section_sizes):
    neighbors = []
    for i in range(len(solution)):
        for size in section_sizes:
            if size != solution[i]:
                neighbor = solution.copy()
                neighbor[i] = size
                neighbors.append(neighbor)
    return neighbors

# 禁忌搜索算法
def tabu_search(lengths, forces, density, allowed_stress, tabu_tenure, max_iterations):
    current_solution = generate_initial_solution(num_members, section_sizes)
    best_solution = current_solution
    best_cost = calculate_weight(current_solution, lengths, density)
    tabu_list = []
    tabu_tenures = {}

    for _ in range(max_iterations):
        neighbors = get_neighbors(current_solution, section_sizes)
        best_neighbor = None
        best_neighbor_cost = float('inf')

        for neighbor in neighbors:
            stress = calculate_stress(neighbor, forces)
            if all(s <= allowed_stress for s in stress):
                neighbor_cost = calculate_weight(neighbor, lengths, density)
                move = tuple(neighbor)

                if (move not in tabu_list or neighbor_cost < best_cost) and neighbor_cost < best_neighbor_cost:
                    best_neighbor = neighbor
                    best_neighbor_cost = neighbor_cost

        current_solution = best_neighbor
        if best_neighbor_cost < best_cost:
            best_solution = best_neighbor
            best_cost = best_neighbor_cost

        move = tuple(current_solution)
        tabu_list.append(move)
        tabu_tenures[move] = tabu_tenure

        if len(tabu_list) > tabu_tenure:
            expired_move = tabu_list.pop(0)
            del tabu_tenures[expired_move]

    return best_solution, best_cost

# 示例使用
lengths = np.random.uniform(1, 10, num_members)  # 示例每个杆件的长度
forces = np.random.uniform(1, 100, num_members)  # 示例每个杆件的受力
density = 0.1  # 材料密度
allowed_stress = 150  # 允许应力
tabu_tenure = 5
max_iterations = 100

best_sections, best_weight = tabu_search(lengths, forces, density, allowed_stress, tabu_tenure, max_iterations)
print("Best sections:", best_sections)
print("Best weight:", best_weight)
