import numpy as np
import random
from random import randint
import copy
import data_to_json as dj
import xlwt
import threading
import queue
import math as m
import os


def fun(a,b):
    pop_1=copy.deepcopy(a)
    pop_2=copy.deepcopy(b)

    for i in range(0,int(len(pop_1)/4)):
        pop_1[i] = 5*m.sin(pop_1[i])
    for i in range(int(len(pop_1)/4),int(len(pop_1)/4*2)):
        pop_1[i] = 0.1*m.exp(pop_1[i])
    for i in range(int(len(pop_1) / 4*2), int(len(pop_1) / 4 * 3)):
        pop_1[i] = 0.1 * m.log(pop_1[i]+1,3)
    for i in range(int(len(pop_1) / 4*3), int(len(pop_1) / 4 * 4)):
        pop_1[i] = m.sqrt(pop_1[i])

    for i in range(0,int(len(pop_2)/4)):
        pop_2[i] = 5*m.sin(pop_2[i])
    for i in range(int(len(pop_2)/4),int(len(pop_2)/4*2)):
        pop_2[i] = 0.1*m.exp(pop_2[i])
    for i in range(int(len(pop_2) / 4*2), int(len(pop_2) / 4 * 3)):
        pop_2[i] = 0.1 * m.log(pop_2[i]+1,3)
    for i in range(int(len(pop_2) / 4*3), int(len(pop_2) / 4 * 4)):
        pop_2[i] = m.sqrt(pop_2[i])

    result =0
    for i in range(len(pop_1)):
        result+=pop_1[i]

    for i in range(len(pop_2)):
        result+=pop_2[i]
    return result,0.5*result

def generate_DNA_coding_story5(num_var,num_room_type,x):
    all_room_num = 2*story_num*5
    room_nu =np.linspace(1, 3, 3)
    pop = np.zeros((POP_SIZE,num_var+num_room_type+all_room_num))
    for i in range(len(pop)):
        sec = list(map(int,random.sample(x.tolist(), num_var)))
        sec.sort()
        room_ty = list(map(int,random.sample(room_nu.tolist(), num_room_type)))
        room_ty.sort()
        for j in range(num_var):
            pop[i][j] = sec[j]
        for j in range(num_var,num_var+num_room_type):
            pop[i][j] = randint(0,3)
        for j in range(num_var+num_room_type,num_var+num_room_type+2*story_num*3):
            pop[i][j] = randint(0,num_var-1)
        for j in range(num_var+num_room_type+2*story_num*3,num_var+num_room_type+2*story_num*5):
            pop[i][j] = randint(0,1)
    return pop

def decoding(pop,num_var,num_room_type,labels):
    pop1_jiequ = pop[:,num_var+num_room_type:num_var+num_room_type+2*story_num*3]
    pop1_method = pop[:, num_var+num_room_type+2*story_num*3:num_var+num_room_type+2*story_num*5]
    pop_all = np.zeros((POP_SIZE,DNA_SIZE))
    pop_room_label = np.zeros((POP_SIZE, len(labels)))
    for i in range(POP_SIZE):
        for j in range(len(pop1_jiequ[0])):
            posi = int(pop1_jiequ[i][j])
            pop_all[i][j] = pop[i][posi]
    for i in range(POP_SIZE):
        for z in range(story_num):
            for j in range(z*modular_length_num*2,(z+1)*modular_length_num*2):
                posi = int(pop1_method[i][z*2+int(labels[j])-1])
                if posi == 0:
                    pop_room_label[i][j] = 0
                else:
                    pop_room_label[i][j] = pop[i][num_var]
    return pop_all,pop_room_label

def mulitrun_GA(pop1,pop_all,pop3,q,result,weight_1,col_up,beam_up,memory_pools_all,memory_pools_fit,memory_pools_weight,memory_pools_col,memory_pools_beam):
    while True:
        if q.empty():
            break

        time = q.get()
        pop = pop1[time]
        pop_room_label = pop3[time]
        pop2= pop_all[time]
        # 判断记忆池
        for j in range(len(memory_pools_all)):
            tf = []
            for z in range(len(pop2)):
                if pop2[z] != memory_pools_all[j][z]:
                    tf.append(z)
            if len(tf) == 0:
                break
        # 记忆池中存在满足条件的个体
        if len(tf) == 0:
            result[time] = memory_pools_fit[j]
            weight_1[time] = memory_pools_weight[j]
            col_up[time] = memory_pools_col[j]
            beam_up[time] = memory_pools_beam[j]
        # 记忆池中不存在满足条件的个体
        else:
            # pop_all.append(pop)

            res1, res2 = fun(pop,pop_room_label)

            # num3 += 1
            weight_1[time] = res2
            col_up[time] = 1
            beam_up[time] = 1
            result[time] = res1
        # 记忆池更新
            memory_pools_all.append(pop2)
            memory_pools_fit.append(res1)
            memory_pools_weight.append(res2)
            memory_pools_col.append(1)
            memory_pools_beam.append(1)

def thread_sap(num,pop1,pop2,pop3,result,weight_1,col_up,beam_up,memory_pools_all,memory_pools_fit,memory_pools_weight,memory_pools_col,memory_pools_beam):


    pop_n = [0 for i in range(len(pop2[0]))]
    if len(memory_pools_all)==0:
        memory_pools_all.append(pop_n)
        memory_pools_fit.append(100000)
        memory_pools_weight.append(100000)
        memory_pools_col.append(100000)
        memory_pools_beam.append(100000)

    q = queue.Queue()
    threads = []
    for i in range(len(pop1)):
        q.put(i)
    for i in range(num_thread):
        t = threading.Thread(target=mulitrun_GA, args=(pop1,pop2,pop3,q,
                            result,weight_1,col_up,beam_up,memory_pools_all,memory_pools_fit,memory_pools_weight,memory_pools_col,memory_pools_beam))
        t.start()
        threads.append(t)
    for i in threads:
        i.join()
    return result,weight_1,col_up,beam_up

def select_2(pop, fitness):  # nature selection wrt pop's fitness

    fit_ini = fitness
    luyi = fitness
    luyi.sort(reverse=True)
    sort_num = []
    lst = list(range(1, len(fit_ini)+1))
    list_new = []
    for i in range(len(fit_ini)):
        sort_num.append(fit_ini.index(luyi[i]))
    for i in range(len(fit_ini)):
        list_new.append(lst[sort_num[i]])
    for i in range(len(list_new)):
        list_new[i] = m.e ** (list_new[i] * 1.2)
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=np.array(list_new) / (sum(list_new)))
    pop2 = np.zeros((POP_SIZE, len(pop[0])))
    for i in range(len(pop2)):
        pop2[i] = pop[int(idx[i])]
    return pop2

def crossover_and_mutation_coding_story5(pop2,num_var,num_room_type,CROSSOVER_RATE):
    pop = pop2
    room_nu = np.linspace(1, 12, 12)
    new_pop = np.zeros((len(pop),len(pop[0])))
    for i in range(len(pop)):
        father = pop[i]
        child = father
        if np.random.rand() < CROSSOVER_RATE:
            mother = pop[np.random.randint(POP_SIZE)]
            cross_points1 = np.random.randint(low=0, high=len(pop[0]))
            cross_points2 = np.random.randint(low=0, high=len(pop[0]))
            while cross_points2==cross_points1:
                cross_points2 = np.random.randint(low=0, high=len(pop[0]))
            exchan = []
            exchan.append(cross_points2)
            exchan.append(cross_points1)
            for j in range(min(exchan),max(exchan)):
                child[j] = mother[j]
        mutation_1_stort5(child,x,num_var,num_room_type,MUTATION_RATE)
        new_pop[i] = child

    for i in range(len(new_pop)):
        sec_sort = []
        room_sort = []
        for j in range(num_var):
            sec_sort.append(new_pop[i][j])
        sec_sort.sort()
        for j in range(num_var):
            new_pop[i][j] = sec_sort[j]



    return new_pop

def mutation_1_stort5(child, x,num_var,num_room_type,MUTATION_RATE):
    num_var = int(num_var)
    room_nu = np.linspace(1, 12, 12)
    num_room_type = int(num_room_type)
    for mutate_point in range(num_var):
        x_var = list(map(int, x.tolist()))
        for mutate_point_1 in range(num_var):
            if child[mutate_point_1] in x_var:
                x_var.remove(child[mutate_point_1])
        if np.random.rand() < MUTATION_RATE:  # 以MUTATION_RATE的概率进行变异
            child[mutate_point] = np.random.choice(x_var)
    for j in range(num_room_type + num_var, num_room_type + num_var + DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            child[j] = randint(0,num_var-1)
    for j in range(num_var, num_room_type + num_var ):
        if np.random.rand() < MUTATION_RATE:
            child[j] = randint(0,3)


def run(num_var,num_room_type,x,labels):
    pop2= generate_DNA_coding_story5(num_var, num_room_type, x)
    pop_decoe_1 = copy.deepcopy(pop2)
    pop1,pop3 = decoding(pop_decoe_1,num_var,num_room_type,labels)

    pop_zhongqun_all = []  # 记录每代种群（不重复）
    pop_zhongqun_all_2 = []#记录种群所有
    pop_zhongqun_all_3 = []
    memory_pools_all = []
    memory_pools_fit = []
    memory_pools_weight = []
    memory_pools_col = []
    memory_pools_beam = []
    col_up_all= []
    beam_up_all=[]
    pop_all_weight=[]
    pop_all_fitness=[]
    weight_min=[]
    max_ru = []
    sap_run_time = 0
    for run_time in range(N_GENERATIONS):
        pop_zhongqun_all.append(pop1)
        pop_zhongqun_all_2.append(pop2)
        pop_zhongqun_all_3.append(pop3)

        # 计算fitness等参数
        fit = [0 for i in range(len(pop2))]
        weight = [0 for i in range(len(pop2))]
        clo_val = [0 for i in range(len(pop2))]
        beam_val = [0 for i in range(len(pop2))]
        result1,weight_pop,clo_up_1,beam_up_1=thread_sap(num_thread, pop1, pop2, pop3, fit, weight, clo_val, beam_val, memory_pools_all, memory_pools_fit,
                   memory_pools_weight, memory_pools_col, memory_pools_beam)

        col_up_all.append(clo_up_1)
        beam_up_all.append(beam_up_1)
        pop_all_weight.append(weight_pop)
        fitness2 =result1
        pop_all_fitness.append(fitness2)
        mm = fitness2.index(min(fitness2))
        weight_min.append(weight_pop[mm])
        min1 = min(fitness2)
        mm2 = pop1[mm]# 最小值对应pop1编码
        mm2_all = pop2[mm]# 最小值对应pop2编码
        mm2_all3 = pop3[mm]  # 最小值对应pop2编码
        max_ru.append(min(fitness2))# 统计历代最小值
        #选择
        pop2 = select_2(pop2, fitness2)
        #交叉变异
        pop2 = crossover_and_mutation_coding_story5(pop2, num_var, num_room_type, CROSSOVER_RATE)

        # 引入新个体
        run_time +=1
        if run_time % 5 == 0:
            pop2_new = generate_DNA_coding_story5(num_var, num_room_type, x)
            exchange_num = int(0.3*len(pop2_new))
            for ex_num in range(exchange_num):
                pop2[len(pop1) - 1 - ex_num] = pop2_new[ex_num]

        if run_time %5==0:
            print(run_time)
        pop1, pop3 = decoding(pop2, num_var, num_room_type, labels)

        aaa = []
        aaa.append(pop1[0])
        pop3_ga = []
        pop3_ga.append(pop3[0])
        # if max1 <= m.log(GA(aaa,pop3_ga)[0][0]):
        if min1 <=fun(pop1[0], pop3[0])[0]:
            sap_run_time += 1
            pop1[0] = mm2
            pop2[0] = mm2_all
            pop3[0] = mm2_all3

    out_put_result(pop_zhongqun_all, pop_zhongqun_all_2, pop_zhongqun_all_3,max_ru,weight_min,pop_all_fitness,pop_all_weight)



    return pop_zhongqun_all,pop_zhongqun_all_2,pop_zhongqun_all_3

def out_put_result(pop1_all,pop2_all,pop3_all,fitness_all,weight_all,pop_all_fitness,pop_all_weight):
    wb1 = xlwt.Workbook()
    out_pop1_all = wb1.add_sheet('pop1_all')
    loc = 0
    for i in range(len(pop1_all)):
        out_pop1_all.write(loc, 0, f'{[i]}')
        for j in range(len(pop1_all[i])):
            loc += 1
            for z in range(len(pop1_all[i][j])):
                out_pop1_all.write(loc, z, pop1_all[i][j][z])
        loc += 1

    out_pop2_all = wb1.add_sheet('pop2_all')
    loc = 0
    for i in range(len(pop2_all)):
        out_pop2_all.write(loc, 0, f'{[i]}')
        for j in range(len(pop2_all[i])):
            loc += 1
            for z in range(len(pop2_all[i][j])):
                out_pop2_all.write(loc, z, pop2_all[i][j][z])
        loc += 1

    out_pop3_all = wb1.add_sheet('pop3_all')
    loc = 0
    for i in range(len(pop3_all)):
        out_pop3_all.write(loc, 0, f'{[i]}')
        for j in range(len(pop3_all[i])):
            loc += 1
            for z in range(len(pop3_all[i][j])):
                out_pop3_all.write(loc, z, pop3_all[i][j][z])
        loc += 1

    pop_all_fit = wb1.add_sheet('pop_all_fitness')
    loc = 0
    for i in range(len(pop_all_fitness)):
        pop_all_fit.write(loc, 0, f'{[i]}')
        loc += 1
        for j in range(len(pop_all_fitness[i])):
            pop_all_fit.write(loc, j, pop_all_fitness[i][j])
        loc += 1


    pop_all_wei = wb1.add_sheet('pop_all_weight')
    loc = 0
    for i in range(len(pop_all_weight)):
        pop_all_wei.write(loc, 0, f'{[i]}')
        loc += 1
        for j in range(len(pop_all_weight[0])):
            pop_all_wei.write(loc, j, pop_all_weight[i][j])
        loc += 1


    outmaxfitness = wb1.add_sheet('max_fitness')
    loc = 0
    outmaxfitness.write(loc, 0, 'max_fitness_all')
    loc += 1
    for i in range(len(fitness_all)):
        outmaxfitness.write(loc, i, fitness_all[i])
    loc += 1
    outmaxfitness.write(loc, 0, 'min_weight_all')
    loc += 1
    for i in range(len(weight_all)):
        outmaxfitness.write(loc, i, weight_all[i])


    APIPath = os.path.join(os.getcwd(), 'out_all_infor')
    SpecifyPath = True
    if not os.path.exists(APIPath):
        try:
            os.makedirs(APIPath)
        except OSError:
            pass

    path1 = os.path.join(APIPath, 'run_infor')


    wb1.save(f'{path1}.xls')


modular_length_num = 8
modular_length = 8000
modular_width = [4000,4000,5400,3600,3600,4400,4400,4000]
modular_heigth = 3000
modular_length_num = 8
modular_dis = 400
story_num = 6
corridor_width = 4000


POP_SIZE = 50
DNA_SIZE = 2*story_num*3
CROSSOVER_RATE = 0.35
MUTATION_RATE = 0.4
N_GENERATIONS = 100
num_thread = 25
min_genera = []

x = np.linspace(0, 12, 13)
num_var = 8
num_room_type=1

label=[1,1,1,1,2,2,2,2]
labels = []
for i in range(12):
    labels.extend(label)

zhan,jia,qi=run(num_var,num_room_type,x,labels)
