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
import xlrd
import matplotlib.pyplot as plt
from CNN import create_model

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

def generate_DNA_coding_story1(num_var,num_room_type,x):
    all_room_num = story_num*4
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
            pop[i][j] = randint(1,3)
        for j in range(num_var+num_room_type,num_var+num_room_type+story_num*3):
            pop[i][j] = randint(0,num_var-1)
        for j in range(num_var+num_room_type+story_num*3,num_var+num_room_type+story_num*4):
            pop[i][j] = randint(0,1)
    return pop

def decoding1(pop,num_var,num_room_type,labels):
    pop1_jiequ = pop[:,num_var+num_room_type:num_var+num_room_type+story_num*3]
    pop1_method = pop[:, num_var+num_room_type+story_num*3:num_var+num_room_type+story_num*4]
    pop_all = np.zeros((POP_SIZE,DNA_SIZE))
    pop_room_label = np.zeros((POP_SIZE, len(labels)))
    for i in range(POP_SIZE):
        for j in range(len(pop1_jiequ[0])):
            posi = int(pop1_jiequ[i][j])
            pop_all[i][j] = pop[i][posi]
    for i in range(POP_SIZE):
        for z in range(story_num):
            for j in range(z*modular_length_num*2,(z+1)*modular_length_num*2):
                posi = int(pop1_method[i][int(labels[j])-1])
                if posi == 0:
                    pop_room_label[i][j] = 0
                else:
                    pop_room_label[i][j] = pop[i][num_var]
    return pop_all,pop_room_label



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

#不加记忆池
def mulitrun_GA_1(pop1,pop_all,pop3,q,result,weight_1,col_up,beam_up):
    while True:
        if q.empty():
            break

        time = q.get()
        pop = pop1[time]
        pop_room_label = pop3[time]
        pop2= pop_all[time]
        value = 0
        for i in range(len(memorize_pool)):
            sum_code = sum(pop2)
            if sum_code==memorize_sum[i]:
                pop2_list = copy.deepcopy(pop2)
                memorize_list = copy.deepcopy(memorize_pool[i])
                if pop2_list.tolist()==memorize_list.tolist():
                    result[time]=memorize_fit[i]
                    weight_1[time]=memorize_weight[i]
                    col_up[time]=memorize_col[i]
                    beam_up[time]=memorize_beam[i]
                    value = 1
                    break
        if value ==0:
            res1, res2 = fun(pop,pop_room_label)

            # num3 += 1
            weight_1[time] = res2
            col_up[time] = 1
            beam_up[time] = 1
            result[time] = res1
            memorize_sum.append(sum(pop2))
            memorize_pool.append(pop2)
            memorize_fit.append(res1)
            memorize_weight.append(res2)
            memorize_col.append(col_up[time])
            memorize_beam.append(beam_up[time])


def thread_sap(num,pop1,pop2,pop3,result,weight_1,col_up,beam_up,memory_pools_all,memory_pools_fit,memory_pools_weight,memory_pools_col,memory_pools_beam):


    q = queue.Queue()
    threads = []
    for i in range(len(pop1)):
        q.put(i)
    for i in range(num_thread):
        t = threading.Thread(target=mulitrun_GA_1, args=(pop1,pop2,pop3,q,
                            result,weight_1,col_up,beam_up))
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
        list_new[i] = m.e ** (list_new[i] * 1)
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

#传统遗传算法
def run(num_var,num_room_type,x,labels):
    pop2= generate_DNA_coding_story1(num_var, num_room_type, x)
    pop_decoe_1 = copy.deepcopy(pop2)
    pop1,pop3 = decoding1(pop_decoe_1,num_var,num_room_type,labels)
    # memo_code = []
    # for i in range(len(pop2[0])):
    #     memo_code.append(20)
    # memorize_sum.append(sum(memo_code))
    # memorize_pool.append(np.array(memo_code))
    # memorize_fit.append(10000)
    # memorize_weight.append(10000)
    # memorize_col.append(10000)
    # memorize_beam.append(10000)

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
    min_ru = []
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
        # if run_time ==2:
        #     for i in range(len(pop2)):
        #         pop2[i]=memo_code
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
        min_ru.append(min(fitness2))# 统计历代最小值
        fit_best.append(min1)
        #选择
        pop2 = select_2(pop2, fitness2)
        #交叉变异
        pop2 = crossover_and_mutation_coding_story5(pop2, num_var, num_room_type, CROSSOVER_RATE)

        # 引入新个体
        run_time +=1
        if run_time % 10 == 0:
            pop2_new = generate_DNA_coding_story1(num_var, num_room_type, x)
            exchange_num = int(0.3*len(pop2_new))
            for ex_num in range(exchange_num):
                pop2[len(pop1) - 1 - ex_num] = pop2_new[ex_num]

        if run_time %10==0:
            print(run_time)
            print(f'记忆池数量:{len(memorize_pool)}')
        pop1, pop3 = decoding1(pop2, num_var, num_room_type, labels)

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

    out_put_result(pop_zhongqun_all, pop_zhongqun_all_2, pop_zhongqun_all_3,min_ru,weight_min,pop_all_fitness,pop_all_weight)



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

def draw_picture(name,title_name):
    value_str = []
    wb = xlrd.open_workbook(
        filename=f'D:\desktop\os\optimization of structure\optimization of structure\optimization of structure\out_all_infor\\run_infor.xls',
        formatting_info=True)
    sheet1 = wb.sheet_by_index(5)
    for z in range(N_GENERATIONS):
        rows = sheet1.row_values(1)[z]
        value_str.append(rows)
    print(value_str[0])

    num1 = 0.8
    num2 = 0.75
    num3 = 3
    num4 = 0
    fig2 = plt.figure(num=1, figsize=(23, 30))
    ax2 = fig2.add_subplot(111)
    ax2.tick_params(labelsize=40)
    ax2.set_xlabel("Iteration",fontsize=50)  # 添加x轴坐标标签，后面看来没必要会删除它，这里只是为了演示一下。
    ax2.set_ylabel(title_name, fontsize=50)  # 添加y轴标签，设置字体大小为16，这里也可以设字体样式与颜色
    ax2.spines['bottom'].set_linewidth(4);###设置底部坐标轴的粗细
    ax2.spines['left'].set_linewidth(4)
    ax2.spines['right'].set_color('none')
    ax2.spines['top'].set_color('none')
    plt.ylim((0, 1000))
    info = copy.deepcopy(value_str)

    for j in range(len(info)):
        if info[j]>=500:
        # info[i][j] = 500+100*(m.log(info[i][j]))
            info[j] = 500 + info[j]/1000


    bbb = np.arange(0, len(info))
    ccc = info
    ax2.plot(bbb, ccc, label = name,linewidth=6)
    ax2.legend(bbox_to_anchor=(num1, num2), loc=num3, borderaxespad=num4,  handlelength=1.5, fontsize=30, shadow=False)

    plt.show()

#生成一个种群
def DNN_GA(run_time,num_pop):
    pop_best = []
    all_room_num = story_num*4
    pop_generation = np.zeros((POP_SIZE,num_var+num_room_type+all_room_num))
    for num in range(num_pop):
        x_train = np.array(memorize_pool)
        y_train = np.array(memorize_fit)
        # 创建神经网络模型
        model = create_model()
        # 训练
        model.fit(x_train, y_train, epochs=1000, batch_size=256)
        generate_DNA_coding_story1(num_var, num_room_type, x)
        pop2 = generate_DNA_coding_story1(num_var, num_room_type, x)
        for i in range(run_time):
            fitness2 =model.predict(pop2)

            mm = fitness2.index(min(fitness2))
            min1 = min(fitness2)
            mm2_all = pop2[mm]
            #选择
            pop2 = select_2(pop2, fitness2)
            # 交叉变异
            pop2 = crossover_and_mutation_coding_story5(pop2, num_var, num_room_type, CROSSOVER_RATE)

            if min1 <= model.predict(pop2[0]):
                pop2[0] = mm2_all
        #取出种群中最优秀的m个个体
        fitness_co = copy.deepcopy(fitness2)
        min_number = []
        min_index = []
        for _ in range(int(POP_SIZE/num_pop)):
            number = min(fitness_co)
            index = fitness_co.index(number)
            fitness_co[index] = 1000000
            min_number.append(number)
            min_index.append(index)

        for ind in range(len(min_index)):
            pop_best.append(pop2[min_index[ind]])
    #将得到的所有优秀个体组合为一个新种群
    for i in range(len(pop_best)):
        pop_generation[i]=pop_best[i]

    return pop_generation

#生成多个优秀个体
def DNN_GA_indi(pop,run_time,num_ind,num_joint):

    pop_best = []
    all_room_num = story_num*4
    pop_generation = np.zeros((num_ind,num_var+num_room_type+all_room_num))
    x_train = np.array(memorize_pool)
    y_train = np.array(memorize_fit)
    # 创建神经网络模型
    model = create_model(num_joint,1)
    # 训练
    history=model.fit(x_train, y_train, epochs=100, batch_size=32)
    loss = history.history['loss']
    generate_DNA_coding_story1(num_var, num_room_type, x)
    pop2 = pop
    for i in range(run_time):
        fitness1 =model.predict(pop2)
        fitness2=[]
        for i in range(len(fitness1)):
            fitness2.append(fitness1[i].tolist()[0])
        mm = fitness2.index(min(fitness2))
        min1 = min(fitness2)
        mm2_all = pop2[mm]
        #选择
        pop2 = select_2(pop2, fitness2)
        # 交叉变异
        pop2 = crossover_and_mutation_coding_story5(pop2, num_var, num_room_type, CROSSOVER_RATE)
        fit_pred = model.predict(pop2)
        if min1 <= fit_pred[0].tolist()[0]:
            pop2[0] = mm2_all
    #取出种群中最优秀的m个个体
    fitness_co = copy.deepcopy(fitness2)
    min_number = []
    min_index = []
    for _ in range(int(num_ind)):
        number = min(fitness_co)
        index = fitness_co.index(number)
        fitness_co[index] = 1000000
        min_number.append(number)
        min_index.append(index)

    #将得到的所有优秀个体组合为一个新种群
    for i in range(num_ind):
        pop_generation[i]=pop2[min_index[i]]

    return pop_generation

#基于神经网络每隔N代引入多个优秀个体
def run_DNN_ind(num_var,num_room_type,x,labels):
    pop2= generate_DNA_coding_story1(num_var, num_room_type, x)
    pop_decoe_1 = copy.deepcopy(pop2)
    pop1,pop3 = decoding1(pop_decoe_1,num_var,num_room_type,labels)
    # memo_code = []
    # for i in range(len(pop2[0])):
    #     memo_code.append(20)
    # memorize_sum.append(sum(memo_code))
    # memorize_pool.append(np.array(memo_code))
    # memorize_fit.append(10000)
    # memorize_weight.append(10000)
    # memorize_col.append(10000)
    # memorize_beam.append(10000)

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
    min_ru = []
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
        # if run_time ==2:
        #     for i in range(len(pop2)):
        #         pop2[i]=memo_code
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
        min_ru.append(min(fitness2))# 统计历代最小值
        fit_best.append(min1)
        #选择
        pop2 = select_2(pop2, fitness2)
        #交叉变异
        pop2 = crossover_and_mutation_coding_story5(pop2, num_var, num_room_type, CROSSOVER_RATE)

        # 引入新个体
        run_time +=1
        if run_time % 10 == 0:
            pop2_new = DNN_GA_indi(pop2,run_time,int(0.3*len(pop2)),len(pop2[0]))
            exchange_num = int(0.3*len(pop2))
            for ex_num in range(exchange_num):
                pop2[len(pop1) - 1 - ex_num] = pop2_new[ex_num]

        if run_time %10==0:
            print(run_time)
            print(f'记忆池数量:{len(memorize_pool)}')
        pop1, pop3 = decoding1(pop2, num_var, num_room_type, labels)

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

    out_put_result(pop_zhongqun_all, pop_zhongqun_all_2, pop_zhongqun_all_3,min_ru,weight_min,pop_all_fitness,pop_all_weight)



    return pop_zhongqun_all,pop_zhongqun_all_2,pop_zhongqun_all_3

def draw_plot_picture(info):

    num1 = 0.8
    num2 = 0.60
    num3 = 3
    num4 = 0
    fig2 = plt.figure(num=1, figsize=(23, 30))
    ax2 = fig2.add_subplot(111)
    ax2.tick_params(labelsize=40)
    ax2.set_xlabel("Iteration",fontsize=50)  # 添加x轴坐标标签，后面看来没必要会删除它，这里只是为了演示一下。
    ax2.set_ylabel('title_name', fontsize=50)  # 添加y轴标签，设置字体大小为16，这里也可以设字体样式与颜色
    ax2.spines['bottom'].set_linewidth(4);###设置底部坐标轴的粗细
    ax2.spines['left'].set_linewidth(4)
    ax2.spines['right'].set_color('none')
    ax2.spines['top'].set_color('none')
    # plt.ylim((150, 400))
    for i in range(len(info)):
        bbb = np.arange(0, len(info[i]))
        ccc = info[i]
        ax2.plot(bbb, ccc, label = i,linewidth=6)
        ax2.legend(bbox_to_anchor=(num1, num2), loc=num3, borderaxespad=num4,  handlelength=1.5, fontsize=30, shadow=False)

    plt.show()


modular_length_num = 8
modular_length = 8000
modular_width = [4000,4000,5400,3600,3600,4400,4400,4000]
modular_heigth = 3000
modular_length_num = 8
modular_dis = 400
story_num = 6
corridor_width = 4000


POP_SIZE = 50
DNA_SIZE = story_num*3
CROSSOVER_RATE = 0.35
MUTATION_RATE = 0.3
N_GENERATIONS = 100
num_thread = 25
min_genera = []

x = np.linspace(0, 6, 7)
num_var = 5
num_room_type=1

memorize_pool = []
memorize_fit = []
memorize_weight = []
memorize_col = []
memorize_beam = []
memorize_sum = []

fit_best = []
fit_all_best = []
fit_all_best2 = []
# label=[1,1,1,1,2,2,2,2]
# labels = []
# for i in range(12):
#     labels.extend(label)
labels = []
for i in range(1,7):
    for j in range(16):
        labels.append(i)
for  i in range(3):
    for num_var in [6]:
        fit_best = []
        memorize_pool = []
        memorize_fit = []
        memorize_weight = []
        memorize_col = []
        memorize_beam = []
        memorize_sum = []

        zhan,jia,qi=run_DNN_ind(num_var,num_room_type,x,labels)
        fit_all_best.append(fit_best)
draw_plot_picture(fit_all_best)
# draw_picture('name','title')