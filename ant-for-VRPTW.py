# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 13:51:01 2018
@author: tianz
"""

import numpy as np
import random
import copy
import math
import time

#读入数据
data = np.loadtxt('data.txt').astype(np.int64)
#data = [[0, 3, 3, 0, 0, 999, 0], 
#        [1, 2, 2, 10, 10, 20, 10], 
#        [2, 4, 1, 10, 20, 30, 10]]

#参数
tc = 1#单位运输成本
weight = 200#车辆载重
em = 2000#空车重
cr = 3.0959#燃油转化为碳排放的转化率
speed = 1#车辆速度 公里每分钟
discount_rate = 0.1#超时折扣率
truck_price = 200#卡车价格
truck_limit = 37#卡车数量限制
soft_time = 20#软时间窗允许超过的时间

#图基本属性
#商家数
num_node = len(data)
#商家之间距离
dis = [[0] * (num_node) for row in range(num_node)]
for i in data:
    for j in data:
        dis[i[0]][j[0]] = ((i[1]-j[1])**2+(i[2]-j[2])**2)**0.5
#蚁群优化求解
max_t = 2#最大迭代次数
m = 2#蚂蚁个数
a = 1#伪随机比例行动选择规则权重因子（信息素）
b = 2#伪随机比例行动选择规则权重因子（启发式信息）
p = 0.8#挥发系数
q0 = 0.5#伪随机比例行动选择规则参数
T_max = 6.0#信息素最大值
T_min = 0.001#信息素最小值
Ph_matrix = [[T_max] * (num_node) for row in range(num_node)]#信息素矩阵
for i in range(num_node):
    Ph_matrix[i][i] = 0
best_result_matrix = []#最优解

#轮盘赌选点
def roulette(probability_table):
    probability_table = copy.deepcopy(probability_table)
    for i in range(1,len(probability_table)):#累加概率
        probability_table[i] = probability_table[i-1]+probability_table[i]
    tmp = random.uniform(0,max(probability_table))#生成随机数
    if tmp <= probability_table[1]:#进行轮盘赌
        return 1
    elif tmp > probability_table[-2]:
        return len(probability_table)-1
    else:
        for i in range(1,len(probability_table)):
            if tmp > probability_table[i] and tmp <= probability_table[i+1]:
                return i+1
#启发式信息
def get_heuristic(i, j, data, dis):
    Q = [x[3] for x in data]
    c = dis[i]
    l = [x[5] for x in data]
    heuristic = 3 - (Q[j] - min(Q)) / (max(Q) - min(Q)) - (c[j] - min(c)) / (max(c) - min(c)) - (l[j] - min(l)) / (max(l) - min(l))
    return heuristic
#蚂蚁移动到下一节点
def get_next_node(current_node, candidate, Ph_matrix, dis, data, num_node, a, b, q0):
    #相邻边信息素
    tmp1 = [0 for i in range(num_node)]
    for i in list(candidate):
        tmp1[i] = Ph_matrix[current_node][i]
    #启发式信息
    tmp2 = [0 for i in range(num_node)]
    for i in list(candidate):
        tmp2[i] = get_heuristic(current_node, i, data, dis)
#    print(candidate)
#    print(tmp1)
#    print(a)
#    print(tmp2)
    p_next_node = list(map(lambda x,y:x**a * y**b/(sum(list(map(lambda x,y:x**a*y**b, tmp1, tmp2)))), tmp1, tmp2))#按行动准则计算移动到某个节点的概率
    q = random.uniform(0,1)
    if q <= q0:#按照q0接受最优节点，否则轮盘赌
        next_node = np.argmax(p_next_node)
    else:
        next_node = roulette(p_next_node)
    return next_node
#    return roulette(p_next_node)

#约束条件判断
def get_candidate(current_node, current_time, tabu_list, dis, data, load, speed):
    candidate = []
    candidate_tmp = list(set(range(1, len(data))) - set(tabu_list))
    for i in candidate_tmp:
        if load >= data[i][3]:#满足需求
            if (dis[current_node][i] + dis[i][0])/speed + data[i][6] + current_time <= data[0][5]:#能返回仓库
                if (dis[current_node][i] / speed + current_time) <= min(data[i][5] + soft_time, data[0][5]):#满足硬时间窗
                    candidate.append(i)
    return candidate
#成本计算
def F(tc, dis, route_matrix, num_truck, truck_price):
    F = num_truck * truck_price
    for i in range(len(dis)):
        for j in range(len(dis)):
            F = F + (tc * dis[i][j] * route_matrix[i][j])
    return F
#服务质量计算
def S():
    pass
#碳排放计算
def C(f_matrix, em, cr, dis, route_matrix):
    C = 0
    for i in range(len(dis)):
        for j in range(len(dis)):
            C = C + ((185.049 + 0.00149 * (em + f_matrix[i][j]) * route_matrix[i][j]) * dis[i][j] * cr) / 44220
    return C
#判断是否非劣解
def pareto(x, y, matrix):#1则加入
    flag = 1
    if matrix != []:
        for i in matrix:
            if ((i[3] < x) & (i[4] <= y)) | ((i[3] <= x) & (i[4] < y)):
                flag = 0
                break
    return flag
    
#信息素更新
def update_pheromone(Ph_matrix, p, T_max, T_min, tc, dis, route_matrix, f_matrix, em, cr, local_best_result_matrix, local_max_F, local_min_F, local_max_C, local_min_C):#信息素更新
    Ph_matrix = np.matrix(Ph_matrix)
    Ph_matrix = Ph_matrix*(1 - p)#信息素挥发
    for i in local_best_result_matrix:
        delta = 2 - (i[3] - local_min_F) / (local_max_F - local_min_F) - (i[4] - local_min_C) / (local_max_C - local_min_C)
        for i1 in range(len(Ph_matrix)):
            for i2 in range(len(Ph_matrix)):
                if i[0][i1][i2] == 1:
                    Ph_matrix[[i1, i2]] = Ph_matrix[[i1, i2]] + delta
    mask1 = Ph_matrix <= T_min#限制信息素范围
    Ph_matrix[mask1] = T_min
    mask2 = Ph_matrix >= T_max
    Ph_matrix[mask2] = T_max
    Ph_matrix = Ph_matrix.tolist()
    return Ph_matrix
#主函数
for t in range(0, max_t):
    local_best_result_matrix = []#局部最优解
    local_min_F = local_min_C = float('inf')
    local_max_F = local_max_C = 0
    for ant in range(0, m):
        num_truck = 1#车辆数初始化为1
        total_time = 0#车辆总运行时间
        load = weight#初始车载重为最大载重
        candidate = range(1, num_node)#候选节点为1——100
        tabu_list = [0]#禁忌列表初始化为0
        current_node = 0#初始节点为0
        current_time = 0#初始化时间为零
        route_matrix = [[0] * (num_node) for row in range(num_node)]#路径矩阵
        arrival_time = [0] * (num_node)#到达时间
        f_matrix = [[0] * (num_node) for row in range(num_node)]#载重矩阵
        trajectory = [0]#轨迹
#        while (len(tabu_list) != len(data)) & (current_time <= data[0][5]):#存在未遍历节点，并且未超时
        while (len(tabu_list) != len(data)):#存在未遍历节点
#            print(t, ant)
#            print(num_truck)
            if current_time + dis[current_node][0]/speed > data[0][5]:#超时则寻找新解
                num_truck = 1#车辆数初始化为1
                total_time = 0#车辆总运行时间
                load = weight#初始车载重为最大载重
                candidate = range(1, num_node)#候选节点为1——100
                tabu_list = [0]#禁忌列表初始化为0
                current_node = 0#初始节点为0
                current_time = 0#初始化时间为零
                route_matrix = [[0] * (num_node) for row in range(num_node)]#路径矩阵
                arrival_time = [0] * (num_node)#到达时间
                f_matrix = [[0] * (num_node) for row in range(num_node)]#载重矩阵
                trajectory = [0]#轨迹
                continue
            candidate = get_candidate(current_node, current_time, tabu_list, dis, data, load, speed)
            if candidate != []:#存在可行节点
#                print(current_node)
                last_node = current_node
                current_node = get_next_node(current_node, candidate, Ph_matrix, dis, data, num_node, a, b, q0)
                trajectory.append(current_node)
                if (dis[current_node][last_node] / speed + current_time) <= data[current_node][4]:#等待节点开始服务
                   current_time = data[current_node][4]
#                print(last_node)
                f_matrix[last_node][current_node] = f_matrix[current_node][last_node] = load
                load = load - data[current_node][3]
                current_time = current_time + dis[current_node][last_node]/speed + data[current_node][6]
                arrival_time[current_node] = current_time
                tabu_list.append(current_node)
                tabu_list = list(set(tabu_list))
                route_matrix[last_node][current_node] = 1
            else:#返回仓库
                last_node = current_node
                current_node = 0
                trajectory.append(current_node)
                f_matrix[last_node][current_node] = f_matrix[current_node][last_node] = load
                load = weight
                current_time = current_time + dis[current_node][last_node]/speed + data[current_node][6]
                route_matrix[last_node][current_node] = 1
                candidate = get_candidate(current_node, current_time, tabu_list, dis, data, load, speed)
                if candidate != []:#存在可行节点，不换车
                    last_node = current_node
                    current_node = get_next_node(current_node, candidate, Ph_matrix, dis, data, num_node, a, b, q0)
                    trajectory.append(current_node)
                    if (dis[current_node][last_node] / speed + current_time) <= data[current_node][4]:#等待节点开始服务
                       current_time = data[current_node][4]
                    f_matrix[last_node][current_node] = f_matrix[current_node][last_node] = load
                    load = load - data[current_node][3]
                    current_time = current_time + dis[current_node][last_node]/speed + data[current_node][6]
                    arrival_time[current_node] = current_time
                    tabu_list.append(current_node)
                    tabu_list = list(set(tabu_list))
                    route_matrix[last_node][current_node] = 1
                else:#会仓库换车
                    total_time = total_time + current_time#车辆总运行时间
                    current_time = 0
                    num_truck = num_truck + 1
                    if num_truck > truck_limit:#车辆加限制参数超过则放弃当前解
                        num_truck = 1#车辆数初始化为1
                        total_time = 0#车辆总运行时间
                        load = weight#初始车载重为最大载重
                        candidate = range(1, num_node)#候选节点为1——100
                        tabu_list = [0]#禁忌列表初始化为0
                        current_node = 0#初始节点为0
                        current_time = 0#初始化时间为零
                        route_matrix = [[0] * (num_node) for row in range(num_node)]#路径矩阵
                        arrival_time = [0] * (num_node)#到达时间
                        f_matrix = [[0] * (num_node) for row in range(num_node)]#载重矩阵
                        trajectory = [0]#轨迹
                        continue
                    trajectory.append('@')
                    candidate = get_candidate(current_node, current_time, tabu_list, dis, data, load, speed)
                    last_node = current_node
                    current_node = get_next_node(current_node, candidate, Ph_matrix, dis, data, num_node, a, b, q0)
                    trajectory.append(current_node)
                    if (dis[current_node][last_node] / speed + current_time) <= data[current_node][4]:#等待节点开始服务
                       current_time = data[current_node][4]
                    f_matrix[last_node][current_node] = f_matrix[current_node][last_node] = load
                    load = load - data[current_node][3]
                    current_time = current_time + dis[current_node][last_node]/speed + data[current_node][6]
                    arrival_time[current_node] = current_time
                    tabu_list.append(current_node)
                    tabu_list = list(set(tabu_list))
                    route_matrix[last_node][current_node] = 1
        print(t, ant)
        last_node = current_node#返回仓库
        current_node = 0
        trajectory.append(current_node)
        f_matrix[last_node][current_node] = f_matrix[current_node][last_node] = load
        current_time = current_time + dis[current_node][last_node]/speed + data[current_node][6]
        total_time = total_time + current_time#车辆总运行时间
        route_matrix[last_node][current_node] = 1
        F_tmp = F(tc, dis, route_matrix, num_truck, truck_price)
        C_tmp = C(f_matrix, em, cr, dis, route_matrix)
#        print(F_tmp, C_tmp)
        if F_tmp > local_max_F:
            local_max_F = F_tmp
        if F_tmp < local_min_F:
            local_min_F = F_tmp
        if C_tmp > local_max_C:
            local_max_C = C_tmp
        if C_tmp < local_min_C:
            local_min_C = C_tmp
#        print(pareto(F_tmp, C_tmp, local_best_result_matrix))
        if pareto(F_tmp, C_tmp, local_best_result_matrix) == 1:#若为帕累托最优解则加入local_best_result_matrix
            local_best_result_matrix.append([route_matrix, trajectory, arrival_time, F_tmp, C_tmp, total_time])
        for i in range(len(local_best_result_matrix) - 1, -1, -1):
            if pareto(local_best_result_matrix[i][3], local_best_result_matrix[i][4], local_best_result_matrix) == 0:
                local_best_result_matrix.pop(i)
    #更新信息素 *分母为零
    Ph_matrix = update_pheromone(Ph_matrix, p, T_max, T_min, tc, dis, route_matrix, f_matrix, em, cr, local_best_result_matrix, local_max_F, local_min_F, local_max_C, local_min_C)
    #更新最优解集
    if best_result_matrix == []:
        for i in local_best_result_matrix:
            best_result_matrix.append(i)
    else:
        for i in local_best_result_matrix:
            if pareto(i[3], i[4], best_result_matrix) == 1:
                best_result_matrix.append(i)
        for i in range(len(best_result_matrix) - 1, -1, -1):
            if pareto(best_result_matrix[i][3], best_result_matrix[i][4], best_result_matrix) == 0:
                best_result_matrix.pop(i)
    
print('\a')
