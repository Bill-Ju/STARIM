
import os
import sys
parent_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(parent_dir)

import numpy as np
import networkx as nx
from multiprocessing import Pool
import utils.graph_utils as graph_utils

pwd = '/home/zjy/project/STARIM'

seed_rates = [0.01,0.05,0.10,0.20,0.30]
mk_times = 1
num_process = 32
steps = 10
sample_num = num_process * 50
datasets = dict()
datasets['networks'] = ['ca-GrQc','Celegans','cora','ego-Facebook','fb-pages-food','USA airports','wiki-Vote']
datasets['undirect_net'] = [
    'ca-GrQc','Celegans','cora','ego-Facebook','fb-pages-food'
    ]
datasets['direct_net'] = [
    'wiki-Vote', 'congress-twitter', 'soc-dolphins'
]

directed = False

diff_model = 'IC'

if not directed:
    file_path_list = []
    for net_name in datasets['undirect_net']:
        file_path = pwd+f'/data/{net_name}/{net_name}.txt'        
        graph, N = graph_utils.read_graph(file_path, ind=0, directed=directed)
        with open(file_path, 'rb')as edges_f:
            G = nx.read_edgelist(edges_f, nodetype=int, create_using=nx.Graph(), encoding='latin1',
                                        data=(('weight', float),))
        
        
        if diff_model == 'SIR':
            # gamma_list = [0.2, 0.4, 0.6, 0.8]
            # gamma_list = [0.2, 0.8]
            gamma = 0.5  # 免疫率
            gamma_list = [gamma]
            def infected_beta(graph):
                # SIR模型的感染率设置
                degree = nx.degree(graph)
                degree_list = []
                degree_sq_list = []
                for i in degree:
                    degree_list.append(i[1])
                    degree_sq_list.append(i[1] * i[1])
                degree_avg = np.mean(degree_list)
                degree_avg_sq = np.mean(degree_sq_list)
                infected = degree_avg / (degree_avg_sq -  2 *degree_avg)
                return infected
            for gamma in gamma_list:
                beta_list = [0.1]                
                for beta in beta_list:
                    traj_all_list = []
                    for seed_rate in seed_rates:
                        seed_num = int(N* seed_rate)
                        seeds_list = []
                        for i in range(sample_num):
                            ind = np.random.permutation(N)
                            seeds = ind[:seed_num]
                            for j in range(mk_times):
                                seeds_list.append(seeds)
                        
                        task_num_per_pro = int(len(seeds_list)/num_process)
                        
                        with Pool(num_process) as p:   
                            traj_list = p.map(graph_utils.Worker_MC_SIR_Traj,
                                [[graph, seeds_list[task_num_per_pro*pro_index : task_num_per_pro*(pro_index+1)],beta,gamma, steps] for pro_index in range(num_process)])
                                                
                            traj_list = np.concatenate(traj_list, axis=0)
                            traj_all_list.append(traj_list)
                        print(f"{net_name} complete: seedrate{seed_rate},beta{beta},gamma{gamma}")
                    result_array = np.concatenate(traj_all_list, axis=0)
                    np.random.shuffle(result_array)
                    
                    save_path = pwd+f'/data/{net_name}/sir/{net_name}_sir_beta{beta}_gamma{gamma}.npy'
                    dir_name = os.path.dirname(save_path)
                    # 如果目录不存在，则创建目录
                    if not os.path.exists(dir_name):
                        os.makedirs(dir_name)
                    np.save(save_path, result_array)
        
        elif diff_model == 'IC':
            traj_all_list = []
            for seed_rate in seed_rates:
                seed_num = int(N* seed_rate)
                seeds_list = []
                for i in range(sample_num):
                    ind = np.random.permutation(N)
                    seeds = ind[:seed_num]
                    for j in range(mk_times):
                        seeds_list.append(seeds)
                
                task_num_per_pro = int(len(seeds_list)/num_process)
                
                with Pool(num_process) as p:   
                    traj_list = p.map(graph_utils.Worker_MC_IC_Traj,
                        [[graph, seeds_list[task_num_per_pro*pro_index : task_num_per_pro*(pro_index+1)], steps] for pro_index in range(num_process)])
                                        
                    traj_list = np.concatenate(traj_list, axis=0)
                    traj_all_list.append(traj_list)
                print(f"{net_name} complete: seedrate{seed_rate}")
            result_array = np.concatenate(traj_all_list, axis=0)
            np.random.shuffle(result_array)
            
            save_path = pwd+f'/data/{net_name}/ic/{net_name}_ic.npy'
            dir_name = os.path.dirname(save_path)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            np.save(save_path, result_array)
    
else:
    for net_name in datasets['direct_net']:
        file_path = pwd+f'/data/{net_name}/{net_name}.txt'  
        graph, N = graph_utils.read_graph(file_path, ind=0, directed=directed)
        if diff_model == 'IC':
            traj_all_list = []
            for seed_rate in seed_rates:
                seed_num = int(N* seed_rate)
                seeds_list = []
                for i in range(sample_num):
                    ind = np.random.permutation(N)
                    seeds = ind[:seed_num]
                    for j in range(mk_times):
                        seeds_list.append(seeds)
                
                task_num_per_pro = int(len(seeds_list)/num_process)
                
                with Pool(num_process) as p:   
                    traj_list = p.map(graph_utils.Worker_MC_IC_Traj,
                        [[graph, seeds_list[task_num_per_pro*pro_index : task_num_per_pro*(pro_index+1)], steps] for pro_index in range(num_process)])
                                        
                    traj_list = np.concatenate(traj_list, axis=0)
                    traj_all_list.append(traj_list)
                print(f"{net_name} complete: seedrate{seed_rate}")
            result_array = np.concatenate(traj_all_list, axis=0)
            np.random.shuffle(result_array)
            
            save_path = pwd+f'/data/{net_name}/ic/{net_name}_ic.npy'
            dir_name = os.path.dirname(save_path)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            np.save(save_path, result_array)
                    