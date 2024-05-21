'''
データセット生成用の関数
'''

import torch
from torch_geometric.data import Data
import numpy as np
import os
import random
from multiprocessing import Pool

# エラーからグラフを表す行列を生成
def errors_to_graph(toric_code, errors):
    node_features = torch.zeros(0,4)
    #エラーからシンドロームを生成
    syndrome_x = toric_code.generate_syndrome_X(errors)
    syndrome_z = toric_code.generate_syndrome_Z(errors)
    #シンドロームからグラフのノードの特徴を表す行列を生成
    for i in range(toric_code.size):
        for j in range(toric_code.size):
            if syndrome_x[i][j] == 1:
                node_features = torch.cat((node_features, torch.tensor([[100,0,i,j]])), dim=0)
            else:
                node_features = torch.cat((node_features, torch.tensor([[-100,0,i,j]])), dim=0)
    for i in range(toric_code.size):
        for j in range(toric_code.size):
            if syndrome_z[i][j] == 1:
                node_features = torch.cat((node_features, torch.tensor([[0,100,i,j]])), dim=0)
            else:
                node_features = torch.cat((node_features, torch.tensor([[0,-100,i,j]])), dim=0)
    num_nodes = node_features.shape[0]
    
    #print(node_features)   # for debug

    #グラフのエッジの接続を表す行列を生成(完全グラフ)
    num_nodes = node_features.shape[0]
    syndrm_list = []
    for i in range(num_nodes):
        if node_features[i][0] == 100 or node_features[i][1] == 100:
            syndrm_list.append(i)
    edge_index = torch.zeros(2,len(syndrm_list)*(len(syndrm_list) - 1), dtype=int)
    j, index = 0, 0
    while j < len(syndrm_list):
        k = 0
        while k < (len(syndrm_list)-1):
            edge_index[0][index] = syndrm_list[j]
            k += 1
            index += 1
        j += 1
    j, index = 0, 0
    while j < len(syndrm_list):
        k = 0
        while k < len(syndrm_list):
            if j == k:
                k += 1
                continue
            else:
                edge_index[1][index] = syndrm_list[k]
            k += 1
            index += 1
        j += 1
    
    node_features = node_features
    edge_index = edge_index
    return node_features, edge_index

def generate_graph_list(num_grphs, toric_code):
    graphs = []
    i = 0
    while i < num_grphs:
        errors = toric_code.generate_errors()
        if np.all(errors == 0):
            continue
        node_features, edge_index = errors_to_graph(toric_code=toric_code, errors=errors)
        errors_x, errors_z = toric_code.errors_to_errorsXZ(errors)
        errors_x_tensor = torch.tensor(errors_x)
        errors_z_tensor = torch.tensor(errors_z)
        syndrome_x = toric_code.generate_syndrome_Z_tensor(errors_x_tensor)
        syndrome_z = toric_code.generate_syndrome_X_tensor(errors_z_tensor)
        y=torch.zeros((2, toric_code.size, toric_code.size))
        y[0] = y[0] + syndrome_x
        y[1] = y[1] + syndrome_z
        graph = Data(x=node_features, edge_index=edge_index, y=y)
        graphs.append(graph)
        i += 1
    return graphs

#単一のプロセスによる生成を行う場合
def generate_graphs(num_grphs, toric_code_1, toric_code_2, toric_code_3, toric_code_4):
    graphs_1 = generate_graph_list(num_grphs=num_grphs, toric_code=toric_code_1)
    graphs_2 = generate_graph_list(num_grphs=num_grphs, toric_code=toric_code_2)
    graphs_3 = generate_graph_list(num_grphs=num_grphs, toric_code=toric_code_3)
    graphs_4 = generate_graph_list(num_grphs=num_grphs, toric_code=toric_code_4)
    graphs = graphs_1 + graphs_2 + graphs_3 + graphs_4
    return graphs

#マルチプロセスによる生成を行う場合
def parallel_generate_graphs(argument):
    num_graphs = argument[0]
    toric_code = argument[1]
    graphs = generate_graph_list(num_grphs=num_graphs, toric_code=toric_code)
    return graphs

def generate_graphs_multi(num_graphs, toric_code_1, toric_code_2, toric_code_3, toric_code_4, num_core):
    toric_list = [toric_code_1, toric_code_2, toric_code_3, toric_code_4]
    pool = Pool(processes=num_core)
    arguments = []
    per_toric_code = int(num_core/4)
    i=0
    while i < 4:
        j=0
        while j < per_toric_code:
            argument = [4*int(num_graphs/num_core), toric_list[i]]
            arguments.append(argument)
            j += 1
        i += 1
    results = pool.map(parallel_generate_graphs, arguments)
    graphs = []
    i=0
    while i < num_core:
        graphs = graphs + results[i]
        i += 1
    return graphs
