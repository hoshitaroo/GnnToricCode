'''
学習済みモデルによる復号計算のシミュレーション用コード
~/GNN_decoder で実行する
'''

'''
< 現状 >

'''

import torch
from torch_geometric.data import Data
from tqdm import trange
import numpy as np
import time
import os

from toric_code import ToricCode
from param import param
from graph import errors_to_graph
from train_load_ver import GCNNet

device = 'cpu'

def evaluate(n_iter, size, error_rate):
    count = 0
    #符号距離に対応する学習済みモデルのロード
    current_directory = os.getcwd()
    model_directory_path = os.path.join(current_directory, 'learned_model')
    os.chdir(model_directory_path)
    gcn_net = GCNNet(size)
    gcn_net.load_state_dict(torch.load('gcn' + str(size) + '.pt'))
    # Param, Toric code をインスタンス化する
    param_sim = param(size=size, p=error_rate)
    toric_code = ToricCode(param_sim)

    spent_graph = 0
    spent_decode = 0
    count_no_error = 0
    for _ in trange(n_iter):
        #エラーを取得する
        errors = toric_code.generate_errors()
        is_no_error = np.all(errors == 0)
        if is_no_error:
            count_no_error += 1
            continue
        print(errors)
        #エラーからグラフを生成する
        before_graph = time.perf_counter()
        node_features, edge_index = errors_to_graph(toric_code, errors)
        spent_graph += time.perf_counter() - before_graph
        
        #グラフから復号マップを推論
        before_decode = time.perf_counter()
        data = Data(x=node_features, edge_index=edge_index)
        pred = gcn_net(data)
        spent_decode = time.perf_counter() - before_decode
        #復号操作
        map = pred.reshape((2*size, size))
        print(map)
        map = (map >= 0.5).int()
        print(map)
        errors = toric_code.operate_x(errors, map)
        #非自明なループとシンドローム測定の確認
        if np.all(toric_code.generate_syndrome_X(errors)==0) and np.all(toric_code.generate_syndrome_Z(errors)==0): 
            if toric_code.not_has_non_trivial_x(errors) and toric_code.not_has_non_trivial_z(errors):
                count = count + 1
    #論理エラーとなった割合をと各計算時間を出力
    print('generate graph : ' + str(spent_graph / n_iter) + str(" seconds"))
    print('decode         : ' + str(spent_decode / n_iter) + str(" seconds"))
    print(f"logical error rates: {n_iter - count_no_error - count}/{n_iter - count_no_error}", (n_iter - count_no_error - count) / (n_iter - count_no_error))

if  __name__ == '__main__':
    size = 5
    error_rate = 0.05
    print("Toric code simulation // code distance is " + str(size))
    print('error_rate = ' + str(error_rate))
    evaluate(1, size=size, error_rate=error_rate)