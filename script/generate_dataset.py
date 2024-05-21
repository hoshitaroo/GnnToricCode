'''
GNN decoder データ生成用コード
~/GNN_decoder で実行する
'''

'''
< 残している実装 >
マルチプロセスによるデータ生成処理
'''

import torch
from datetime import datetime
import os
import time

from toric_code import ToricCode
from param import param
from graph import generate_graphs
from graph import generate_graphs_multi


SIZE = 5                            # 符号距離
p_error = [0.01, 0.05, 0.10, 0.15]  # サンプリングする誤り確率
num_train_graphs = int(100000)         # 4で割り切れるように指定
num_test_graphs = 10000            # 4で割り切れるように指定
batch_size = 1000
NUM_CORE = 1
EPOCH = 400                        # 4の倍数を指定

custom = False

# データセットの生成の実行
if __name__ == '__main__' and not custom:
    
    size = SIZE
    param_1 = param(p_error[0], size)
    toric_code_1 = ToricCode(param=param_1)
    param_2 = param(p_error[1], size)
    toric_code_2 = ToricCode(param=param_2)
    param_3 = param(p_error[2], size)
    toric_code_3 = ToricCode(param=param_3)
    param_4 = param(p_error[3], size)
    toric_code_4 = ToricCode(param=param_4)

    now = datetime.now()
    now_f = now.strftime("%Y-%m-%d %H:%M:%S")
    print(now_f)
    # データセットの用意
    print('number of process is '+ str(NUM_CORE))
    if NUM_CORE == 1:
        train_data_list = generate_graphs(num_train_graphs, toric_code_1, toric_code_2, toric_code_3, toric_code_4)
        test_data_list = generate_graphs(num_test_graphs, toric_code_1, toric_code_2, toric_code_3, toric_code_4)
    else:
        train_data_list = generate_graphs_multi(num_train_graphs, toric_code_1, toric_code_2, toric_code_3, toric_code_4, num_core=NUM_CORE)
        test_data_list = generate_graphs_multi(num_test_graphs, toric_code_1, toric_code_2, toric_code_3, toric_code_4, num_core=NUM_CORE)
    # パスの指定と保存
    current_directory = os.getcwd()
    dataset_directory_path = os.path.join(current_directory, 'dataset')
    dataset_directory_path = os.path.join(dataset_directory_path, 'code_'+str(SIZE))
    os.chdir(dataset_directory_path)
    
    torch.save(train_data_list, 'train_data_list_0.pt')
    torch.save(test_data_list, 'test_data_list.pt')
    now = datetime.now()
    now_f = now.strftime("%Y-%m-%d %H:%M:%S")
    print(now_f)
    
    for i in range(EPOCH):
        print('part:' + str(i+1))
        befor = time.perf_counter()
        if i % 1 == 0 :  # 1エポックごとに実行 (適切な頻度で調整可能)
            num_replace = int(0.25 * num_train_graphs)  # 置き換えるデータ数 (25%)
            if NUM_CORE==1:
                new_data_list = generate_graphs(num_replace, toric_code_1, toric_code_2, toric_code_3, toric_code_4)
            else:
                new_data_list = generate_graphs_multi(num_replace, toric_code_1, toric_code_2, toric_code_3, toric_code_4, num_core=NUM_CORE)
            #train_data_list[:4*num_replace] = new_data_list
        torch.save(new_data_list, 'train_data_list_'+str(i+1)+'.pt')
        print(time.perf_counter() - befor)
    print('completed')

if custom:
    print('custom')

    size = SIZE
    param_1 = param(p_error[0], size)
    toric_code_1 = ToricCode(param=param_1)
    param_2 = param(p_error[1], size)
    toric_code_2 = ToricCode(param=param_2)
    param_3 = param(p_error[2], size)
    toric_code_3 = ToricCode(param=param_3)
    param_4 = param(p_error[3], size)
    toric_code_4 = ToricCode(param=param_4)

    start = 52
    end   = 200
    
    i = start+1
    current_directory = os.getcwd()
    dataset_directory_path = os.path.join(current_directory, 'dataset')
    dataset_directory_path = os.path.join(dataset_directory_path, 'code_'+str(SIZE))
    os.chdir(dataset_directory_path)
    train_data_list = torch.load('train_data_list_'+str(start)+'.pt')
    while i<=end:
        print('part:' + str(i))
        if i % 1 == 0 :  # 1エポックごとに実行 (適切な頻度で調整可能)
            num_replace = int(0.25 * num_train_graphs)  # 置き換えるデータ数 (25%)
            if NUM_CORE==1:
                new_data_list = generate_graphs(num_replace, toric_code_1, toric_code_2, toric_code_3, toric_code_4)
            else:
                new_data_list = generate_graphs_multi(num_replace, toric_code_1, toric_code_2, toric_code_3, toric_code_4, num_core=NUM_CORE)
            train_data_list[:4*num_replace] = new_data_list
        torch.save(train_data_list, 'train_data_list_'+str(i)+'.pt')
        i += 1
    print('completed')
