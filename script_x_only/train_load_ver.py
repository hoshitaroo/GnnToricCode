'''
GNN decoder 学習用コード
~/GNN_decoder で実行する
データセットは既に生成済みのものを使用する
'''

'''
< 残している実装 >
マルチプロセスによるデータ生成処理
'''

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch_geometric.nn import GCNConv, SAGEConv
import torch_geometric.data
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from datetime import datetime
import os
import time

from param import param
from toric_code import ToricCode

SIZE = 5                            # 符号距離
EPOCH = 50
batch_size = 100
num_replace = int(0.25 * 1000)
NUM_CORE = 1
lr = 1e-5

loss_fn = torch.nn.BCELoss()

def output_to_syndrome_z(toric_code, pred , num_chunks=batch_size, dim=0):
    #テンソルの形状を長方形に変形
    reshaped_pred = pred.view((2*SIZE*batch_size, SIZE))
    #テンソルをバッチサイズの数に分割する
    splited_tensors_list = torch.chunk(reshaped_pred, num_chunks, dim=dim)
    syndrome = torch.zeros((0,SIZE), requires_grad=True)
    for i in range(num_chunks):
        syndrome_z_tensor = toric_code.generate_syndrome_Z_tensor(splited_tensors_list[i])
        syndrome = torch.cat((syndrome, syndrome_z_tensor), dim=0)
    syndrome_flat = syndrome.flatten()
    return syndrome_flat

# ネットワークの定義
class GCNNet(torch.nn.Module):
    def __init__(self, code_distance):
        super(GCNNet, self).__init__()
        self.size = code_distance
        self.conv1 = SAGEConv(3,12)
        self.batch_normal_1 = nn.BatchNorm1d(12)
        self.conv2 = SAGEConv(12,12)
        self.batch_normal_2 = nn.BatchNorm1d(12)
        self.conv3 = SAGEConv(12,12)
        self.batch_normal_3 = nn.BatchNorm1d(12)
        self.conv4 = SAGEConv(12,12)
        self.batch_normal_4 = nn.BatchNorm1d(12)
        self.conv5 = SAGEConv(12,12)
        self.batch_normal_5 = nn.BatchNorm1d(12)
        self.conv6 = SAGEConv(12,12)
        self.batch_normal_6 = nn.BatchNorm1d(12)
        self.conv7 = SAGEConv(12,12)
        self.batch_normal_7 = nn.BatchNorm1d(12)
        self.conv8 = SAGEConv(12,12)
        self.activate = nn.SELU()
        self.activate_relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.linear = nn.Linear(6, 6)
        self.linear_out = nn.Linear(6, 1)
        self.batch_normal = nn.BatchNorm1d(6)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        #グラフ畳み込みによる処理
        x = self.conv1(x, edge_index)
        x = self.activate_relu(x)
        x = self.batch_normal_1(x)
        x = self.conv2(x, edge_index)
        x = self.activate_relu(x)
        x = self.batch_normal_2(x)
        x = self.conv3(x, edge_index)
        x = self.activate_relu(x)
        x = self.batch_normal_3(x)
        x = self.conv4(x, edge_index)
        x = self.activate_relu(x)
        x = self.batch_normal_4(x)
        x = self.conv5(x, edge_index)
        x = self.activate_relu(x)
        x = self.batch_normal_5(x)
        x = self.conv6(x, edge_index)
        x = self.activate_relu(x)
        x = self.batch_normal_6(x)
        x = self.conv7(x, edge_index)
        x = self.activate_relu(x)
        x = self.batch_normal_7(x)
        x = self.conv8(x, edge_index)
        x = self.activate_relu(x)
        #形状の変更
        x = x.view(-1, 6)
        #線型層による処理
        x = self.dropout(x)
        x = self.linear(x)
        x = self.batch_normal(x)
        x = self.activate_relu(x)
        x = self.linear(x)
        x = self.batch_normal(x)
        x = self.activate_relu(x)
        x = self.linear(x)
        x = self.batch_normal(x)
        x = self.activate_relu(x)
        x = self.linear_out(x)
        x = self.sigmoid(x)
        return x

#自作データセットクラス
class CustomDataset(Dataset):
    def __init__(self, data_list):
        super(CustomDataset, self).__init__()
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

# データセットの用意とモデルの学習の実行
if __name__ == '__main__':
    
    size = SIZE
    p = param(p=0.05, size=SIZE)
    train_toric_code = ToricCode(param=p)
    # データセットのロード
    current_directory = os.getcwd()
    dataset_directory_path = os.path.join(current_directory, 'dataset')
    dataset_directory_path = os.path.join(dataset_directory_path, 'code_'+str(SIZE))
    os.chdir(dataset_directory_path)
    train_data_list = torch.load('train_data_list_0.pt')
    test_data_list = torch.load('test_data_list.pt')
    train_dataset = CustomDataset(train_data_list)
    train_batch = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_CORE)
    test_dataset = CustomDataset(test_data_list)
    test_batch = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_CORE)

    # デバイスの設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device is ' + str(device))

    #ネットワークのロード
    gcn_net = GCNNet(size).to(device)

    # パラメータカウント
    '''params = 0
    for p in gcn_net.parameters():
        if p.requires_grad:
            params += p.numel()
        
    print(params)  # 121898'''

    # train_datasetの中身を確認
    '''print(len(train_data_list))
    print(train_data_list[0].x)
    print(train_data_list[1].x)
    print(train_data_list[2].x)
    print(train_data_list[3].x)'''

    '''
    print("訓練用データセットの確認")
    node_features, edge_index , y= train_dataset[0]
    print('node_feature')
    print(node_features)
    print('edge_index')
    print(edge_index)
    print('y')
    print(y)
    '''

    #訓練用ミニバッチデータセットの確認
    '''
    print("訓練用ミニバッチデータセットの確認")
    for batch in train_batch:
        print(batch.x)
        print(batch.edge_index)
        print(batch.y)
        print(batch)
        break
    '''

    # test_datasetの中身を確認
    '''
    print("テスト用データセットの確認")
    node_features, edge_index , y= test_dataset[0]
    print('node_feature')
    print(node_features)
    print('edge_index')
    print(edge_index)
    print('y')
    print(y)
    '''

    #テスト用ミニバッチデータセットの確認
    '''
    print("テスト用ミニバッチデータセットの確認")
    for batch in test_batch:
        print(batch.x)
        print(batch.edge_index)
        print(batch.y)
        print(batch)
        break
    '''

    #損失関数と最適化関数の定義
    criterion = loss_fn
    def compute_loss(x, y):
        return criterion(x, y)
    zero_syndrome = torch.zeros((SIZE*SIZE*batch_size), requires_grad=True)
    optimizer_gcn = optim.AdamW(gcn_net.parameters(), lr=lr)

    param_ = param(size=SIZE, p=0.05)
    toric_code = ToricCode(param_)

    #各値を格納するリストの生成
    train_loss_list_gcn = []
    train_accyracy_list = []
    test_loss_list_gcn = []
    test_accuracy_list = []

    #エポックの実行
    now = datetime.now()
    now_f = now.strftime("%Y-%m-%d %H:%M:%S")
    print(now_f)
    print('--------------------------------------------------------')
    print('start training')
    print('--------------------------------------------------------')
    epoch = EPOCH
    for i in range(epoch):
        before = time.perf_counter()
        print('-----------------------')
        print("Epoch: {}/{}".format(i+1, epoch))

        train_loss_gcn = 0
        train_accuracy = 0
        test_loss_gcn = 0
        test_accuracy = 0

        gcn_net.train()
        zero_syndrome = zero_syndrome.to(device)

        for batch in train_batch:
            data = batch.to(device)
            y = data.y
            # GCNレイヤーによる計算
            pred = gcn_net(data)
            y = y.flatten()
            pred = pred.flatten()
            # syndromeで損失を計算する場合の処理↓
            '''pred = torch.abs(torch.sub(pred, y))
            pred_syndrome = output_to_syndrome_z(toric_code= train_toric_code,pred = pred,num_chunks=batch_size,dim=0).to(device)
            pred_syndrome = pred_syndrome.flatten()
            loss = compute_loss(pred_syndrome, zero_syndrome)'''
            #エラーで損失を計算する場合
            loss = compute_loss(pred, y)

            # 各層の最適化の実行
            optimizer_gcn.zero_grad()
            loss.backward()
            optimizer_gcn.step()

            train_loss_gcn += loss.item()

        if i % 1 == 0:  # 1エポックごとに実行 (適切な頻度で調整可能)
            new_data_list = torch.load('train_data_list_'+str(i+1)+'.pt')
            train_data_list[:4*num_replace] = new_data_list
            # データセットオブジェクトを再構築
            train_dataset = CustomDataset(train_data_list)
            train_batch = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_CORE)
        
        epoch_train_loss_gcn = train_loss_gcn / len(train_batch)

        gcn_net.eval()
        with torch.no_grad():
            for batch in test_batch:
                data = batch.to(device)
                y = data.y

                pred = gcn_net(data)
                y = y.flatten()
                pred = pred.flatten()
                # syndromeで損失を計算する場合の処理↓
                '''pred = torch.abs(torch.sub(pred, y))
                pred_syndrome = output_to_syndrome_z(toric_code= train_toric_code,pred = pred,num_chunks=batch_size,dim=0).to(device)
                pred_syndrome = pred_syndrome.flatten()
                loss = compute_loss(pred_syndrome, zero_syndrome)'''
                #エラーで損失を計算する場合
                loss = compute_loss(pred, y)
                test_loss_gcn += loss.item()
        
        epoch_test_loss_gcn = test_loss_gcn / len(test_batch)

        print("Train_Loss_gcn   : {:.10f}".format(epoch_train_loss_gcn))
        print("Test_Loss_gcn    : {:.10f}".format(epoch_test_loss_gcn))
        spent = time.perf_counter() - before
        formatted_time = "{:.3f}".format(spent)
        print(f"time             : {formatted_time} seconds")

        train_loss_list_gcn.append(epoch_train_loss_gcn)
        test_loss_list_gcn.append(epoch_test_loss_gcn)
    
    now = datetime.now()
    now_f = now.strftime("%Y-%m-%d %H:%M:%S")
    print(now_f)
    
    current_directory = os.path.abspath(os.path.dirname(__file__))
    parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
    parent_directory = os.path.abspath(os.path.join(parent_directory, os.pardir))
    parent_directory = os.path.abspath(os.path.join(parent_directory, os.pardir))
    os.chdir(parent_directory)

    current_directory = os.getcwd()
    model_directory_path = os.path.join(current_directory, 'learned_model')
    os.chdir(model_directory_path)
    torch.save(gcn_net.state_dict(), 'gcn' + str(size) + '.pt')
    
    # 学習の結果の可視化
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    png_directory_path = os.path.join(parent_directory, 'png')
    os.chdir(png_directory_path)

    x = list(range(epoch))
    y_train = train_loss_list_gcn
    y_test = test_loss_list_gcn
    plt.figure()
    plt.title('GCN_linear loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(x, y_train, label='train')
    plt.plot(x, y_test, label='test')
    plt.legend()
    plt.savefig('gcn_linear_loss')