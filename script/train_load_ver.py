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
EPOCH = 200
batch_size = 100
num_replace = int(0.25 * 10000)
NUM_CORE = 1
lr = 1e-5

loss_fn = torch.nn.MSELoss()

def output_to_syndrome_x(toric_code, pred , num_chunks=batch_size, dim=0):
    #テンソルの形状を長方形に変形
    reshaped_pred = pred.view((2*SIZE*batch_size, SIZE))
    #テンソルをバッチサイズの数に分割する
    splited_tensors_list = torch.chunk(reshaped_pred, num_chunks, dim=dim)
    syndrome = torch.zeros((0,SIZE), requires_grad=True)
    for i in range(num_chunks):
        syndrome_x_tensor = toric_code.generate_syndrome_X_tensor(splited_tensors_list[i])
        syndrome = torch.cat((syndrome, syndrome_x_tensor), dim=0)
    syndrome_flat = syndrome.flatten()
    return syndrome_flat

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
        self.conv1 = SAGEConv(4,16)
        self.batch_normal_1 = nn.BatchNorm1d(16)
        self.conv2 = SAGEConv(16, 16)
        self.batch_normal_2 = nn.BatchNorm1d(16)
        self.conv3 = SAGEConv(16,16)
        self.batch_normal_3 = nn.BatchNorm1d(16)
        self.conv4 = SAGEConv(16,16)
        self.batch_normal_4 = nn.BatchNorm1d(16)
        self.conv5 = SAGEConv(16,16)
        self.batch_normal_5 = nn.BatchNorm1d(16)
        self.conv6 = SAGEConv(16,16)
        self.batch_normal_6 = nn.BatchNorm1d(16)
        self.conv7 = SAGEConv(16,16)
        self.batch_normal_7 = nn.BatchNorm1d(16)
        self.conv8 = SAGEConv(16,16)
        self.activate = nn.SELU()
        self.activate_relu = nn.ReLU()
        '''
        torch.nn.init.ones_(self.conv1.lin.weight)
        torch.nn.init.ones_(self.conv2.lin.weight)
        torch.nn.init.ones_(self.conv3.lin.weight)
        torch.nn.init.ones_(self.conv4.lin.weight)
        torch.nn.init.ones_(self.conv5.lin.weight)
        torch.nn.init.ones_(self.conv6.lin.weight)
        torch.nn.init.ones_(self.conv7.lin.weight)'''
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
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
        high_d_feature = self.activate_relu(x)
        return high_d_feature

def init_ones(model):
    for layer in model:
        if isinstance(layer, nn.Linear):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)


class Linear_x(nn.Module):
    def __init__(self, code_distance):
        super(Linear_x, self).__init__()
        self.size = code_distance
        self.x_linear = nn.Sequential(
            nn.Dropout(),
            nn.Linear(16, 16),
            nn.SELU(),
            nn.BatchNorm1d(16),
            nn.Linear(16,16),
            nn.SELU(),
            nn.BatchNorm1d(16),
            nn.Linear(16,16),
            nn.SELU(),
            nn.BatchNorm1d(16),
            nn.Linear(16, 1)
        )
        self.activate = nn.Sigmoid()
        '''
            nn.SELU(),
            nn.Linear(25, 20),
            nn.SELU(),
            nn.Linear(20, 15),
            nn.SELU(),
            nn.Linear(15, 10),
            nn.SELU(),
            nn.Linear(10,5),
            nn.SELU(),
            nn.Linear(5,1),
            nn.Sigmoid()
        '''
        #init_ones(self.x_linear)
    
    def forward(self, data):
        x = self.x_linear(data)
        x = self.activate(x)
        return x

class Linear_z(nn.Module):
    def __init__(self, code_distance):
        super(Linear_z, self).__init__()
        self.size = code_distance
        self.z_linear = nn.Sequential(
            nn.Dropout(),
            nn.Linear(16, 16),
            nn.SELU(),
            nn.BatchNorm1d(16),
            nn.Linear(16,16),
            nn.SELU(),
            nn.BatchNorm1d(16),
            nn.Linear(16,16),
            nn.SELU(),
            nn.BatchNorm1d(16),
            nn.Linear(16, 1)
        )
        self.activate = nn.Sigmoid()
        '''
            nn.SELU(),
            nn.Linear(25, 20),
            nn.SELU(),
            nn.Linear(20, 15),
            nn.SELU(),
            nn.Linear(15, 10),
            nn.SELU(),
            nn.Linear(10,5),
            nn.SELU(),
            nn.Linear(5,1),
            nn.Sigmoid()
        '''
        #init_ones(self.z_linear)
    
    def forward(self, data):
        x = self.z_linear(data)
        x = self.activate(x)
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
    Lx_net = Linear_x(size).to(device)
    Lz_net = Linear_z(size).to(device)

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
    optimizer_x = optim.AdamW(Lx_net.parameters(), lr=lr)
    optimizer_z = optim.AdamW(Lz_net.parameters(), lr=lr)
    optimizer_gcn = optim.AdamW(gcn_net.parameters(), lr=lr)

    param_ = param(size=SIZE, p=0.05)
    toric_code = ToricCode(param_)

    #各値を格納するリストの生成
    train_loss_list_x = []
    train_loss_list_z = []
    train_loss_list_gcn = []
    train_accyracy_list = []
    test_loss_list_x = []
    test_loss_list_z = []
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

        train_loss_x = 0
        train_loss_z = 0
        train_loss_gcn = 0
        train_accuracy = 0
        test_loss_x = 0
        test_loss_z= 0
        test_loss_gcn = 0
        test_accuracy = 0

        Lx_net.train()
        Lz_net.train()
        gcn_net.train()

        zero_syndrome = zero_syndrome.to(device)

        for batch in train_batch:
            data = batch.to(device)
            y = data.y
            #print(y.shape)
            # GCNレイヤーによる計算
            high_d_feature = gcn_net(data)

            # X線形層による計算と損失の計算
            x_pred_prob = Lx_net(high_d_feature)
            x_pred_syndrome = output_to_syndrome_z(toric_code= train_toric_code,pred = x_pred_prob,num_chunks=batch_size,dim=0).to(device)
            x_pred_syndrome = x_pred_syndrome.flatten()
            '''y_x = torch.zeros((0, 5)).to(device)
            j = 0
            while j< batch_size:
                y_x = torch.cat((y_x, y[2*j]), dim=0)
                j += 1
            y_x = y_x.view((-1, 1))'''
            loss_x = compute_loss(x_pred_syndrome, zero_syndrome)
            
            # Z線型層による計算と損失の計算
            z_pred_prob = Lz_net(high_d_feature)
            z_pred_syndrome = output_to_syndrome_x(toric_code= train_toric_code,pred = z_pred_prob,num_chunks=batch_size,dim=0).to(device)
            z_pred_syndrome = z_pred_syndrome.flatten()
            '''y_z = torch.zeros((0, 5)).to(device)
            j = 0
            while j<batch_size:
                y_z = torch.cat((y_z, y[2*j+1]), dim=0)
                j += 1
            y_z = y_z.view((-1, 1))'''
            loss_z = compute_loss(z_pred_syndrome, zero_syndrome)

            # GCNレイヤーの損失の計算
            loss_gcn = loss_x + loss_z

            # 各層の最適化の実行
            optimizer_gcn.zero_grad()
            optimizer_x.zero_grad()
            optimizer_z.zero_grad()
            loss_gcn.backward(retain_graph=True)
            loss_z.backward(retain_graph=True)
            loss_x.backward()
            optimizer_gcn.step()
            optimizer_z.step()
            optimizer_x.step()

            train_loss_x += loss_x.item()
            train_loss_z += loss_z.item()
            train_loss_gcn += loss_gcn.item()
        
        if i % 1 == 0:  # 1エポックごとに実行 (適切な頻度で調整可能)
            new_data_list = torch.load('train_data_list_'+str(i+1)+'.pt')
            train_data_list[:4*num_replace] = new_data_list
            # データセットオブジェクトを再構築
            train_dataset = CustomDataset(train_data_list)
            train_batch = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_CORE)

        epoch_train_loss_x = train_loss_x / len(train_batch)
        epoch_train_loss_z = train_loss_z / len(train_batch)
        epoch_train_loss_gcn = train_loss_gcn / len(train_batch)

        Lx_net.eval()
        Lz_net.eval()
        gcn_net.eval()
        with torch.no_grad():
            for batch in test_batch:
                data = batch.to(device)
                y = data.y

                high_d_feature = gcn_net(data)
                x_pred_prob = Lx_net(high_d_feature)
                x_pred_syndrome = output_to_syndrome_z(toric_code=train_toric_code, pred=x_pred_prob,num_chunks=batch_size,dim=0).to(device)
                z_pred_prob = Lz_net(high_d_feature)
                z_pred_syndrome = output_to_syndrome_x(toric_code=train_toric_code, pred=z_pred_prob,num_chunks=batch_size,dim=0).to(device)

                '''# yからy_x, y_zを再構成する必要がある
                y_x, y_z = torch.zeros((0, SIZE)).to(device), torch.zeros((0, SIZE)).to(device)
                j = 0
                while j< batch_size:
                    y_x = torch.cat((y_x, y[2*j]), dim=0)
                    y_z = torch.cat((y_z, y[2*j + 1]), dim=0)
                    j += 1
                y_x = y_x.reshape((-1, 1))
                y_z = y_z.reshape((-1, 1))'''
                loss_x = criterion(x_pred_syndrome, zero_syndrome)
                loss_z = criterion(z_pred_syndrome, zero_syndrome)
                loss_gcn = loss_x + loss_z

                test_loss_x += loss_x.item()
                test_loss_z += loss_z.item()
                test_loss_gcn += loss_gcn.item()
        
        epoch_test_loss_x = test_loss_x / len(test_batch)
        epoch_test_loss_z = test_loss_z / len(test_batch)
        epoch_test_loss_gcn = test_loss_gcn / len(test_batch)

        print("Train_Loss_x     : {:.10f}".format(epoch_train_loss_x))
        print("Test_Loss_x      : {:.10f}".format(epoch_test_loss_x))
        print("Train_Loss_z     : {:.10f}".format(epoch_train_loss_z))
        print("Test_Loss_z      : {:.10f}".format(epoch_test_loss_z))
        print("Train_Loss_gcn   : {:.10f}".format(epoch_train_loss_gcn))
        print("Test_Loss_gcn    : {:.10f}".format(epoch_test_loss_gcn))
        spent = time.perf_counter() - before
        formatted_time = "{:.3f}".format(spent)
        print(f"time             : {formatted_time} seconds")

        train_loss_list_x.append(epoch_train_loss_x)
        train_loss_list_z.append(epoch_train_loss_z)
        train_loss_list_gcn.append(epoch_train_loss_gcn)
        test_loss_list_x.append(epoch_test_loss_x)
        test_loss_list_z.append(epoch_test_loss_z)
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
    torch.save(Lx_net.state_dict(), 'x_linear' + str(size) + '.pt')
    torch.save(Lz_net.state_dict(), 'z_linear' + str(size) + '.pt')
    torch.save(gcn_net.state_dict(), 'gcn' + str(size) + '.pt')
    
    # 学習の結果の可視化
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    png_directory_path = os.path.join(parent_directory, 'png')
    os.chdir(png_directory_path)
    
    x = list(range(epoch))
    y_train = train_loss_list_x
    y_test = test_loss_list_x
    plt.figure()
    plt.title('X_linear loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(x, y_train, label='train')
    plt.plot(x, y_test, label='test')
    plt.legend()
    plt.savefig('x_linear_loss.png')
    
    x = list(range(epoch))
    y_train = train_loss_list_z
    y_test = test_loss_list_z
    plt.figure()
    plt.title('Z_linear loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(x, y_train, label='train')
    plt.plot(x, y_test, label='test')
    plt.legend()
    plt.savefig('z_linear_loss.png')

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