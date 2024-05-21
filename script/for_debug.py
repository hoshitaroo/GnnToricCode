import torch
from toric_code import ToricCode
from param import param
import numpy as np

SIZE = 5                            # 符号距離
EPOCH = 400
batch_size = 3
num_replace = int(0.25 * 250000)
NUM_CORE = 1
lr = 1e-5

def output_to_syndrome_x(toric_code, pred , num_chunks=batch_size, dim=0):
    #テンソルの形状を長方形に変形
    reshaped_pred = pred.view((2*SIZE*batch_size, SIZE))
    #テンソルをバッチサイズの数に分割する
    splited_tensors_list = torch.chunk(reshaped_pred, num_chunks, dim=dim)
    syndrome = torch.zeros()
    for i in splited_tensors_list:
        j = toric_code.generate_syndrome_X_tensor(i)
        syndrome = torch.cat((syndrome, j), dim=0)
    syndrome_flat = syndrome.flatten()
    return syndrome_flat

def output_to_syndrome_z(toric_code, pred , num_chunks=batch_size, dim=0):
    #テンソルの形状を長方形に変形
    reshaped_pred = pred.view((2*SIZE*batch_size, SIZE))
    #テンソルをバッチサイズの数に分割する
    splited_tensors_list = torch.chunk(reshaped_pred, num_chunks, dim=dim)
    syndrome = torch.zeros((0, SIZE))
    for i in splited_tensors_list:
        j = toric_code.generate_syndrome_Z_tensor(i)
        syndrome = torch.cat((syndrome, j), dim=0)
    print(syndrome)
    print(syndrome.shape)
    syndrome_flat = syndrome.flatten()
    return syndrome_flat

param_ = param(p=0.05, size=SIZE)
toric_code = ToricCode(param=param_)

errors = toric_code.generate_errors()
errors_x, errors_z = toric_code.errors_to_errorsXZ(errors)
errors_x_tensor = torch.rand(10*batch_size, 5)
print(errors_x_tensor)
syndrome_x = output_to_syndrome_z(toric_code=toric_code,
                                  pred=errors_x_tensor,
                                  num_chunks=batch_size,
                                  dim=0)

