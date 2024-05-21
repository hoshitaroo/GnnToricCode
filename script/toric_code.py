'''
Toric code シミュレーション用
paramクラスで符号距離とエラーレートを設定
'''

'''
GNN decoder 用
'''

import numpy as np
from enum import IntEnum, auto, unique
import random
import os
import torch
from torch import nn
from torch.nn import functional as F
import math

from param import param

class Pauli(IntEnum):
    I = auto()
    X = auto()
    Y = auto()
    Z = auto()

    def __mul__(self, b):
        if self == b:
            return Pauli.I
        if self is Pauli.I:
            return b
        if b is Pauli.I:
            return self
        return ({Pauli.X, Pauli.Y, Pauli.Z} - {self, b}).pop()


class ToricCode:

    #Toric codeの初期化
    def __init__(self, param) -> None:
        self.size = int(param.code_distance)
        self.errors_rate = param.errors_rate
        self.rng = np.random.default_rng()
        self.conv_kernel1 = nn.Parameter(
                torch.ones(1, 1, 1, 2), requires_grad=False)
        self.conv_kernel2 = nn.Parameter(
                torch.ones(1, 1, 2, 1), requires_grad=False)

    #エラーの発生
    def generate_errors(self) -> np.array:
        errors_list = self.rng.choice([0, 1, 2, 3], # I, X, Y, Z
                                     p = [1-self.errors_rate[0]-self.errors_rate[1]-self.errors_rate[2],self.errors_rate[0],self.errors_rate[1],self.errors_rate[2]],
                                     size = 2*self.size*self.size)
        errors = np.array(errors_list)
        errors = errors.reshape(2*self.size, self.size)
        return errors
    
    # X作用素によるパリティチェックとシンドローム生成(点上スタビライザー)
    def generate_syndrome_X(self,errors) -> np.array:

        syndrome = np.zeros((self.size, self.size), dtype=int)
        i = 0
        j = 0
        
        while 1:
            while 1:
                count = 0
                #左のqubitのcheck
                if j==0:
                    if errors[2*i][self.size-1] == 3 or errors[2*i][self.size-1] == 2:
                        count = count+1
                else:
                    if errors[2*i][j-1] == 3 or errors[2*i][j-1] == 2:
                        count = count+1
                #上のqubitのcheck
                if i==0:
                    if errors[2*self.size-1][j] == 3 or errors[2*self.size-1][j] == 2:
                        count = count+1
                else:
                    if errors[2*i-1][j] == 3 or errors[2*i-1][j] == 2:
                        count = count+1
                #右のqubitのcheck
                if errors[2*i][j] == 3 or errors[2*i][j] == 2:
                    count = count+1
                #下のqubitのcheck
                if errors[2*i+1][j] == 3 or errors[2*i+1][j] == 2:
                    count =count+1
                #パリティチェック
                if count%2 == 0:
                    syndrome[i][j] = 0
                else:
                    syndrome[i][j] = 1
                
                # j 終了判定
                if j==self.size-1:
                    j = 0
                    break
                else:
                    j = j+1
            # i 終了判定
            if i==self.size-1:
                break
            else:
                i = i+1
        
        return syndrome
    
    # Z作用素によるパリティチェックとシンドローム生成(面上スタビライザー)
    def generate_syndrome_Z(self,errors) -> np.array:

        syndrome = np.zeros((self.size, self.size), dtype =int)
        i = 0
        j = 0
        
        while 1:
            while 1:
                count = 0
                #右のqubitのcheck
                if j==self.size-1:
                    if errors[2*i+1][0] == 1 or errors[2*i+1][0] == 2:
                        count = count+1
                else:
                    if errors[2*i+1][j+1] == 1 or errors[2*i+1][j+1] == 2:
                        count = count+1
                #下のqubitのcheck
                if i==self.size-1:
                    if errors[0][j] == 1 or errors[0][j] == 2:
                        count = count+1
                else:
                    if errors[2*i+2][j] == 1 or errors[2*i+2][j] == 2:
                        count = count+1
                #左のqubitのcheck
                if errors[2*i+1][j] == 1 or errors[2*i+1][j] == 2:
                    count = count+1
                #上のqubitのcheck
                if errors[2*i][j] == 1 or errors[2*i][j] == 2:
                    count =count+1
                #パリティチェック
                if count%2 == 0:
                    syndrome[i][j] = 0
                else:
                    syndrome[i][j] = 1
                
                # j 終了判定
                if j==self.size-1:
                    j = 0
                    break
                else:
                    j = j+1
            # i 終了判定
            if i==self.size-1:
                break
            else:
                i = i+1
        
        return syndrome
    
    def not_has_non_trivial_x(self,errors) -> bool:
        '''
        shape of errors is (2*size,size)
        '''
        errors_tmp = np.zeros((2,self.size,self.size), dtype = int)
        
        i=0
        j=0
        m=0
        n=0
        #横
        while i < 2*self.size:
            while j < self.size:
                errors_tmp[0,m,n] = errors[i,j]
                j = j+1
                n = n+1
            n = 0
            j = 0
            i = i+2
            m = m+1
        
        i=1
        j=0
        m=0
        n=0
        #縦
        while i < 2*self.size:
            while j < self.size:
                errors_tmp[1,m,n] = errors[i,j]
                j = j+1
                n = n+1
            n = 0
            j = 0
            i = i+2
            m = m+1
        
        count = [0,0]
        for i in range(self.size):
            for j in range(self.size):
                if errors_tmp[0,i,j]==1 or errors_tmp[0,i,j]==2:
                    count[0] = count[0]+1
        
        for i in range(self.size):
            for j in range(self.size):
                if errors_tmp[1,i,j]==1 or errors_tmp[1,i,j]==2:
                    count[1] = count[1]+1
        return (count[0]%2==0 and count[1]%2==0)
    
    def syndrome(self, errors: torch.Tensor) -> torch.Tensor:
        '''
        get extended syndrome of `errors`, this function is differentiable
        '''
        split = torch.split(errors, self.size, dim=2)
        errors1, errors2 = split[0], split[1]
        x1 = torch.cat(
                (errors1[:, :, :, -1].unsqueeze(3), errors1), dim=3)
        x2 = torch.cat(
                (errors2[:, :, -1].unsqueeze(2), errors2), dim=2)
        x = F.conv2d(x1, self.conv_kernel1) + F.conv2d(x2, self.conv_kernel2)
        return 0.5 * (1 - torch.cos(np.pi * x))

    def not_has_non_trivial_z(self,errors) -> bool:
        '''
        shape of errors is (2*size,size)
        '''
        errors_tmp = np.zeros((2,self.size,self.size), dtype = int)
        
        i=0
        j=0
        m=0
        n=0
        #横
        while i < 2*self.size:
            while j < self.size:
                errors_tmp[0,m,n] = errors[i,j]
                j = j+1
                n = n+1
            n = 0
            j = 0
            i = i+2
            m = m+1
        
        i=1
        j=0
        m=0
        n=0
        #縦
        while i < 2*self.size:
            while j < self.size:
                errors_tmp[1,m,n] = errors[i,j]
                j = j+1
                n = n+1
            n = 0
            j = 0
            i = i+2
            m = m+1
        count = [0,0]
        for i in range(self.size):
            for j in range(self.size):
                if errors_tmp[0,i,j]==3 or errors_tmp[0,i,j]==2:
                    count[0] = count[0]+1
        
        for i in range(self.size):
            for j in range(self.size):
                if errors_tmp[1,i,j]==3 or errors_tmp[1,i,j]==2:
                    count[1] = count[1]+1
        return (count[0]%2==0 and count[1]%2==0)

    def decode_x_error(self,errors,u,v):
        
        u = list(u)
        v = list(v)
        #u,v in matching_z

        #行成分から訂正を行う
        while u[1] != v[1]:  # small [u / v] large
            if u[1] > v[1]:
                u, v = v, u #uの方が左にあることが確定
            x = u[0]*2 + 1
            if v[1] - u[1] > self.size // 2: #境界を跨ぐパターン
                y = u[1]%self.size #左のqubit
                if errors[x, y]%2==0:
                    errors[x, y] += 1
                else:
                    errors[x, y] -= 1
                u[1] -= 1
                u[1] %= self.size
            else: #境界を跨がないパターン
                y = (u[1] + 1) % self.size
                if errors[x, y]%2==0:
                    errors[x, y] += 1
                else:
                    errors[x, y] -= 1
                u[1] += 1
                u[1] %= self.size

        #次に列成分の訂正を行う
        while u[0] != v[0]:  # small [u   v] large
            if u[0] > v[0]:
                u, v = v, u #uの方が上にあることが確定
            y = u[1]
            if v[0] - u[0] > self.size // 2:  #境界を跨ぐパターン
                x = 2*u[0]
                if errors[x, y]%2==0:
                    errors[x, y] += 1
                else:
                    errors[x, y] -= 1
                u[0] -= 1
                u[0] %= self.size
            else: # 境界を跨がないパターン
                x = (2*u[0]+2) % (2*self.size)
                
                if errors[x, y]%2==0:
                    errors[x, y] += 1
                else:
                    errors[x, y] -= 1
                
                u[0] += 1
                u[0] %= self.size
        return errors
    
    def decode_z_error(self,errors,u,v):
        
        u = list(u)
        v = list(v)
        #u,v in matching_x

        #行成分の訂正
        while u[1] != v[1]:  # small [u / v] large
            if u[1] > v[1]:
                u, v = v, u #uが左にあることが確定
            x = 2*u[0]
            if v[1] - u[1] > self.size // 2: #境界を跨ぐパターン
                y = (u[1] - 1) % self.size
                errors[x, y] = 3 - errors[x, y]
                u[1] -= 1
                u[1] %= self.size
            else: #境界を跨がないパターン
                y = u[1] % self.size
                errors[x, y] = 3 - errors[x, y]
                u[1] += 1
                u[1] %= self.size

        #列成分の訂正
        while u[0] != v[0]:  # small [u   v] large
            if u[0] > v[0]:
                u, v = v, u #uが上にあることが確定
            y = u[1]
            if v[0] - u[0] > self.size // 2: #境界を跨ぐパターン
                x = (2*u[0] - 1) % (self.size*2)
                errors[x, y] = 3 - errors[x, y]
                u[0] -= 1
                u[0] %= self.size
            else: #境界を跨がないパターン
                x = (2*u[0] + 1) % (self.size*2)
                errors[x, y] = 3 - errors[x, y]
                u[0] += 1
                u[0] %= self.size
        return errors
    
    def errors_to_errorsXZ(self, errors):
        errors_x = np.zeros((2*self.size, self.size), dtype=int)
        errors_z = np.zeros((2*self.size, self.size), dtype=int)

        for i in range(2*self.size):
            for j in range(self.size):
                if errors[i][j] == 1 or errors[i][j] == 2:
                    errors_x[i][j] = 1
                if errors[i][j] == 3 or errors[i][j] == 2:
                    errors_z[i][j] = 1
        
        return errors_x, errors_z
    
    def operate_x(self, errors, x_map):
        for i in range(2*self.size - 1):
            for j in range(self.size - 1):
                if x_map[i, j] == 1:
                    errors[i, j] = 3 - errors[i, j]
        return errors

    def operate_z(self, errors, z_map):
        for i in range(2*self.size - 1):
            for j in range(self.size - 1):
                if z_map[i, j] == 1:
                    if errors[i, j]%2==0:
                        errors[i, j] += 1
                    else:
                        errors[i, j] -= 1
        return errors
    
    def generate_syndrome_X_tensor(self, errors):
        syndrome = torch.zeros((self.size, self.size), dtype=torch.float32)
        i = 0
        j = 0

        while True:
            while True:
                count = 0
                # 左のqubitのcheck
                if j == 0:
                    count += errors[2 * i][self.size - 1]
                else:
                    count += errors[2 * i][j - 1]
                # 上のqubitのcheck
                if i == 0:
                    count += errors[2 * self.size - 1][j]
                else:
                    count += errors[2 * i - 1][j]
                # 右のqubitのcheck
                count += errors[2 * i][j]
                # 下のqubitのcheck
                count += errors[2 * i + 1][j]
                # パリティチェック
                syndrome[i][j] = 0.5*(1-torch.cos(np.pi * count))

                # j 終了判定
                if j == self.size - 1:
                    j = 0
                    break
                else:
                    j += 1
            # i 終了判定
            if i == self.size - 1:
                break
            else:
                i += 1
        return syndrome

    def generate_syndrome_Z_tensor(self, errors):
        syndrome = torch.zeros((self.size, self.size), dtype=torch.float32)
        i = 0
        j = 0
        while True:
            while True:
                count = 0
                # 右のqubitのcheck
                if j == self.size - 1:
                    count += errors[2 * i + 1][0]
                else:
                    count += errors[2 * i + 1][j + 1]
                # 下のqubitのcheck
                if i == self.size - 1:
                    count += errors[0][j]
                else:
                    count += errors[2 * i + 2][j]
                # 左のqubitのcheck
                count += errors[2 * i + 1][j]
                # 上のqubitのcheck
                count += errors[2 * i][j]
                # パリティチェック
                syndrome[i][j] = 0.5*(1-torch.cos(np.pi * count))

                # j 終了判定
                if j == self.size - 1:
                    j = 0
                    break
                else:
                    j += 1
            # i 終了判定
            if i == self.size - 1:
                break
            else:
                i += 1
        return syndrome

#test
'''
param_1 = param(p=0.05, size=5)
toric_code = ToricCode(param_1)
errors = toric_code.generate_errors()
print("errors")
print(errors)
syndrome_x = toric_code.generate_syndrome_X(errors)
print("syndrome_x")
print(syndrome_x)
syndrome_z = toric_code.generate_syndrome_Z(errors)
print("syndrome_z")
print(syndrome_z)
'''

'''
param_1 = param(p=0.15, size=5)
toric_code = ToricCode(param_1)
errors = toric_code.generate_errors()
print(errors)
errors_x, errors_z = toric_code.errors_to_errorsXZ(errors)
print(errors_x)
print(errors_z) 
'''