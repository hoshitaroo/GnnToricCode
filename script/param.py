'''
Toric code シミュレーションの際のパラメータ設定用
'''

import numpy as np

class param:
    def __init__(self, p, size):
        self.code_distance = size                   # must be odd
        self.p = p
        self.errors_rate = np.array([p/3, p/3, p/3])# X, Y, Z