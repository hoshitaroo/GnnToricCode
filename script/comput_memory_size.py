import torch

# テンソルの形状とデータ型
n = 5
shape1 = (2*n, n)
shape2 = (2, int(2*n*n*(2*n*n-1)/2))
shape3 = (2*n, n)
dtype = torch.float32  # または torch.float64 など

# メモリ使用量の計算
memory_bytes = (torch.prod(torch.tensor(shape1)) +
                torch.prod(torch.tensor(shape2)) +
                torch.prod(torch.tensor(shape3))) * dtype.itemsize * 4e6

# バイト単位からギガバイト単位に変換
memory_gb = memory_bytes / (1024 ** 3)

print(f"概算メモリ使用量: {memory_gb:.2f} GB")
