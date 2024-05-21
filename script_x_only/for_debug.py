import torch

# 20x5のランダムなテンソルを作成
original_tensor = torch.rand((5, 20))

# テンソルの要素数が一致するように10x10に変更
reshaped_tensor = original_tensor.view(10, 10)

# 変更前と変更後のテンソルの形状を表示
print("変更前の形状:", original_tensor.shape)
print(original_tensor)
print("変更後の形状:", reshaped_tensor.shape)
print(reshaped_tensor)