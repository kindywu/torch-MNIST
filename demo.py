import torch

# 给定的列表
values = [-8.4667, -9.3926, -2.3098, -9.4697, -0.1122, -12.9132, -5.3320,
          -7.1854, -7.7535, -7.6385]

# 将列表转换为PyTorch张量
tensor = torch.tensor(values)

# 使用torch.max获取最大值
max_value =  torch.max(tensor)
print(max_value)

# 使用torch.max获取最大值的索引
max_value =  torch.argmax(tensor)
print(max_value)

