import torch

rand = torch.rand(16, 32, 100)
temp = torch.rand(16, 32, 100)
print(temp[rand < 0.5].size())
