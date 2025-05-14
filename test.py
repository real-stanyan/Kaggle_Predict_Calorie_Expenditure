import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import numpy as np

# 模拟数据（1000 条样本，20 个特征）
X = np.random.rand(1000, 20).astype(np.float32)
y = np.random.rand(1000).astype(np.float32)
dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 简单模型
class SimpleModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# 测试函数
def benchmark(device_str):
    device = torch.device(device_str)
    model = SimpleModel(20).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    start = time.time()
    model.train()
    for epoch in range(5):
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
            pred = model(xb)
            loss = criterion(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    end = time.time()
    print(f"{device_str.upper()} training time: {end - start:.2f} seconds")

# 运行对比
benchmark("cpu")

if torch.backends.mps.is_available():
    benchmark("mps")
else:
    print("MPS not available on this device.")
