import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import time
import os
import joblib

# 创建模型保存目录
os.makedirs("model", exist_ok=True)

# 自定义 RMSLE 损失函数
class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        y_pred = torch.clamp(y_pred, min=0)
        log_pred = torch.log1p(y_pred)
        log_true = torch.log1p(y_true)
        return torch.sqrt(torch.mean((log_pred - log_true) ** 2))

def load_train_dataset(file_path, save_scaler_path='model/scaler.pkl'):
    df = pd.read_csv(file_path)
    df = df.dropna()
    df = df.drop(columns=['id'])
    df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})

    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x)
    joblib.dump(scaler, save_scaler_path)

    x_scaled = x_scaled.astype(np.float32)
    y = y.astype(np.float32)

    dataset = TensorDataset(torch.tensor(x_scaled), torch.tensor(y.values))
    return dataset, x.shape[1]

def load_test_dataset(file_path, scaler_path='model/scaler.pkl'):
    df = pd.read_csv(file_path)
    df = df.dropna()
    df = df.drop(columns=['id'])
    df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})

    x = df.iloc[:, :]
    y = df.iloc[:, -1] if 'Price' in df.columns else pd.Series([0]*len(x))  # 若测试集无标签
    scaler = joblib.load(scaler_path)
    x_scaled = scaler.transform(x)

    x_scaled = x_scaled.astype(np.float32)
    y = y.astype(np.float32)

    dataset = TensorDataset(torch.tensor(x_scaled), torch.tensor(y.values))
    return dataset, x.shape[1]

# 定义模型
# class Model(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.linear1 = nn.Linear(dim, 128)
#         self.linear2 = nn.Linear(128, 256)
#         self.linear3 = nn.Linear(256, 512)
#         self.linear4 = nn.Linear(512, 256)
#         self.linear5 = nn.Linear(256, 128)
#         self.output = nn.Linear(128, 1)

#     def forward(self, x):
#         x = torch.relu(self.linear1(x))
#         x = torch.relu(self.linear2(x))
#         x = torch.relu(self.linear3(x))
#         x = torch.relu(self.linear4(x))
#         x = torch.relu(self.linear5(x))
#         return self.output(x)

class Model(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear1 = nn.Linear(dim, 32)
        self.linear2 = nn.Linear(32, 16)
        self.output = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        return self.output(x)


# 训练模型
def train(train_dataset, dim, num_epoch):
    dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    model = Model(dim)
    criterion = RMSLELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    epoch_losses = []

    for epoch in range(num_epoch):
        model.train()
        total_loss = 0.0
        batch_num = 0
        start_time = time.time()

        for x, y in dataloader:
            y = y.unsqueeze(1)
            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_num += 1

        avg_loss = total_loss / batch_num
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Time: {time.time() - start_time:.2f}s")

    # 保存模型
    torch.save(model.state_dict(), 'model/model.pth')
    print("模型已保存到 model/model.pth")

    # 绘制 loss 曲线
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epoch + 1), epoch_losses, marker='o')
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('RMSLE Loss')
    plt.grid(True)
    plt.savefig('model/loss_curve.png')
    plt.show()


# 测试模型
def test(test_dataset, dim):
    dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    model = Model(dim)
    model.load_state_dict(torch.load('model/model.pth'))
    model.eval()

    predictions = []
    with torch.no_grad():
        for x, _ in dataloader:
            y_pred = model(x)
            predictions.extend(y_pred.squeeze().tolist())

    print("测试集预测完成，前10个结果：", predictions[:10])
    return predictions

# 主程序
if __name__ == '__main__':
    train_dataset, dim = load_train_dataset('data/train.csv')
    test_dataset, _ = load_test_dataset('data/test.csv')
    
    train(train_dataset, dim, num_epoch=10)
    test(test_dataset, dim)
