# %%
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch_optimizer import Lookahead
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import time
import os
import joblib
import torch.nn.functional as F

# 设置设备：支持 MPS（Apple Silicon）或 CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# print("Using device:", device)

# 创建模型保存目录
os.makedirs("model", exist_ok=True)

# %%
class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        y_pred = torch.clamp(y_pred, min=0)
        log_pred = torch.log1p(y_pred)
        log_true = torch.log1p(y_true)
        return torch.sqrt(torch.mean((log_pred - log_true) ** 2))

# %%
def create_dataset(file_path, save_scaler_path='model/scaler.pkl', test_size=0.2, random_state=42, augment=True):
    df = pd.read_csv(file_path).dropna()
    df = df.drop(columns=['id'])
    df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})

    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    x_train, x_valid, y_train, y_valid = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )

    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_valid_scaled = scaler.transform(x_valid)
    joblib.dump(scaler, save_scaler_path)

    if augment:
        x_aug, y_aug = [], []
        times = 10
        binary_mask = (np.array(x.columns) != 'Sex').astype(int)

        for _ in range(times):
            # 随机扰动强度
            gaussian_std = np.random.uniform(0.01, 0.04)
            uniform_range = np.random.uniform(0.01, 0.03)

            gaussian_noise = np.random.normal(0, gaussian_std, size=x_train_scaled.shape)
            uniform_noise = np.random.uniform(-uniform_range, uniform_range, size=x_train_scaled.shape)
            noise = (gaussian_noise + uniform_noise) * binary_mask

            x_aug.append(x_train_scaled + noise)

            y_noise = y_train.values + np.random.normal(0, 0.015 * y_train.std(), size=y_train.shape)
            y_aug.append(y_noise)

        x_train_scaled = np.vstack([x_train_scaled] + x_aug).astype(np.float32)
        y_train = np.concatenate([y_train.values] + y_aug).astype(np.float32)
    else:
        x_train_scaled = x_train_scaled.astype(np.float32)
        y_train = y_train.astype(np.float32)

    x_valid_scaled = x_valid_scaled.astype(np.float32)
    y_valid = y_valid.astype(np.float32)

    train_dataset = TensorDataset(torch.tensor(x_train_scaled), torch.tensor(y_train))
    valid_dataset = TensorDataset(torch.tensor(x_valid_scaled), torch.tensor(y_valid.values))

    return train_dataset, valid_dataset, x.shape[1]




# %%
class Model(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(dim, 256),
            nn.BatchNorm1d(256),
            nn.SiLU()
        )

        self.res_stack1 = self.make_block(256, 512, layers=2)
        self.res_stack2 = self.make_block(512, 1024, layers=3)
        self.res_stack3 = self.make_block(1024, 768, layers=2)
        self.res_stack4 = self.make_block(768, 512, layers=2)
        self.res_stack5 = self.make_block(512, 256, layers=2)

        self.output = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def make_block(self, in_dim, out_dim, layers):
        layers_list = []

        # 用 projection 确保维度对齐
        self.add_module(f'proj_{in_dim}_{out_dim}', nn.Linear(in_dim, out_dim))

        for _ in range(layers):
            layers_list.extend([
                nn.Linear(out_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.SiLU(),
                nn.Dropout(0.3)
            ])

        return nn.Sequential(*layers_list)

    def forward(self, x):
        x = self.input(x)

        x = self._forward_block(x, self.res_stack1, 'proj_256_512')
        x = self._forward_block(x, self.res_stack2, 'proj_512_1024')
        x = self._forward_block(x, self.res_stack3, 'proj_1024_768')
        x = self._forward_block(x, self.res_stack4, 'proj_768_512')
        x = self._forward_block(x, self.res_stack5, 'proj_512_256')

        x = self.output(x)
        return x

    def _forward_block(self, x, block, proj_name):
        residual = getattr(self, proj_name)(x)
        x = block(residual) + residual
        return x

def load_test_data(file_path, scaler_path='model/scaler.pkl'):
    df = pd.read_csv(file_path).dropna()
    ids = df['id'].values  # 保留 id
    df = df.drop(columns=['id'])
    df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})

    # 加载训练时保存的 scaler 进行归一化
    scaler = joblib.load(scaler_path)
    x = scaler.transform(df)
    x_tensor = torch.tensor(x.astype(np.float32))
    return ids, x_tensor

def predict_and_save(test_tensor, dim, ids, output_path='output.csv'):
    model = Model(dim).to(device)
    model.load_state_dict(torch.load('model/model.pth'))
    model.eval()

    with torch.no_grad():
        test_tensor = test_tensor.to(device)
        preds = model(test_tensor).squeeze().cpu().numpy()

    df_out = pd.DataFrame({'id': ids, 'Calories': preds})
    df_out.to_csv(output_path, index=False)
    print(f"预测已保存到 {output_path}")


# %%
def train(train_dataset, valid_dataset, dim, num_epoch):
    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=2048, shuffle=False, num_workers=4)

    model = Model(dim).to(device)
    criterion = RMSLELoss()
    base_optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    optimizer = Lookahead(base_optimizer)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        threshold=1e-4,
        cooldown=5,
        verbose=True
    )

    scaler = GradScaler()  # 混合精度梯度缩放器

    train_losses = []
    valid_losses = []

    for epoch in range(num_epoch):
        model.train()
        total_train_loss = 0.0
        train_batches = 0
        start_time = time.time()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device).unsqueeze(1)

            optimizer.zero_grad()

            with autocast():  # 自动精度推理
                pred = model(x)
                loss = criterion(pred, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()
            train_batches += 1

        avg_train_loss = total_train_loss / train_batches
        train_losses.append(avg_train_loss)

        # ===== 验证阶段 =====
        model.eval()
        total_valid_loss = 0.0
        valid_batches = 0
        with torch.no_grad():
            for x, y in valid_loader:
                x, y = x.to(device), y.to(device).unsqueeze(1)
                with autocast():
                    pred = model(x)
                    loss = criterion(pred, y)
                total_valid_loss += loss.item()
                valid_batches += 1

        avg_valid_loss = total_valid_loss / valid_batches
        valid_losses.append(avg_valid_loss)

        scheduler.step(avg_valid_loss)
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}, Time: {time.time() - start_time:.2f}s")

    torch.save(model.state_dict(), 'model/model.pth')
    print("模型已保存到 model/model.pth")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epoch + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epoch + 1), valid_losses, label='Valid Loss', marker='s')
    plt.title('Training & Validation Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('RMSLE Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('model/loss_curve.png')
    plt.show()
# %%
def test(test_dataset, dim):
    dataloader = DataLoader(
        test_dataset,
        batch_size=2048,
        shuffle=False,
        num_workers=4
    )
    model = Model(dim).to(device)
    model.load_state_dict(torch.load('model/model.pth'))
    model.eval()

    predictions = []
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            y_pred = model(x)
            predictions.extend(y_pred.cpu().squeeze().tolist())

    print("测试集预测完成，前10个结果：", predictions[:10])
    return predictions


# %%
if __name__ == '__main__':
    train_dataset, valid_dataset, dim = create_dataset('data/train.csv')

    train(train_dataset, valid_dataset, dim, num_epoch=200)
     # 加载 test.csv 并预测
    ids, test_tensor = load_test_data('data/test.csv')
    _, _, dim = create_dataset('data/train.csv')  # 获取特征维度
    predict_and_save(test_tensor, dim, ids, output_path='output.csv')


