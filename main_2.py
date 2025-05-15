# 修复 nan 并优化 MPS
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
import os, joblib

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
if device.type=='mps': torch.set_float32_matmul_precision('high')
os.makedirs("model", exist_ok=True)

class RMSLELoss(nn.Module):
    def forward(self, y_pred, y_true):
        y_pred = torch.clamp(y_pred, min=0)
        y_true = torch.clamp(y_true, min=0)
        return torch.sqrt(((torch.log1p(y_pred)-torch.log1p(y_true))**2).mean())

def create_dataset(fp, scaler_path='model/scaler.pkl', augment=True):
    df=pd.read_csv(fp).dropna().drop(columns=['id'])
    df['Sex']=df['Sex'].map({'male':1,'female':0})
    x, y = df.iloc[:,:-1].values.astype(np.float32), df.iloc[:,-1].values.astype(np.float32)
    xt,xv,yt,yv = train_test_split(x,y,test_size=0.2,random_state=42)
    scaler=MinMaxScaler().fit(xt); joblib.dump(scaler,scaler_path)
    xtr, xvl = scaler.transform(xt), scaler.transform(xv)
    if augment:
        mask = (np.arange(xtr.shape[1])!=df.columns.get_loc('Sex')).astype(np.float32)
        xa, ya = [], []
        for _ in range(10):
            g = np.random.normal(0, np.random.uniform(0.01,0.04), xtr.shape)
            u = np.random.uniform(-np.random.uniform(0.01,0.03), np.random.uniform(0.01,0.03), xtr.shape)
            xa.append(xtr + (g+u)*mask)
            y_noise = yt + np.random.normal(0,0.015*yt.std(), yt.shape)
            ya.append(np.clip(y_noise, 0, None))
        xtr = np.vstack([xtr]+xa).astype(np.float32)
        yt = np.concatenate([yt]+ya).astype(np.float32)
    return TensorDataset(torch.from_numpy(xtr),torch.from_numpy(yt)), TensorDataset(torch.from_numpy(xvl),torch.from_numpy(yv)), x.shape[1]

class Model(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.input=nn.Sequential(nn.Linear(d,256),nn.BatchNorm1d(256),nn.SiLU())
        def blk(i,o,n):
            proj=nn.Linear(i,o); self.add_module(f'p{i}_{o}',proj)
            layers=[l for _ in range(n) for l in (nn.Linear(o,o),nn.BatchNorm1d(o),nn.SiLU(),nn.Dropout(0.3))]
            return proj, nn.Sequential(*layers)
        self.p1,self.r1=blk(256,512,2)
        self.p2,self.r2=blk(512,1024,3)
        self.p3,self.r3=blk(1024,768,2)
        self.p4,self.r4=blk(768,512,2)
        self.p5,self.r5=blk(512,256,2)
        self.output=nn.Sequential(
            nn.Linear(256,128),nn.BatchNorm1d(128),nn.SiLU(),nn.Dropout(0.2),
            nn.Linear(128,64),nn.BatchNorm1d(64),nn.SiLU(),nn.Dropout(0.2),
            nn.Linear(64,1)
        )
    def forward(self,x):
        x=self.input(x)
        for p,r in [(self.p1,self.r1),(self.p2,self.r2),(self.p3,self.r3),(self.p4,self.r4),(self.p5,self.r5)]:
            res=p(x); x=r(res)+res
        return self.output(x)

def train(tr_ds,vl_ds,dim,epochs=200):
    tr_dl=DataLoader(tr_ds,512,shuffle=True)
    vl_dl=DataLoader(vl_ds,512,shuffle=False)
    m=Model(dim).to(device)
    opt=Lookahead(optim.AdamW(m.parameters(),lr=1e-4),k=5,alpha=0.5)
    sched=optim.lr_scheduler.ReduceLROnPlateau(opt,'min',factor=0.5,patience=10,threshold=1e-4,cooldown=5)
    scaler=GradScaler(); crit=RMSLELoss()
    for e in range(epochs):
        m.train(); tl=0
        for x,y in tr_dl:
            x,y=x.to(device),y.unsqueeze(1).to(device)
            opt.zero_grad()
            with autocast():
                loss=crit(m(x),y)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.0)
            scaler.step(opt); scaler.update()
            tl+=loss.item()
        vl=0
        m.eval()
        with torch.no_grad():
            for x,y in vl_dl:
                x,y=x.to(device),y.unsqueeze(1).to(device)
                with autocast(): vl+=crit(m(x),y).item()
        tr_loss, vl_loss = tl/len(tr_dl), vl/len(vl_dl)
        sched.step(vl_loss)
        print(f"{e+1}/{epochs} tr={tr_loss:.4f} vl={vl_loss:.4f}")
    torch.save(m.state_dict(),'model/model.pth')
    plt.plot; plt.savefig('model/loss_curve.png')

if __name__=='__main__':
    tr,vl,d = create_dataset('data/train.csv')
    train(tr,vl,d)
