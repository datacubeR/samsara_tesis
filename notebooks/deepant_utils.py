from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import torch.nn as nn

import pytorch_lightning as pl

class CanadianTS(Dataset):
    def __init__(self, df, lookback_size, prediction = False):
        self.lookback_size = lookback_size
        self.X, self.y, self.sc = self.preprocess_data(df)
        self.prediction = prediction
        self.ts = df.index[self.lookback_size:df.shape[0]].tolist()
    
    def __len__(self):
        return len(self.X)
    
    def preprocess_data(self, df):
        sc = MinMaxScaler()
        df_sc = sc.fit_transform(df)
        X = np.zeros(shape=(df_sc.shape[0]-self.lookback_size,self.lookback_size,df_sc.shape[1]))
        y = np.zeros(shape=(df_sc.shape[0]-self.lookback_size,df_sc.shape[1]))

        for i in range(self.lookback_size-1, df.shape[0]-1):
            #timesteps.append(df_sc.index[i+1]) # yo creo que este debería ser el index de y (i+1)
            y[i-self.lookback_size+1] = df_sc[i+1]
            X[i-self.lookback_size+1] = df_sc[i-self.lookback_size+1:i+1]
        
        return X,y, sc    
    def __getitem__(self, idx):
        
        return torch.tensor(self.X[idx],dtype=torch.float), torch.tensor(self.y[idx],dtype=torch.float)
    

class CanadianTSModule(pl.LightningDataModule):
    def __init__(self, dataset, lookback_size):
        super().__init__()
        self.dataset = dataset
        self.lookback_size = lookback_size
    
    def setup(self, stage = None):
        self.train_dataset = CanadianTS(self.dataset, self.lookback_size)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=32, num_workers = 8, pin_memory = True, shuffle=False)
    
    def predict_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=1, num_workers = 8, pin_memory = True, shuffle=False)

class DeepAnT(nn.Module):
    def __init__(self, LOOKBACK_SIZE, OUT_DIMENSION):
        super().__init__()
        
        self.convblock1 = nn.Sequential(
            nn.Conv1d(in_channels=LOOKBACK_SIZE, out_channels = 16, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2)
        )
        self.convblock2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels = 16, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2)
        )
        self.flatten = nn.Flatten()
        self.denseblock = nn.Sequential(
            nn.Linear(80,40),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.25)
        )
        self.out = nn.Linear(40,OUT_DIMENSION)
        
    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.flatten(x)
        x = self.denseblock(x)
        x = self.out(x)
        return x
    
class AnomalyDetectorForecast(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.criterion = nn.MSELoss()
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        X,y = batch
        y_pred  = self(X)
        loss = self.criterion(y_pred, y)
        self.log('train_loss', loss,  prog_bar = True, logger = True)
        return loss
    def predict_step(self, batch, batch_idx):
        X,y = batch
        y_pred = self(X)
        return y_pred, torch.linalg.norm(y_pred-y)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-5)