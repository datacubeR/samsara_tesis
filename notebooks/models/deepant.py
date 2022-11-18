import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class ForecastBasedDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        return (torch.tensor(self.sequences[idx], dtype = torch.float).reshape(1,-1),
                torch.tensor(self.targets[idx], dtype=torch.float))

class DataModule(pl.LightningDataModule):
    def __init__(self, sequences, targets, forecast = True):
        super().__init__()
        self.sequences = sequences
        self.targets = targets
        self.forecast = forecast
        
    def setup(self, stage = None):
        if self.forecast:
            self.train_dataset = ForecastBasedDataset(self.sequences, self.targets)
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=32, num_workers = 8, pin_memory = True, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=1, num_workers = 8, pin_memory = True, shuffle=False)

class DeepAnT(nn.Module):
    def __init__(self, SEQ_LEN, OUT_DIMENSION, conv_kernel = 1, pool_kernel = 1):
        super().__init__()
        
        self.convblock1 = nn.Sequential(
            nn.Conv1d(in_channels=SEQ_LEN, out_channels = 16, kernel_size=conv_kernel),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_kernel)
        )
        self.convblock2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels = 16, kernel_size=conv_kernel),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_kernel)
        )
        self.flatten = nn.Flatten()
        self.denseblock = nn.Sequential(
            nn.Linear(16,40),
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


class ForecastBasedAD(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.criterion = nn.MSELoss() # Debería ser MAE
    
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