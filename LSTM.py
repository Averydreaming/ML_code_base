import torch
import torch.nn.functional as F

from copy import deepcopy
from tqdm import tqdm
import argparse
import numpy as np

# 把LSTM写成类
class LSTM(torch.nn.Module):
    def __init__(self, epochs, learning_rate):
        super(LSTM, self).__init__()
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.model = torch.nn.Sequential(
            torch.nn.LSTM(310, 100, 2),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 3),
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        
    def forward(self, x):
        return self.model(x)
    
    def fit(self, train_data, train_label, val_data, val_label):
        self.model.train()
        total_epoch= self.epochs
        # 进度条
        pbar = tqdm(range(total_epoch))
        for epoch in pbar:
            # 前向传播
            y_pred = self.model(train_data)
            # 计算损失
            loss = self.loss_fn(y_pred, train_label)
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # 更新进度条
            pbar.set_description("loss: %.4f" % loss.item())
            # 每100次验证一次
            if epoch % 100 == 0:
                # 验证
                self.model.eval()
                with torch.no_grad():
                    val_pred = self.model(val_data)
                    val_loss = self.loss_fn(val_pred, val_label)
                    # 存储最好的模型为best_model
                    if epoch == 0:
                        best_loss = val_loss
                        best_model = deepcopy(self.model)
                    else:
                        if val_loss < best_loss:
                            best_loss = val_loss
                            best_model = deepcopy(self.model)
                    # 更新进度条
                    pbar.set_description("train_loss: %.4f, validation_loss: %.4f" % (loss.item(), val_loss.item()))
                self.model.train()
        #输出best_model的结果
    
        with torch.no_grad():
            y_pred = best_model(train_data)
            train_loss = self.loss_fn(y_pred, train_label)
            train_pred = torch.argmax(F.softmax(y_pred, dim=1), dim=1)
            train_acc = (train_pred == train_label).sum().item() / len(train_label)
            val_pred = best_model(val_data)
            val_loss = self.loss_fn(val_pred, val_label)
            val_pred = torch.argmax(F.softmax(val_pred, dim=1), dim=1)
            val_acc = (val_pred == val_label).sum().item() / len(val_label)
            print("train_loss: %.4f, train_acc: %.4f, validation_loss: %.4f, validation_acc: %.4f" % (
            train_loss.item(), train_acc, val_loss.item(), val_acc))
            
        return best_model,val_acc