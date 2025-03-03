# models/lstm.py
import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_dim):
        super(LSTM, self).__init__()
        self.hidden_size = 64
        self.lstm = nn.LSTM(input_dim, self.hidden_size, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, 1)
    
    def forward(self, x):
        h_lstm, _ = self.lstm(x)
        out = self.fc(h_lstm[:, -1, :])
        return out