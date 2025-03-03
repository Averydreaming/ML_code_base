# models/transformer.py
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class Transformer(nn.Module):
    def __init__(self, input_dim):
        super(Transformer, self).__init__()
        self.embedding = nn.Linear(input_dim, 64)
        encoder_layers = TransformerEncoderLayer(d_model=64, nhead=8)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=3)
        self.fc = nn.Linear(64, 1)
    
    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)  # Add sequence dimension
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Average over sequence length
        return self.fc(x)