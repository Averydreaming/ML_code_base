# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import load_data
from config import Config
from models.linear_regression import LinearRegression
from models.mlp import MLP
from models.cnn import CNN
from models.lstm import LSTM
from models.transformer import Transformer

def get_model(input_dim):
    if Config.MODEL_TYPE == 'linear_regression':
        return LinearRegression(input_dim)
    elif Config.MODEL_TYPE == 'mlp':
        return MLP(input_dim)
    elif Config.MODEL_TYPE == 'cnn':
        return CNN(input_dim)
    elif Config.MODEL_TYPE == 'lstm':
        return LSTM(input_dim)
    elif Config.MODEL_TYPE == 'transformer':
        return Transformer(input_dim)
    else:
        raise ValueError("Unknown model type")

def train():
    train_loader, val_loader, _ = load_data()
    
    model = get_model(Config.INPUT_DIM)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    for epoch in range(Config.EPOCHS):
        model.train()
        epoch_loss = 0
        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{Config.EPOCHS}, Loss: {epoch_loss / len(train_loader)}')
    
    # Save the model
    torch.save(model.state_dict(), 'model.pth')

if __name__ == '__main__':
    train()