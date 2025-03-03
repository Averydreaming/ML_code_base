# evaluate.py
import torch
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

def evaluate():
    _, _, test_loader = load_data()
    
    model = get_model(Config.INPUT_DIM)
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    
    test_loss = 0
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            outputs = model(batch_features)
            loss = nn.MSELoss()(outputs, batch_labels)
            test_loss += loss.item()
    
    print(f'Test Loss: {test_loss / len(test_loader)}')

if __name__ == '__main__':
    evaluate()