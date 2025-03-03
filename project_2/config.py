# config.py
class Config:
    DATA_PATH = 'path/to/your/dataset'
    INPUT_DIM = 128  # Example input dimension
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 0.001
    MODEL_TYPE = 'mlp'  # Options: 'linear_regression', 'mlp', 'cnn', 'lstm', 'transformer'