# data_loader.py

import os
import numpy as np
import torch
from config import Config
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class CustomDataset(Dataset):
    def __init__(self, data_root):
        """
        Args:
            data_root (str): 根目录，包含所有数据文件夹。
        """
        self.data_root = data_root
        self.file_pairs = self._load_file_pairs()

    def _load_file_pairs(self):
        """
        加载所有 (X, Y) 文件对的路径。
        """
        file_pairs = []
        for folder_name in os.listdir(self.data_root):
            folder_path = os.path.join(self.data_root, folder_name)
            if not os.path.isdir(folder_path):
                continue

            # 获取文件夹中的所有 X 和 Y 文件
            x_files = sorted([f for f in os.listdir(folder_path) if f.startswith('X_')])
            y_files = sorted([f for f in os.listdir(folder_path) if f.startswith('Y_')])

            # 确保 X 和 Y 文件一一对应
            for x_file, y_file in zip(x_files, y_files):
                x_path = os.path.join(folder_path, x_file)
                y_path = os.path.join(folder_path, y_file)
                file_pairs.append((x_path, y_path))

        return file_pairs

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        """
        按需加载单个 (X, Y) 对。
        """
        x_path, y_path = self.file_pairs[idx]
        x = np.load(x_path)  # 加载 X 数据
        y = np.load(y_path)  # 加载 Y 数据

        # 转换为 PyTorch 张量
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        return x, y

def load_data():
    # Load your dataset here
    data_root = Config.DATA_PATH
    dataset = CustomDataset(data_root)

    # X = data['features']
    # y = data['labels']
    
    # # Split into train, validation, and test sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # # Normalize data
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_val = scaler.transform(X_val)
    # X_test = scaler.transform(X_test)
    
    # # Create datasets
    # train_dataset = CustomDataset(X_train, y_train)
    # val_dataset = CustomDataset(X_val, y_val)
    # test_dataset = CustomDataset(X_test, y_test)
    
    # # Create dataloaders
    # train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    # 定义划分比例
    train_ratio = 0.7
    val_ratio = 0.2
    test_ratio = 0.1

    # 计算划分的样本数量
    total_samples = len(dataset)
    train_samples = int(total_samples * train_ratio)
    val_samples = int(total_samples * val_ratio)
    test_samples = total_samples - train_samples - val_samples

    # 使用 random_split 划分数据集
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_samples, val_samples, test_samples])

    # 创建 DataLoader
    batch_size = 32
    num_workers = 4

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 测试 DataLoader
    # for batch_x, batch_y in train_loader:
    #     print("Train Batch X shape:", batch_x.shape)
    #     print("Train Batch Y shape:", batch_y.shape)
    #     break

    # for batch_x, batch_y in val_loader:
    #     print("Validation Batch X shape:", batch_x.shape)
    #     print("Validation Batch Y shape:", batch_y.shape)
    #     break

    # for batch_x, batch_y in test_loader:
    #     print("Test Batch X shape:", batch_x.shape)
    #     print("Test Batch Y shape:", batch_y.shape)
    #     break
    
    return train_loader, val_loader, test_loader