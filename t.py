import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os

# 假设文档存储在名为 data_dir 的文件夹中
data_dir = '/path/to/data_dir'

# 假设标签存储在一个单独的文件中，每个文件的标签与相应的特征文件一一对应
# 例如，Y 的标签数据可以存储在 CSV 格式的文件中，每一行代表对应文档的标签

class CustomDataset(Dataset):
    def __init__(self, data_dir, label_file, batch_size=64):
        """
        data_dir: 存放特征文件的文件夹
        label_file: 标签文件，假设包含所有文档的标签（CSV 格式）
        batch_size: 数据加载时每个批次的大小
        """
        self.data_dir = data_dir
        self.label_file = label_file
        self.batch_size = batch_size
        
        # 加载标签
        self.labels = pd.read_csv(label_file).values  # 假设标签存储在 CSV 格式文件中
        
        # 获取特征文件的路径
        self.feature_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        self.feature_files.sort()  # 确保文件按顺序排列
        
    def __len__(self):
        # 数据集的长度，等于特征文件的数量
        return len(self.feature_files)
    
    def __getitem__(self, idx):
        # 读取特定文件的特征
        feature_file = os.path.join(self.data_dir, self.feature_files[idx])
        X = pd.read_csv(feature_file).values  # 读取特征数据
        
        # 获取对应的标签
        y = self.labels[idx]
        
        # 转换为 PyTorch 张量
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    
# 创建数据集实例
dataset = CustomDataset(data_dir='/path/to/data_dir', label_file='/path/to/label_file.csv', batch_size=64)

# 使用 DataLoader 加载数据
data_loader = DataLoader(dataset, batch_size=dataset.batch_size, shuffle=True)


class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # 线性回归模型，输入维度是特征数，输出是一个标签
        
    def forward(self, x):
        return self.linear(x)

# 假设每个文件的特征维度是 X 的列数
# 我们从第一个文件读取并获取特征维度
sample_file = os.path.join(data_dir, dataset.feature_files[0])
sample_data = pd.read_csv(sample_file)
input_dim = sample_data.shape[1]  # 获取特征维度

# 初始化模型
model = LinearRegressionModel(input_dim=input_dim)

# 损失函数和优化器
criterion = nn.MSELoss()  # 均方误差损失函数
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 10  # 假设训练 10 个 epochs

for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in data_loader:
        optimizer.zero_grad()  # 清除之前的梯度
        
        # 前向传播
        outputs = model(inputs)
        
        # 计算损失
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        
        # 更新模型参数
        optimizer.step()
        
        running_loss += loss.item()
    
    # 打印每个 epoch 的平均损失
    avg_loss = running_loss / len(data_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# 保存模型
torch.save(model.state_dict(), 'linear_model.pth')
