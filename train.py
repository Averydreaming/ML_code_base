import torch
import torch.nn.functional as F

from copy import deepcopy
from tqdm import tqdm
import argparse
import numpy as np


from MLP import MLP
from CNN import CNN
from RNN import RNN
from LSTM import LSTM

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="choose the model from MLP, CNN, RNN, LSTM",
        default="MLP",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2000,
        help="number of epochs of training",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="learning rate",
    )
    
    args = parser.parse_args()
    return args

def get_model(args):
    if args.model == "MLP":
        ModelCalss = MLP
    elif args.model == "CNN":
        ModelCalss = CNN
    elif args.model == "RNN":
        ModelCalss = RNN
    elif args.model == "LSTM":
        ModelCalss = LSTM
    else:
        raise ValueError("no such model")
    return ModelCalss(args.epochs, args.learning_rate)

# 读取数据
def read_data():
    #用for循环读取数据
    data_list = []
    label_list = []
    for i in range(1, 13):
        data = np.load('dataset/' + str(i) + '/data.npy')
        label = np.load('dataset/' + str(i) + '/label.npy')
        #转换data数据为两维
        data = data.reshape(data.shape[0], -1)
        data_list.append(data)
        label_list.append(label)
    #print("data_list",data_list)
    #print("label_list",label_list)
    return data_list, label_list

# 留一交叉验证
def leave_one_cross_validation(data_list, label_list, i):
    if torch.cuda.is_available():
        device = torch.device("cuda")  
    else:
        device = torch.device("cpu")
    # 用for循环读取数据
    train_data_list = []
    train_label_list = []
    val_data = data_list[i]
    val_label = label_list[i]
    for j in range(len(data_list)):
        if j != i:
            train_data_list.append(data_list[j])
            train_label_list.append(label_list[j])
    # 把好多个numpy组成的list合并为一个numpy    
    train_data = np.concatenate(train_data_list, axis=0)
    train_label = np.concatenate(train_label_list, axis=0)
    # 转换为tensor
    train_data = torch.from_numpy(train_data).float().to(device)
    train_label = torch.from_numpy(train_label).long().to(device)
    val_data = torch.from_numpy(val_data).float().to(device)
    val_label = torch.from_numpy(val_label).long().to(device)
    return train_data, train_label, val_data, val_label

def main():
    args = get_args()
    # 预处理数据
    # 读取数据
    data_list, label_list=read_data()
    accs=[]
    # 分别对1到12的数据集留一交叉验证
    for i in range(12):
        train_data, train_label, val_data, val_label=leave_one_cross_validation(data_list, label_list, i)
        # 训练模型
        model = get_model(args)
        model,val_acc = model.fit(train_data, train_label, val_data, val_label)
        accs.append(val_acc)
        torch.save(model, args.model+"model.pth")
        
    print(f"model: {args.model}")
    print(f"accs={accs}")
    print(f"mean_acc={np.mean(accs)}")
    print(f"std_acc={np.std(accs)}")


if __name__ == "__main__":
    main()
