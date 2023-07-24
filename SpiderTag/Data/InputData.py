from torch.utils.data import Dataset
import torch
import numpy as np

class train_dataWord2Vec(Dataset):
    def __init__(self):  # __init__是初始化该类的一些基础参数
        x = np.load('..//myData//word2vec//train_des_vec.npy')
        y = np.load('..//myData//word2vec//train_tag50_vec.npy')
        z = np.load('..//myData//word2vec//train_tag51_vec.npy')
        self.x_data = torch.from_numpy(x)
        self.y_data = torch.from_numpy(y)
        self.z_data = torch.from_numpy(z[:, -1:])
        self.all_tag = torch.from_numpy(z)
        self.len = len(x)

    def __len__(self):  # 返回整个数据集的大小
        return self.len

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        return self.x_data[index], self.y_data[index], self.z_data[index], self.all_tag[index]

class val_dataWord2Vec(Dataset):
    def __init__(self):  # __init__是初始化该类的一些基础参数
        x = np.load('..//myData//word2vec//val_des_vec.npy')
        y = np.load('..//myData//word2vec//val_tag50_vec.npy')
        z = np.load('..//myData//word2vec//val_tag51_vec.npy')
        self.x_data = torch.from_numpy(x)
        self.y_data = torch.from_numpy(y)
        self.z_data = torch.from_numpy(z[:, -1:])
        self.all_tag = torch.from_numpy(z)
        self.len = len(x)

    def __len__(self):  # 返回整个数据集的大小
        return self.len

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        return self.x_data[index], self.y_data[index], self.z_data[index], self.all_tag[index]

class test_dataWord2Vec(Dataset):
    def __init__(self):  # __init__是初始化该类的一些基础参数
        x = np.load('..//myData//word2vec//test_des_vec.npy')
        y = np.load('..//myData//word2vec//test_tag50_vec.npy')
        z = np.load('..//myData//word2vec//test_tag51_vec.npy')
        self.x_data = torch.from_numpy(x)
        self.y_data = torch.from_numpy(y)
        self.z_data = torch.from_numpy(z[:, -1:])
        self.all_tag = torch.from_numpy(z)
        self.len = len(x)

    def __len__(self):  # 返回整个数据集的大小
        return self.len

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        return self.x_data[index], self.y_data[index], self.z_data[index], self.all_tag[index]

class train_dataWord2Vec128(Dataset):
    def __init__(self):  # __init__是初始化该类的一些基础参数
        x = np.load('..//myData//final_data//train_des_vec128.npy')
        y = np.load('..//myData//final_data//train_tag50_vec.npy')
        z = np.load('..//myData//final_data//train_tag51_vec.npy')
        self.x_data = torch.from_numpy(x)
        self.y_data = torch.from_numpy(y)
        self.z_data = torch.from_numpy(z[:, -1:])
        self.all_tag = torch.from_numpy(z)
        self.len = len(x)

    def __len__(self):  # 返回整个数据集的大小
        return self.len

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        return self.x_data[index], self.y_data[index], self.z_data[index], self.all_tag[index]

class val_dataWord2Vec128(Dataset):
    def __init__(self):  # __init__是初始化该类的一些基础参数
        x = np.load('..//myData//final_data//val_des_vec128.npy')
        y = np.load('..//myData//final_data//val_tag50_vec.npy')
        z = np.load('..//myData//final_data//val_tag51_vec.npy')
        self.x_data = torch.from_numpy(x)
        self.y_data = torch.from_numpy(y)
        self.z_data = torch.from_numpy(z[:, -1:])
        self.all_tag = torch.from_numpy(z)
        self.len = len(x)

    def __len__(self):  # 返回整个数据集的大小
        return self.len

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        return self.x_data[index], self.y_data[index], self.z_data[index], self.all_tag[index]

class test_dataWord2Vec128(Dataset):
    def __init__(self):  # __init__是初始化该类的一些基础参数
        x = np.load('..//myData//final_data//test_des_vec128.npy')
        y = np.load('..//myData//final_data//test_tag50_vec.npy')
        z = np.load('..//myData//final_data//test_tag51_vec.npy')
        self.x_data = torch.from_numpy(x)
        self.y_data = torch.from_numpy(y)
        self.z_data = torch.from_numpy(z[:, -1:])
        self.all_tag = torch.from_numpy(z)
        self.len = len(x)

    def __len__(self):  # 返回整个数据集的大小
        return self.len

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        return self.x_data[index], self.y_data[index], self.z_data[index], self.all_tag[index]