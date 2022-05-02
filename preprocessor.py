import matplotlib
import numpy as np
import matplotlib.pyplot as plt

import torch
import sklearn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader


class MNIST_Dataset(Dataset): 
    '''initializer method to load the data'''
    def __init__(self,filename): 
        self.data = torch.load(filename)[0]
        self.labels = torch.load(filename)[1]
    '''helper method to com data size'''
    def __len__(self):
        return len(self.labels)
    '''helper method to return elems by index'''
    def __getitem__(self, index):
        return self.data[index], self.labels[index]

def train(train_data,valid_data, model, cost, n_epoch = 100, batch_size = 64):
    loss_values = []
    acc_values = []
    batch_size = 64
    n_epoch = n_epoch 

    for epoch in range(n_epoch):
        model.train()
        loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        
