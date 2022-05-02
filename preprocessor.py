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

'''function to compute accuracy between model probability predictions and expected class value'''
def compute_accuracy(logits, expect):
    clasPred = logits.argmax(dim=1)
    return (clasPred == expect).type(torch.float).mean()

'''generalized training function for iterative gradient decent training method '''
def train(train_data,valid_data, model, cost, opt, n_epoch = 100, batch_size = 64):
    loss_values = []
    acc_values = []
    batch_size = 64
    n_epoch = n_epoch 

    #early stopping functions
    prev_loss = 10000 #start out of range
    patience = 2 #num epochs to wait 
    triggers = 0 #count depreciation occurances 

    for epoch in range(n_epoch):
        model.train()
        loader = data.DataLoader(train_data, valid_data, batch_size=batch_size, shuffle=True)
        epoch_loss = []
        for X_batch, y_batch in loader:
            X_batch = torch.reshape(X_batch,(batch_size,1,28,28) )
            opt.zero_grad()    
            logits = model(X_batch.float())
            loss = cost(logits, y_batch)
            loss.backward()
            opt.step()        
            epoch_loss.append(loss.detach())
        loss_values.append(torch.tensor(epoch_loss).mean())
        model.eval()
        loader = data.DataLoader(train_data, batch_size=len(valid_data), shuffle=False)
        X, y = next(iter(loader))
        X = torch.reshape(X,[len(valid_data),1,28,28])
        logits = model(X.float())
        acc = compute_accuracy(logits, y)
        acc_values.append(acc)
