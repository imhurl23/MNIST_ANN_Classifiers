# import matplotlib
import numpy as np
import matplotlib.pyplot as plt

import torch
import sklearn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


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
        dat = torch.reshape(self.data[index].float(),[1,28,28])
        lab = self.labels[index]
        mean = dat.mean()
        std = dat.std()
        transform = transforms.Normalize(mean,std)
        dat = transform(dat)
        return dat, self.labels[index]

'''function to compute accuracy between model probability predictions and expected class value inspired by '''
def compute_accuracy(logits, expect):
    clasPred = logits.argmax(dim=1)
    return (clasPred == expect).type(torch.float).mean()

'''generalized training function for iterative gradient decent training method with Early Stopping Method implementation'''
def train(train_data,valid_data, model, cost, opt, n_epoch = 100, batch_size = 64):
    loss_values = []
    acc_values = []
    batch_size = 64
    n_epoch = n_epoch 

    #early stopping functions
    prev_loss = 10000 #start out of range
    patience = 3 #num epochs to wait 
    triggers = 0 #count depreciation occurances 

    for epoch in range(n_epoch):
        model.train()
        loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
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

        #early stopping check 
        curr_loss = validation(model,valid_data,cost,batch_size)
        print("current loss", curr_loss)
        if curr_loss > prev_loss:
            triggers+=1
            print("early stop trigger:", triggers)
            if triggers > patience: 
                #STOP EARLY 
                print("early stopping at epoch:", epoch)
                return model
        else: 
            triggers = 0 
        prev_loss = curr_loss
        

        model.eval()
        loader = data.DataLoader(train_data, batch_size=len(valid_data), shuffle=False)
        X, y = next(iter(loader))
        X = torch.reshape(X,[len(valid_data),1,28,28])
        logits = model(X.float())
        acc = compute_accuracy(logits, y)
        acc_values.append(acc)

'''generalized training function for iterative gradient decent training method '''
def train2(train_data,valid_data, model, cost, opt, n_epoch = 100, batch_size = 64):
    loss_values = []
    acc_values = []
    batch_size = 64
    n_epoch = n_epoch 

    

    for epoch in range(n_epoch):
        model.train()
        loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
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


'''validation function developed with inspiration from GFK: Training Neural Networks with Validation using PyTorch and
https://pythonguides.com/pytorch-early-stopping/'''
def validation(model, valid_data, loss_funct, batch_size):
    valid_loader = data.DataLoader(valid_data, batch_size=batch_size, shuffle=True)
    model.eval()
    loss_total = 0

    # Test validation data
    with torch.no_grad():
        for X,label in valid_loader:  
            outputs = model(X.view(X.shape[0], -1).float())
            loss = loss_funct(outputs, label)
            loss_total += loss.item()


    return loss_total / len(valid_loader)