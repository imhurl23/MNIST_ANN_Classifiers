from preprocessor.py import MNIST_Dataset
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

import torch
import sklearn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image


'''function to compute accuracy between model probability predictions and expected class value inspired by '''
def compute_accuracy(logits, expect):
    clasPred = logits.argmax(dim=1)
    return (clasPred == expect).type(torch.float).mean()


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

        # #early stopping check 
        # curr_loss = validation(model,device,)


        model.eval()
        loader = data.DataLoader(train_data, batch_size=len(valid_data), shuffle=False)
        X, y = next(iter(loader))
        X = torch.reshape(X,[len(valid_data),1,28,28])
        logits = model(X.float())
        acc = compute_accuracy(logits, y)
        acc_values.append(acc)


class SimpleFeedForwardNetwork(nn.Module):
    def __init__(self):
        super(SimpleFeedForwardNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 10),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_stack(x)
        return logits

def main():
    #setup model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))
        
    model = SimpleFeedForwardNetwork().to(device)
    print(model)
    opt = optim.Adam(model.parameters())
    cost = torch.nn.CrossEntropyLoss()
    mnist_real_train = MNIST_Dataset("MNIST/processed/training.pt")
    mnist_train, mnist_validation = data.random_split(mnist_real_train, (48000, 12000))

    #train model 
    opt = optim.Adam(model.parameters())
    cost = torch.nn.CrossEntropyLoss()

    #individual model tests
    path = input("Please enter a filepath \n > ")
    img = Image.open(path)
    transform = transforms.ToTensor()
    tensor = transform(img)

    model.eval()              # turn the model to evaluate mode
    with torch.no_grad():     # does not calculate gradient
        class_index = model(tensor).argmax()   #gets the prediction for the image's class

if __name__ == "__main__":
    main()