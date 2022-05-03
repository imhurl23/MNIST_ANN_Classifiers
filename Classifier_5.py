#simple single hidden layer classifier implemnetation 
#Isabelle Hurley


# import matplotlib
from locale import normalize
import numpy as np
import matplotlib.pyplot as plt
import sys
import torch
import sklearn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image
from preprocessor import MNIST_Dataset
from preprocessor import compute_accuracy
from preprocessor import train2

class CNNDropNet(nn.Module):
     # structural influence and layout courtesy of Dr. Agnieszka Ławrynowicz class materials: https://colab.research.google.com/drive/1mcB0tftQihL3iTKSBxtGOEzuc_34VQGa?usp=sharing 
     # as well as Ayush Thakur's Tutorial: https://wandb.ai/authors/ayusht/reports/Implementing-Dropout-in-PyTorch-With-Example--VmlldzoxNTgwOTE 
    def __init__(self):
        super(CNNDropNet, self).__init__()
        layers =  [nn.Conv2d(1, 5, 3, padding=1), 
        nn.LeakyReLU(),
        nn.MaxPool2d(3, padding=1),
        nn.Flatten(), 
        nn.Dropout(p=0.2),
        nn.Linear(500, 10)] 
        self.layers = nn.Sequential(*layers)
        
        

    def forward(self, x):
        x = self.layers(x)
        logits = x
        return logits
        



def main():
    #setup model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))
    model = CNNDropNet().to(device)
    print(model)


    print(model)
    opt = optim.Adam(model.parameters())
    cost = torch.nn.CrossEntropyLoss()
    mnist_real_train = MNIST_Dataset("MNIST/processed/training.pt")
    mnist_train, mnist_validation = data.random_split(mnist_real_train, (48000, 12000))

    #train model 
    train2(mnist_train, mnist_validation,model,cost,opt,n_epoch=100)
    #individual model tests
    path = input("Please enter a filepath \n > ")
    while (path != "exit"):
        img = Image.open(path)
        transform = transforms.ToTensor()
        tensor = transform(img)
        tensor = torch.reshape(tensor,[1,1,28,28])        
        m = torch.mean(tensor)
        std = torch.std(tensor)
        normalize = transforms.Normalize(m,std)
        tensor = normalize(tensor)   
      

        model.eval()              # turn the model to evaluate mode
        with torch.no_grad():     # does not calculate gradient
            class_index = model(tensor).argmax()   #gets the prediction for the image's class
        print("Classifier:" + str(class_index.item()))
        path = input("Please enter a filepath \n > ")
if __name__ == "__main__":
    main()