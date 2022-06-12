#simple single hidden layer classifier implemnetation with RELU
#Isabelle Hurley
#3.5.21
#please note the inline citations present as this work was created with referenced to that of many others 

# import matplotlib
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
from preprocessor import train 


class FeedForwardNetwork(nn.Module):
    # structural influence and layout courtesy of Dr. Agnieszka Ławrynowicz class materials: https://colab.research.google.com/drive/1mcB0tftQihL3iTKSBxtGOEzuc_34VQGa?usp=sharing '''
    def __init__(self):
        super(FeedForwardNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

        
      
def main():
    #setup model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))
        
    model = FeedForwardNetwork().to(device)
    print(model)
    opt = optim.Adam(model.parameters())
    cost = torch.nn.CrossEntropyLoss()
    mnist_real_train = MNIST_Dataset("MNIST/processed/training.pt")
    mnist_train, mnist_validation = data.random_split(mnist_real_train, (48000, 12000))

    #train model 
    opt = optim.Adam(model.parameters())
    cost = torch.nn.CrossEntropyLoss()
    train(mnist_train, mnist_validation,model,cost,opt,n_epoch=100)
    #individual model tests
    path = input("Please enter a filepath \n > ")
    while (path != "exit"):
        img = Image.open(path)
        transform = transforms.ToTensor()
        tensor = transform(img)
        model.eval()              # turn the model to evaluate mode
        with torch.no_grad():     # does not calculate gradient
            class_index = model(tensor).argmax()   #gets the prediction for the image's class
        print("Classifier:" + str(class_index.item()))
        path = input("Please enter a filepath \n > ")
if __name__ == "__main__":
    main()