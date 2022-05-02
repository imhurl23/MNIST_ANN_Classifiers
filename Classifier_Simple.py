import preprocessor.py 
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
        class_index = model(single_image).argmax()   #gets the prediction for the image's class

if __name__ == "__main__":
    main()