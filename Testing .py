#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from preprocessor import MNIST_Dataset
from preprocessor import compute_accuracy
from preprocessor import train
from preprocessor import train2

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


# In[2]:


batch_size = 64
mnist_real_train = MNIST_Dataset("MNIST/processed/training.pt")

mnist_train, mnist_validation = data.random_split(mnist_real_train, (48000, 12000))


train_dataloader = DataLoader(mnist_train, batch_size=batch_size)
validation_dataloader = DataLoader(mnist_validation, batch_size=batch_size)

mnist_test = MNIST_Dataset("MNIST/processed/test.pt")
test_dataloader = DataLoader(mnist_test, batch_size=batch_size)

len(mnist_real_train), len(mnist_train), len(mnist_validation), len(mnist_test)


# In[ ]:





# Classifier 1 

# In[4]:


from Classifier_1 import SimpleFeedForwardNetwork
modelC1 = SimpleFeedForwardNetwork().to(device)
print(modelC1)
opt = optim.Adam(modelC1.parameters())
cost = torch.nn.CrossEntropyLoss()
train(mnist_train, mnist_validation,modelC1,cost,opt,n_epoch = 50)


# In[19]:


#this functionality was inspiried by the example in pytorch documentation https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
corr = 0
tot = 0 
with torch.no_grad(): 
    for data in test_dataloader: 
        images, labels = data
        outputs = modelC1(images)
        _, predicted = torch.max(outputs.data, 1)
        tot += labels.size(0)
        corr += (predicted == labels).sum().item()
print(corr, "out of", tot, "correct")    
print((corr/tot)*100,'%')


# Classifier 2 

# In[32]:


from Classifier_2 import FeedForwardNetwork
modelC2 = FeedForwardNetwork().to(device)
print(modelC2)
opt = optim.Adam(modelC2.parameters())
cost = torch.nn.CrossEntropyLoss()
train(mnist_train, mnist_validation,modelC2,cost,opt,n_epoch = 100)


# In[33]:


corr = 0
tot = 0 
with torch.no_grad(): 
    for data in test_dataloader: 
        images, labels = data
        outputs = modelC2(images)
        _, predicted = torch.max(outputs.data, 1)
        tot += labels.size(0)
        corr += (predicted == labels).sum().item()
print(corr, "out of", tot, "correct")    
print((corr/tot)*100,'%')


# Classifier 3 

# In[34]:


from Classifier_3 import MultiLayerFeedForwardNetwork
modelC3 = MultiLayerFeedForwardNetwork().to(device)
print(modelC3)
opt = optim.SGD(modelC3.parameters(), lr=0.1, momentum=0.9)
cost = torch.nn.CrossEntropyLoss()
train(mnist_train, mnist_validation,modelC3,cost,opt,n_epoch = 100)


# In[35]:


corr = 0
tot = 0 
with torch.no_grad(): 
    for data in test_dataloader: 
        images, labels = data
        outputs = modelC2(images)
        _, predicted = torch.max(outputs.data, 1)
        tot += labels.size(0)
        corr += (predicted == labels).sum().item()
print(corr, "out of", tot, "correct")    
print((corr/tot)*100,'%')


# Classifier 4

# In[27]:


from Classifier_4 import ConvNeuralNetwork
modelC4 = ConvNeuralNetwork().to(device)
print(modelC4)
opt = optim.Adam(modelC4.parameters())
cost = torch.nn.CrossEntropyLoss()
train2(mnist_train, mnist_validation,modelC4,cost,opt,n_epoch=100)


# In[28]:


corr = 0
tot = 0 
with torch.no_grad(): 
    for data in test_dataloader: 
        images, labels = data
        outputs = modelC4(images)
        _, predicted = torch.max(outputs.data, 1)
        tot += labels.size(0)
        corr += (predicted == labels).sum().item()
print(corr, "out of", tot, "correct")    
print((corr/tot)*100,'%')


# Classifier 5

# In[29]:


from Classifier_5 import CNNDropNet
modelC5 = CNNDropNet().to(device)
print(modelC5)
opt = optim.Adam(modelC5.parameters())
cost = torch.nn.CrossEntropyLoss()
train2(mnist_train, mnist_validation,modelC5,cost,opt,n_epoch=100)


# In[23]:


corr = 0
tot = 0 
with torch.no_grad(): 
    for data in test_dataloader: 
        images, labels = data
        outputs = modelC5(images)
        _, predicted = torch.max(outputs.data, 1)
        tot += labels.size(0)
        corr += (predicted == labels).sum().item()
print(corr, "out of", tot, "correct")    
print((corr/tot)*100,'%')


# K-Fold testing implementation 

# In[9]:


from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, ConcatDataset


# In[13]:




