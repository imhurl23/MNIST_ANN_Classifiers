
# Project Purpose
This project outlines 5 different models for digit classification of the MNIST dataset 
* each model can be invoked and trained by running python Classifier_{SUFFIX}.py
* *note*: This project was completed as an extention of coursework for CSC3022 at the University of Cape Town. 

# How to run 
make 
source ./venv/bin/activate
python classifier_{SUFFIX}.py

# Files
* preprocessor.py holds the dataset class and the generalized methods for training which can be invoked specially by the model. this preprocessor handles normalization as well as the splitting of training and testing sets. 

* testing.ipynb is a jupyter notebook which was used in development for ease of testing. testing.py can is esencially equivilent but must be run repeatedly in order to conduct multiple rounds of testing. The jupetyer notebook environment was chosen for this purpose 

* practice.ipynb is the development file for the rest of the project. Due to the ease of training in the notebook environment this file was used. It is now essencially depreciated but an important piece of the git history and information migration. 

## The Models 
- Classifier_1.py a simple single hidden layer classifier implemnetation with Tahn 
- Classifier_2.py a simple single hidden layer classifier implemnetation with RELU
- Classifier_3.py a multilayered network employing SGD 
- Classifier_4.py a very simple CNN with a small kernel 
- Classifier_5.py a simple CNN with dropout 

## Resources 
Many resources were consulted in the creation of this project. In addition to those listed here there are some in code citations for places in which ideas were drawn very heavily and so direct affilation of credit was nessacary. 
https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-k-fold-cross-validation-with-pytorch.md
https://www.kaggle.com/code/cdeotte/how-to-choose-cnn-architecture-mnist/notebook
https://medium.com/@nutanbhogendrasharma/pytorch-convolutional-neural-network-with-mnist-dataset-4e8a4265e118
https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6#:~:text=tanh%20is%20also%20like%20logistic,sigmoidal%20(s%20%2D%20shaped).&text=The%20advantage%20is%20that%20the,zero%20in%20the%20tanh%20graph.

