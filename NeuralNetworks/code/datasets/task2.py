import os
import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from NeuralNetworks.code.layers.fullyconnected import FC
from NeuralNetworks.code.layers.convolution2d import Conv2D
from NeuralNetworks.code.layers.maxpooling2d import MaxPool2D
from NeuralNetworks.code.activations.activations import get_activation
from NeuralNetworks.code.losses.meansquarederror import MeanSquaredError
from NeuralNetworks.code.losses.binarycrossentropy import BinaryCrossEntropy
from NeuralNetworks.code.models.model import Model
from NeuralNetworks.code.optimizers.adam import Adam
from NeuralNetworks.code.optimizers.gradientdescent import GD

df = pd.read_csv("mnist_dataset.csv")
num_rows, num_cols = df.shape

# Separate features (pixels) and labels
X = df.drop('label', axis=1)  # Features (pixel columns)
y = df['label']  # Labels

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.values.reshape(-1, 28, 28, 1)
X_test = X_test.values.reshape(-1, 28, 28, 1)

arch = {
    'Conv1': Conv2D(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), name='Conv1'),
    'MaxPool1': MaxPool2D(kernel_size=(2, 2), stride=(2, 2), mode='max'),
    'Conv2': Conv2D(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), name='Conv2'),
    'MaxPool2': MaxPool2D(kernel_size=(2, 2), stride=(2, 2), mode='max'),
    'FC1': FC(input_size=3136, output_size=128, name='FC1'),
    'Activation1': get_activation('relu'),
    'FC2': FC(input_size=128, output_size=64, name='FC2'),
    'Activation2': get_activation('relu'),
    'FC3': FC(input_size=64, output_size=1, name='FC3'),
    'Activation3': get_activation('sigmoid'),  # Use sigmoid activation for binary classification
}

criterion = BinaryCrossEntropy()
optimizer = Adam(arch, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)

model = Model(arch, criterion, optimizer)
